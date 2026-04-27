# type: ignore
# franka interface is referenced from https://github.com/JeanElsner/panda-py
import logging
import time
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional

import numpy as np
from i2rt.robots.robot import Robot
from i2rt.utils.utils import RateRecorder

try:
    import panda_py
    from panda_py import controllers
except ImportError as exc:
    raise ImportError(
        "Franka support requires panda_py. Install with: `uv pip install -e .[franka_panda]` "
        "(or `uv sync --extra franka_panda`)."
    ) from exc
from scipy.spatial.transform import Rotation as R

from robots_realtime.robots.utils import Rate

logger = logging.getLogger(__name__)

###############################################################################
# Single Kp, Kd for both position and orientation in OSC
###############################################################################
# Try lowering orientation P and increasing orientation D ratio
KP_pos = 150.0
KD_pos = 30.0
KP_ori = 170.0
KD_ori = 25.0

# Define these as constants or class attributes
MAX_POS_ERR = 0.08  # Caps max force/velocity for translation
MAX_ORI_ERR = 0.2  # Caps max torque/velocity for rotation

KP_6D = np.array([KP_pos] * 3 + [KP_ori] * 3)
KD_6D = np.array([KD_pos] * 3 + [KD_ori] * 3)

# Joint impedance for null space (example, can be changed)
# Kp_null = np.array([50.0, 50.0, 50.0, 50.0, 40.0, 25.0, 25.0])
# Kp_null = np.array([30.0, 30.0, 25.0, 25.0, 20.0, 10.0, 10.0])
# Kp_null = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
Kp_null = np.array([30.0, 30.0, 25.0, 25.0, 15.0, 10.0, 10.0])
damping_ratio = 2.0
Kd_null = damping_ratio * 2.0 * np.sqrt(Kp_null)
# Kp_null = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
# damping_ratio = 2.0
# Kd_null = damping_ratio * 2.0 * np.sqrt(Kp_null)

GRIPPER_DEFAULT_SPEED = 10.0
GRIPPER_INITIAL_FORCE = 1.0
GRIPPER_ACTIVE_FORCE = 4.0
GRIPPER_MAX_WIDTH = 0.1
GRIPPER_MOVE_THRESHOLD = 0.055
GRIPPER_COMMAND_EPSILON = 1e-3
GRIPPER_UPDATE_TIMEOUT_S = 0.05


def orientation_error(current_rot: np.ndarray, desired_rot: np.ndarray) -> np.ndarray:
    """
    Compute orientation error in axis-angle form (3-vector).
    """
    q_current = R.from_matrix(current_rot).as_quat()
    q_desired = R.from_matrix(desired_rot).as_quat()
    q_err = R.from_quat(q_desired) * R.from_quat(q_current).inv()
    return q_err.as_rotvec()


class FrankaPanda(Robot):
    def __init__(
        self,
        host_name: str = "172.16.0.2",
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
        enable_gripper: bool = False,
    ) -> None:
        """Initialize the Franka Panda robot arm without gripper.

        Args:
            host_name: str
                The IP address of the robot controller
            username: str
                The username for robot controller login
            password: str
                The password for robot controller login
            enable_gripper: bool
                Whether to enable control for a default Franka Panda gripper
        """
        # Store initialization parameters for reinit
        self._init_params = {
            "host_name": host_name,
            "username": username,
            "password": password,
            "name": name,
        }
        self._initialize(host_name, username, password, name, enable_gripper)

    def _initialize(
        self,
        host_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
        enable_gripper: bool = False,
    ) -> None:
        """Internal method to handle the actual initialization."""

        self._joint_state_saver = None
        self.name = name or "franka"
        self.host_name = host_name

        if username and password:
            self.desk = panda_py.Desk(host_name, username, password)
            self.desk.activate_fci()

        try:
            self.interface = panda_py.Panda(host_name)
        except RuntimeError as exc:
            msg = str(exc)
            if "Incompatible library version" in msg:
                raise RuntimeError(
                    f"Failed to connect to Franka FCI at {host_name}: {msg}. "
                    "This usually means the loaded libfranka/panda_py build does not match robot server protocol. "
                    "Reinstall the project Franka extra (`uv pip install -e .[franka_panda]`), "
                    "remove conflicting system/pip panda_py installs, and verify the imported panda_py path/version. "
                    "If multiple libfranka versions exist on the machine, ensure the runtime links against the compatible one. "
                    "See Franka compatibility table: https://frankaemika.github.io/docs/compatibility.html"
                ) from exc

            raise RuntimeError(
                f"Failed to connect to Franka FCI at {host_name}: {msg}. "
                "If FCI is already enabled on the robot, verify that this host is the allowed FCI client, "
                "the robot is in FCI mode, and no other process is connected. "
                "You can also pass Desk credentials (username/password) in robot config so this driver can call activate_fci()."
            ) from exc
        self.state = self.interface.get_state()
        self.fk = panda_py.fk
        self._num_dofs = 7
        self._stop_event = Event()

        if enable_gripper:
            self.gripper = panda_py.libfranka.Gripper(host_name)
            self.gripper.grasp(0.0, GRIPPER_DEFAULT_SPEED, GRIPPER_INITIAL_FORCE)
            self.gripper.grasp(GRIPPER_MAX_WIDTH, GRIPPER_DEFAULT_SPEED, GRIPPER_INITIAL_FORCE)
            self._num_dofs += 1
            self._gripper_lock = Lock()
            self._gripper_update_event = Event()
            self._gripper_target_width = self.gripper.read_once().width
            self._last_gripper_command = self._gripper_target_width
            self._last_gripper_state = self._last_gripper_command
            self._gripper_thread = Thread(
                target=self._gripper_command_loop,
                name="gripper_command_loop",
                daemon=True,
            )
            self._gripper_thread.start()

        # reduce collision sensitivity for enabling contact rich behavoir
        self.torque_limit = 26.5
        self.interface.get_robot().set_collision_behavior([100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6)
        self.model = self.interface.get_model()
        self.frame = panda_py.libfranka.Frame.kFlange

        joint_stiffness = np.array([300, 300, 300, 300, 250, 150, 150], dtype=np.float64)
        joint_damping = np.array([40, 40, 40, 20, 20, 20, 15], dtype=np.float64)
        self.ctrl = controllers.JointPosition(
            stiffness=joint_stiffness,
            damping=joint_damping,
            filter_coeff=1.0
        )

        # self.state는 이미 위에서 self.interface.get_state()로 받아둔 상태
        q0 = np.asarray(self.state.q, dtype=np.float64).copy()

        self._cmd_lock = Lock()
        self._state_lock = Lock()

        if enable_gripper:
            self._joint_cmd = np.concatenate([q0, [float(self._last_gripper_state)]])
        else:
            self._joint_cmd = q0

        joint_stiffness = np.array([300, 300, 300, 300, 250, 150, 150], dtype=np.float64)
        joint_damping = np.array([40, 40, 40, 20, 20, 20, 15], dtype=np.float64)

        self.ctrl = controllers.JointPosition(
            stiffness=joint_stiffness,
            damping=joint_damping,
            filter_coeff=1.0,
        )

        print("[INIT DEBUG] set initial control before start_controller", flush=True)
        self.ctrl.set_control(q0)

        print("starting controller", flush=True)
        self.interface.start_controller(self.ctrl)
        print("controller started", flush=True)

        self.ctrl_thread_start_time = time.time()

        # Temporarily disable RateRecorder in execution-mode debugging path.
        self._update_rate = None
        self._slow_set_control_count = 0
        self._slow_get_state_count = 0
        self._state_lock_timeout_count = 0

        self._server_thread = Thread(target=self.run, name="control_loop", daemon=True)
        self._server_thread.start()

        print("[INIT DEBUG] franka init done", flush=True)

    def __repr__(self) -> str:
        return f"FrankaPanda(name={self.name}, host_name={self.host_name})"

    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "kp_6d": KP_6D,
            "kd_6d": KD_6D,
            "kp_null": Kp_null,
            "kd_null": Kd_null,
            "damping_ratio": damping_ratio,
        }

    def run(self) -> None:
        print("[CONTROL LOOP] entered", flush=True)

        try:
            period = 0.01
            next_t = time.perf_counter()

            print("[CONTROL LOOP] RateRecorder bypassed", flush=True)

            loop_i = 0
            while not self._stop_event.is_set():
                loop_i += 1

                with self._cmd_lock:
                    if hasattr(self, "gripper"):
                        joint_cmd = np.asarray(self._joint_cmd[:-1], dtype=np.float64).copy()
                    else:
                        joint_cmd = np.asarray(self._joint_cmd, dtype=np.float64).copy()

                # Single writer path for the controller command.
                t_set0 = time.perf_counter()
                self.ctrl.set_control(joint_cmd)
                dt_set = time.perf_counter() - t_set0
                if dt_set > 0.02:
                    self._slow_set_control_count += 1
                    if self._slow_set_control_count % 20 == 1:
                        print(
                            f"[FRANKA DEBUG] slow set_control dt={dt_set*1000:.1f}ms "
                            f"(count={self._slow_set_control_count})",
                            flush=True,
                        )

                # Refresh state snapshot in control thread.
                # Important: do NOT hold _state_lock while calling get_state()
                # because get_state() can block in execution mode.
                t_state0 = time.perf_counter()
                new_state = self.interface.get_state()
                dt_state = time.perf_counter() - t_state0
                if dt_state > 0.02:
                    self._slow_get_state_count += 1
                    if self._slow_get_state_count % 20 == 1:
                        print(
                            f"[FRANKA DEBUG] slow get_state dt={dt_state*1000:.1f}ms "
                            f"(count={self._slow_get_state_count})",
                            flush=True,
                        )
                with self._state_lock:
                    self.state = new_state

                if loop_i % 50 == 1:
                    print(
                        "[JOINT POSITION DEBUG]",
                        "cmd=", np.round(joint_cmd, 5),
                        flush=True,
                    )

                next_t += period
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()

        except BaseException as exc:
            logger.exception("Franka control_loop crashed")
            print("[CONTROL LOOP] crashed:", repr(exc), flush=True)
            raise


    def enable_arm(self) -> None:
        self.interface.teaching_mode(False)

    def disable_arm(self) -> None:
        self.interface.teaching_mode(True)

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        if hasattr(self, "gripper"):
            return np.concatenate([self.interface.q, [self._last_gripper_state]])
        return self.interface.q

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert len(joint_pos) == self._num_dofs, (
            f"Joint position array length mismatch. num_dofs: {self._num_dofs}, joint_pos: {len(joint_pos)}."
        )

        if self._update_rate is not None:
            self._update_rate.track()

        with self._cmd_lock:
            self._joint_cmd = np.asarray(joint_pos, dtype=np.float64).copy()

        if hasattr(self, "gripper"):
            arm_cmd = np.asarray(joint_pos[:-1], dtype=np.float64)
            self._submit_gripper_width(joint_pos[-1])
        else:
            arm_cmd = np.asarray(joint_pos, dtype=np.float64)

        # NOTE:
        # ``self.ctrl.set_control`` is intentionally *not* called from this
        # command path.  The dedicated control thread in ``run()`` is the only
        # writer to the controller state and drains ``self._joint_cmd`` at a
        # fixed rate. Calling ``set_control`` from both threads can block under
        # load and stall the RobotNode.step() loop (STATUS stays "live" but
        # STEP/PUB Hz stop updating in the TUI).

    def get_observations(self) -> Dict[str, np.ndarray]:
        # Read the latest snapshot cached by the control thread.
        if self._state_lock.acquire(timeout=0.002):
            try:
                state = self.state
            finally:
                self._state_lock.release()
        else:
            # Avoid blocking RobotNode.step(); publish last known snapshot.
            self._state_lock_timeout_count += 1
            if self._state_lock_timeout_count % 100 == 1:
                print(
                    f"[FRANKA DEBUG] get_observations lock-timeout "
                    f"(count={self._state_lock_timeout_count})",
                    flush=True,
                )
            state = self.state
        obs = {
            "joint_pos": state.q
            if not hasattr(self, "gripper")
            else np.concatenate([state.q, [self._last_gripper_state]]),
            "joint_vel": state.dq,
            "joint_eff": state.tau_J,
        }
        return obs

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def close(self) -> None:
        """Safely close the robot by setting all torques to zero and cleaning up resources."""
        # Set torques to zero first
        self.ctrl.set_control(np.zeros(self._num_dofs))
        # Signal the thread to stop and wait for it to finish
        self._stop_event.set()
        if self._server_thread.is_alive():
            self._server_thread.join()
        if hasattr(self, "_gripper_update_event"):
            self._gripper_update_event.set()
        if hasattr(self, "_gripper_thread") and self._gripper_thread.is_alive():
            self._gripper_thread.join()
        # Stop the controller
        self.interface.stop_controller()
        # Logout from desk
        if hasattr(self, "desk") and self.desk is not None:
            self.desk.logout()

    def reinit(self) -> None:
        """Reinitialize the robot by closing existing connection and creating a new one."""
        logger.info(f"Reinitializing franka panda robot {self.name}")
        self.close()
        # Wait a moment to ensure clean shutdown
        import time

        time.sleep(0.01)

        # Reinitialize with stored parameters
        self._initialize(**self._init_params)
        logger.info(f"Robot {self.name} reinitialized")

    def _gripper_command_loop(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._gripper_update_event.wait(timeout=GRIPPER_UPDATE_TIMEOUT_S)
            if self._stop_event.is_set():
                break
            if not triggered:
                continue
            with self._gripper_lock:
                width = self._gripper_target_width
                self._gripper_update_event.clear()
            if width is None:
                continue
            if (
                self._last_gripper_command is not None
                and abs(self._last_gripper_command - width) < GRIPPER_COMMAND_EPSILON
            ):
                continue
            try:
                if width > GRIPPER_MOVE_THRESHOLD:
                    self.gripper.move(width, GRIPPER_DEFAULT_SPEED)
                else:
                    self.gripper.grasp(width, GRIPPER_DEFAULT_SPEED, GRIPPER_ACTIVE_FORCE)

                self._last_gripper_command = width
                self._last_gripper_state = self.gripper.read_once().width
            except Exception:
                logger.exception("Failed to command gripper to width %s", width)

    def _submit_gripper_width(self, width: float) -> None:
        with self._gripper_lock:
            self._gripper_target_width = width
            self._gripper_update_event.set()

"""RobotNode — wraps any robot driver and bridges it onto the ZMQ bus.

Works with any robot that implements:
    robot.command_joint_pos(joint_pos: np.ndarray) -> None
    robot.get_observations() -> dict  # must contain "joint_pos"

Examples: i2rt MotorChainRobot (YAM), FrankaPanda (OSC torque control).

Published topics:
    ``{name}/joint_state``  — dict from robot.get_observations()

Subscribed topics (configured at construction):
    ``{cmd_topic}``         — e.g. "gello_left/joint_pos"
"""

from __future__ import annotations

import importlib
import threading
import time

import numpy as np
import yaml as _yaml

from robots_realtime.runtime.node import Node, NodeRole


def _resolve(obj):
    """Recursively instantiate any dict containing a ``_target_`` key."""
    if isinstance(obj, dict):
        if "_target_" in obj:
            obj = dict(obj)
            target: str = obj.pop("_target_")
            kwargs = {k: _resolve(v) for k, v in obj.items()}
            module_path, cls_name = target.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, cls_name)(**kwargs)
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(item) for item in obj]
    return obj


def _instantiate_from_target_yaml(config_path: str):
    """Load a YAML config and recursively instantiate all ``_target_`` objects."""
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    return _resolve(cfg)


class RobotNode(Node):
    """Generic robot arm node.

    When loaded from YAML, ``robot`` is omitted and must be injected before
    ``setup()`` is called (or a subclass / factory overrides ``setup()``).
    The ``robot_config`` param is stored for reference but robot instantiation
    is left to the caller for hardware configs.

    Args:
        robot:        Any object implementing ``command_joint_pos()`` and
                      ``get_observations()``. Optional when loading from YAML.
        name:      Node name on the bus.
        cmd_topic: Full topic to subscribe to for joint position commands.
                   If None the node runs in read-only mode.
        writer:    Optional Writer injected at construction for recording.
    """

    role = NodeRole.ROBOT
    published_topics: list[str] = ["joint_state"]
    poll_freq: float | None = None
    subscriber_driven: bool = True

    def __init__(
        self,
        robot=None,
        name: str = "robot",
        cmd_topic: str | None = None,
        robot_config: str | None = None,
        poll_freq: float | None = None,
        writer=None,
        **kwargs,
    ) -> None:
        self.subscribed_topics = [cmd_topic] if cmd_topic else []
        # Explicitly set poll_freq and subscriber_driven before calling super().__init__
        if poll_freq is not None:
            self.poll_freq = poll_freq
            self.subscriber_driven = False  # switch to fixed_rate mode
        super().__init__(name=name, writer=writer, **kwargs)
        self._robot = robot
        self._cmd_topic = cmd_topic
        self._robot_config = robot_config  # stored for reference; instantiation is caller's job
        self._cmd_lock = threading.Lock()
        self._latest_cmd: np.ndarray | None = None
        self._cmd_stop = threading.Event()
        self._cmd_thread: threading.Thread | None = None

    def setup(self) -> None:
        if self._robot is None:
            if self._robot_config is None:
                raise RuntimeError(
                    f"[{self.name}] RobotNode.robot is None — inject a robot driver before starting. "
                    f"(robot_config={self._robot_config!r})"
                )
            self._robot = _instantiate_from_target_yaml(self._robot_config)
        # Dispatch command_joint_pos in a background thread so a blocking robot
        # driver call cannot freeze Node.step()/tick stats.
        self._cmd_thread = threading.Thread(
            target=self._command_loop,
            daemon=True,
            name=f"{self.name}_command_loop",
        )
        self._cmd_thread.start()

    def _command_loop(self) -> None:
        while not self._cmd_stop.is_set():
            cmd = None
            with self._cmd_lock:
                if self._latest_cmd is not None:
                    cmd = self._latest_cmd
                    self._latest_cmd = None
            if cmd is None:
                time.sleep(0.001)
                continue
            self._robot.command_joint_pos(cmd)

    def step(self) -> None:
        ts = time.time()
        if self._cmd_topic:
            cmd = self.get_latest(self._cmd_topic)
            if cmd is not None:
                # Use np.array() to ensure a writable copy (np.asarray may return read-only view)
                joint_pos = np.array(cmd["joint_pos"], dtype=np.float64)
                # Debug: log commands every 100 commands.
                if not hasattr(self, "_cmd_debug_count"):
                    self._cmd_debug_count = 0
                self._cmd_debug_count += 1
                if self._cmd_debug_count % 100 == 0:
                    print(
                        f"[{self.name}] RobotNode: received command #{self._cmd_debug_count}, "
                        f"calling command_joint_pos with {joint_pos}"
                    )
                with self._cmd_lock:
                    self._latest_cmd = joint_pos
            else:
                if not hasattr(self, '_no_cmd_count'):
                    self._no_cmd_count = 0
                self._no_cmd_count += 1
                if self._no_cmd_count % 100 == 0:
                    print(f"[{self.name}] RobotNode: NO COMMAND received from {self._cmd_topic} (count: {self._no_cmd_count})")

        self.publish("joint_state", self._robot.get_observations(), ts=ts)

    def cleanup(self) -> None:
        self._cmd_stop.set()
        if self._cmd_thread is not None:
            self._cmd_thread.join(timeout=1.0)
        if hasattr(self._robot, "stop"):
            self._robot.stop()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        kwargs = {
            "name": params["name"],
            "cmd_topic": params.get("cmd_topic"),
            "robot_config": params.get("robot_config"),
        }
        # Pass through poll_freq if specified
        if "poll_freq" in params:
            kwargs["poll_freq"] = params["poll_freq"]
        return kwargs

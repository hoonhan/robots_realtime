import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser.transforms as vtf
import yaml

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver

_RESOLUTION_MAP: dict[str, tuple[int, int]] = {
    "HD1080": (1920, 1080),
    "HD720": (1280, 720),
    "VGA": (640, 480),
}


@dataclass
class RealSenseCamera(CameraDriver):
    """Intel RealSense RGB(D) camera driver using pyrealsense2."""

    device_id: str | None = None
    resolution: str = "HD720"
    fps: int = 30
    enable_depth: bool = False
    image_transfer_time_offset_ms: float = 0.0
    name: str | None = None
    extrinsics_file: str | None = None

    def __post_init__(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "RealSenseCamera requires the 'pyrealsense2' package. "
                "Install Intel librealsense + python bindings before using driver='RealSenseCamera'."
            ) from exc

        self._rs = rs
        if self.resolution not in _RESOLUTION_MAP:
            raise ValueError(f"Unsupported RealSense resolution '{self.resolution}'. Valid: {list(_RESOLUTION_MAP)}")

        width, height = _RESOLUTION_MAP[self.resolution]
        self.width, self.height = width, height

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.device_id:
            self.config.enable_device(self.device_id)

        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, self.fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)

        self.align = rs.align(rs.stream.color) if self.enable_depth else None

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_stream.get_intrinsics()
        self.intrinsic_data = {
            "rgb": {
                "intrinsics_matrix": np.array(
                    [
                        [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
                        [0.0, color_intrinsics.fy, color_intrinsics.ppy],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                "distortion_coefficients": list(color_intrinsics.coeffs),
                "distortion_model": str(color_intrinsics.model),
            }
        }

        if self.enable_depth:
            depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_intrinsics = depth_stream.get_intrinsics()
            self.intrinsic_data["depth"] = {
                "intrinsics_matrix": np.array(
                    [
                        [depth_intrinsics.fx, 0.0, depth_intrinsics.ppx],
                        [0.0, depth_intrinsics.fy, depth_intrinsics.ppy],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                "distortion_coefficients": list(depth_intrinsics.coeffs),
                "distortion_model": str(depth_intrinsics.model),
            }

        self.extrinsics: dict | None = self._load_extrinsics() if self.extrinsics_file else None

    def _load_extrinsics(self) -> dict | None:
        path = Path(self.extrinsics_file)
        if not path.exists():
            logging.warning(f"RealSenseCamera: extrinsics file not found: {path}. Extrinsics will be unavailable.")
            return None

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        position = np.array(data["position"], dtype=np.float64)
        rpy = data["rpy_radians"]
        wxyz = vtf.SO3.from_rpy_radians(*rpy).wxyz
        pose_mat = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position])).as_matrix()
        return {"position": position, "wxyz": wxyz, "pose_mat": pose_mat}

    def read(self) -> CameraData:
        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSenseCamera: failed to get color frame")

        color = np.ascontiguousarray(np.asarray(color_frame.get_data()))

        timestamp_ms = float(color_frame.get_timestamp()) - self.image_transfer_time_offset_ms
        other_sensors = None

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.ascontiguousarray(np.asarray(depth_frame.get_data()))
                other_sensors = {"depth": depth}

        return CameraData(images={"rgb": color}, timestamp=timestamp_ms, other_sensors=other_sensors)

    def read_calibration_data_intrinsics(self) -> dict:
        return self.intrinsic_data

    def get_camera_info(self) -> dict:
        return {
            "camera_type": "realsense",
            "device_id": self.device_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "enable_depth": self.enable_depth,
            "name": self.name if self.name is not None else "realsense_camera",
            "intrinsics": self.intrinsic_data,
        }

    def stop(self) -> None:
        self.pipeline.stop()

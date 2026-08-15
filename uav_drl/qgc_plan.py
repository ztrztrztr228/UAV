# -*- coding: utf-8 -*-
"""Export validated local-ENU trajectories as QGroundControl Plan files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


# MAVLink enum values used by QGroundControl Plan files.
MAV_AUTOPILOT_ARDUPILOTMEGA = 3
MAV_AUTOPILOT_PX4 = 12
MAV_TYPE_QUADROTOR = 2
MAV_FRAME_MISSION = 2
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22

FIRMWARE_TYPES = {
    "ardupilot": MAV_AUTOPILOT_ARDUPILOTMEGA,
    "px4": MAV_AUTOPILOT_PX4,
}

# WGS-84 ellipsoid constants.
_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.69437999014e-3


def _as_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError("points must contain at least one finite x/y/z coordinate.")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values.")
    return array


def _validate_wgs84_origin(latitude_deg: float, longitude_deg: float, altitude_amsl_m: float) -> None:
    if not all(math.isfinite(value) for value in (latitude_deg, longitude_deg, altitude_amsl_m)):
        raise ValueError("WGS-84 origin coordinates must be finite.")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("WGS-84 origin latitude must be in [-90, 90].")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("WGS-84 origin longitude must be in [-180, 180].")


def enu_to_wgs84(
    east_m: float,
    north_m: float,
    up_m: float,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_altitude_amsl_m: float,
) -> tuple[float, float, float]:
    """Convert a short-range ENU offset to WGS-84 latitude/longitude/AMSL.

    The project maps are at most a few hundred metres wide. Meridional and
    prime-vertical WGS-84 curvature radii therefore provide ample precision
    without adding a coordinate-system dependency.
    """
    _validate_wgs84_origin(
        origin_latitude_deg,
        origin_longitude_deg,
        origin_altitude_amsl_m,
    )
    if not all(math.isfinite(value) for value in (east_m, north_m, up_m)):
        raise ValueError("ENU coordinates must be finite.")

    latitude_rad = math.radians(origin_latitude_deg)
    sin_latitude = math.sin(latitude_rad)
    denominator = math.sqrt(1.0 - _WGS84_E2 * sin_latitude * sin_latitude)
    prime_vertical_radius = _WGS84_A / denominator
    meridional_radius = _WGS84_A * (1.0 - _WGS84_E2) / denominator**3
    cos_latitude = math.cos(latitude_rad)
    if abs(cos_latitude) < 1e-12:
        raise ValueError("ENU conversion is undefined at the geographic poles.")

    latitude = origin_latitude_deg + math.degrees(north_m / meridional_radius)
    longitude = origin_longitude_deg + math.degrees(east_m / (prime_vertical_radius * cos_latitude))
    altitude = origin_altitude_amsl_m + up_m
    return latitude, longitude, altitude


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * segment)))


def _rdp_indices(points: np.ndarray, first: int, last: int, tolerance_m: float) -> list[int]:
    if last <= first + 1:
        return [first, last]
    distances = [
        _point_to_segment_distance(points[index], points[first], points[last])
        for index in range(first + 1, last)
    ]
    max_offset = int(np.argmax(distances))
    max_distance = distances[max_offset]
    if max_distance <= tolerance_m:
        return [first, last]
    split = first + 1 + max_offset
    return _rdp_indices(points, first, split, tolerance_m)[:-1] + _rdp_indices(
        points,
        split,
        last,
        tolerance_m,
    )


def simplify_trajectory(
    points: Sequence[Sequence[float]] | np.ndarray,
    tolerance_m: float = 1.0,
    max_segment_length_m: float = 25.0,
) -> np.ndarray:
    """Simplify a 3D trajectory while retaining original points and endpoints."""
    array = _as_points(points)
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be non-negative.")
    if max_segment_length_m <= 0.0:
        raise ValueError("max_segment_length_m must be positive.")
    if len(array) == 1:
        return array.copy()

    indices = _rdp_indices(array, 0, len(array) - 1, tolerance_m)
    bounded_indices: list[int] = [indices[0]]

    def append_bounded(first: int, last: int) -> None:
        if last <= first + 1 or np.linalg.norm(array[last] - array[first]) <= max_segment_length_m:
            bounded_indices.append(last)
            return
        middle = (first + last) // 2
        append_bounded(first, middle)
        append_bounded(middle, last)

    for first, last in zip(indices[:-1], indices[1:]):
        append_bounded(first, last)
    return array[np.asarray(bounded_indices, dtype=int)]


def prepare_qgc_waypoints(
    trajectory_points: Sequence[Sequence[float]] | np.ndarray,
    takeoff_altitude_m: float,
    simplification_tolerance_m: float = 1.0,
    max_segment_length_m: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertical takeoff point and simplified airborne waypoints."""
    points = _as_points(trajectory_points)
    if not math.isfinite(takeoff_altitude_m) or takeoff_altitude_m <= 0.0:
        raise ValueError("takeoff_altitude_m must be positive and finite.")
    airborne = np.flatnonzero(points[:, 2] >= takeoff_altitude_m)
    if len(airborne) == 0:
        raise ValueError(
            f"trajectory never reaches the QGC takeoff altitude of {takeoff_altitude_m:.2f} m."
        )

    takeoff_point = np.asarray(
        [points[0, 0], points[0, 1], takeoff_altitude_m],
        dtype=np.float64,
    )
    waypoints = simplify_trajectory(
        points[int(airborne[0]) :],
        tolerance_m=simplification_tolerance_m,
        max_segment_length_m=max_segment_length_m,
    )
    return takeoff_point, waypoints


def _simple_position_item(
    command: int,
    sequence: int,
    latitude: float,
    longitude: float,
    relative_altitude_m: float,
    acceptance_radius_m: float,
) -> dict[str, object]:
    params: list[float | None]
    if command == MAV_CMD_NAV_TAKEOFF:
        params = [0.0, 0.0, 0.0, None, latitude, longitude, relative_altitude_m]
    else:
        params = [0.0, acceptance_radius_m, 0.0, None, latitude, longitude, relative_altitude_m]
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": relative_altitude_m,
        "AltitudeMode": 0,
        "autoContinue": True,
        "command": command,
        "doJumpId": sequence,
        "frame": MAV_FRAME_GLOBAL_RELATIVE_ALT,
        "params": params,
        "type": "SimpleItem",
    }


def _terminal_item(command: int, sequence: int) -> dict[str, object]:
    return {
        "autoContinue": True,
        "command": command,
        "doJumpId": sequence,
        "frame": MAV_FRAME_MISSION,
        "params": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "type": "SimpleItem",
    }


def _land_item(sequence: int, latitude: float, longitude: float) -> dict[str, object]:
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": 0.0,
        "AltitudeMode": 0,
        "autoContinue": True,
        "command": MAV_CMD_NAV_LAND,
        "doJumpId": sequence,
        "frame": MAV_FRAME_GLOBAL_RELATIVE_ALT,
        "params": [0.0, 0.0, 0.0, None, latitude, longitude, 0.0],
        "type": "SimpleItem",
    }


def build_qgc_plan(
    home_point_enu: Sequence[float],
    takeoff_point_enu: Sequence[float],
    waypoint_points_enu: Sequence[Sequence[float]] | np.ndarray,
    *,
    origin_latitude_wgs84: float,
    origin_longitude_wgs84: float,
    origin_altitude_amsl_m: float,
    firmware: str = "px4",
    vehicle_type: int = MAV_TYPE_QUADROTOR,
    hover_speed_m_s: float = 5.0,
    cruise_speed_m_s: float = 15.0,
    acceptance_radius_m: float = 2.0,
    end_action: str = "none",
) -> dict[str, object]:
    """Build a QGroundControl Plan JSON object from local ENU coordinates."""
    home = np.asarray(home_point_enu, dtype=np.float64)
    takeoff = np.asarray(takeoff_point_enu, dtype=np.float64)
    waypoints = _as_points(waypoint_points_enu)
    if home.shape != (3,) or takeoff.shape != (3,):
        raise ValueError("home and takeoff points must each contain x/y/z.")
    if firmware not in FIRMWARE_TYPES:
        raise ValueError(f"firmware must be one of {tuple(FIRMWARE_TYPES)}.")
    if vehicle_type <= 0:
        raise ValueError("vehicle_type must be positive.")
    if min(hover_speed_m_s, cruise_speed_m_s, acceptance_radius_m) <= 0.0:
        raise ValueError("QGC speeds and acceptance radius must be positive.")
    if end_action not in ("none", "rtl", "land"):
        raise ValueError("end_action must be one of: none, rtl, land.")
    _validate_wgs84_origin(
        origin_latitude_wgs84,
        origin_longitude_wgs84,
        origin_altitude_amsl_m,
    )

    home_lat, home_lon, _ = enu_to_wgs84(
        home[0],
        home[1],
        0.0,
        origin_latitude_wgs84,
        origin_longitude_wgs84,
        origin_altitude_amsl_m,
    )
    takeoff_lat, takeoff_lon, _ = enu_to_wgs84(
        takeoff[0],
        takeoff[1],
        takeoff[2],
        origin_latitude_wgs84,
        origin_longitude_wgs84,
        origin_altitude_amsl_m,
    )
    items: list[dict[str, object]] = [
        _simple_position_item(
            MAV_CMD_NAV_TAKEOFF,
            1,
            takeoff_lat,
            takeoff_lon,
            float(takeoff[2]),
            acceptance_radius_m,
        )
    ]
    for point in waypoints:
        latitude, longitude, _ = enu_to_wgs84(
            point[0],
            point[1],
            point[2],
            origin_latitude_wgs84,
            origin_longitude_wgs84,
            origin_altitude_amsl_m,
        )
        items.append(
            _simple_position_item(
                MAV_CMD_NAV_WAYPOINT,
                len(items) + 1,
                latitude,
                longitude,
                float(point[2]),
                acceptance_radius_m,
            )
        )
    if end_action == "rtl":
        items.append(_terminal_item(MAV_CMD_NAV_RETURN_TO_LAUNCH, len(items) + 1))
    elif end_action == "land":
        last_latitude, last_longitude, _ = enu_to_wgs84(
            waypoints[-1, 0],
            waypoints[-1, 1],
            0.0,
            origin_latitude_wgs84,
            origin_longitude_wgs84,
            origin_altitude_amsl_m,
        )
        items.append(_land_item(len(items) + 1, last_latitude, last_longitude))

    return {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": float(cruise_speed_m_s),
            "firmwareType": FIRMWARE_TYPES[firmware],
            "globalPlanAltitudeMode": 1,
            "hoverSpeed": float(hover_speed_m_s),
            "items": items,
            "plannedHomePosition": [home_lat, home_lon, float(origin_altitude_amsl_m)],
            "vehicleType": int(vehicle_type),
            "version": 2,
        },
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }


def save_qgc_plan(plan: dict[str, object], output_path: Path) -> Path:
    """Write a QGC Plan atomically so QGC never observes a partial JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def qgc_autoload_path(load_save_directory: Path, system_id: int) -> Path:
    """Return QGC's documented AutoLoad#.plan path for one MAVLink vehicle id."""
    if not 1 <= system_id <= 255:
        raise ValueError("QGC/MAVLink system_id must be in [1, 255].")
    return Path(load_save_directory) / f"AutoLoad{system_id}.plan"


__all__ = [
    "FIRMWARE_TYPES",
    "MAV_TYPE_QUADROTOR",
    "build_qgc_plan",
    "enu_to_wgs84",
    "prepare_qgc_waypoints",
    "qgc_autoload_path",
    "save_qgc_plan",
    "simplify_trajectory",
]

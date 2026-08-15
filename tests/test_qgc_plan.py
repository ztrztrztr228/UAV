from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

import numpy as np

from uav_drl.qgc_plan import (
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    MAV_FRAME_GLOBAL_RELATIVE_ALT,
    build_qgc_plan,
    enu_to_wgs84,
    prepare_qgc_waypoints,
    qgc_autoload_path,
    save_qgc_plan,
    simplify_trajectory,
)


class QGCPlanTests(unittest.TestCase):
    def test_enu_origin_maps_to_wgs84_origin(self) -> None:
        result = enu_to_wgs84(0.0, 0.0, 12.5, 31.0, 121.0, 8.0)
        self.assertAlmostEqual(result[0], 31.0)
        self.assertAlmostEqual(result[1], 121.0)
        self.assertAlmostEqual(result[2], 20.5)

    def test_enu_axes_move_east_and_north(self) -> None:
        east = enu_to_wgs84(100.0, 0.0, 0.0, 31.0, 121.0, 0.0)
        north = enu_to_wgs84(0.0, 100.0, 0.0, 31.0, 121.0, 0.0)
        self.assertGreater(east[1], 121.0)
        self.assertAlmostEqual(east[0], 31.0)
        self.assertGreater(north[0], 31.0)
        self.assertAlmostEqual(north[1], 121.0)

    def test_simplification_keeps_endpoints_and_limits_segment_length(self) -> None:
        points = np.asarray([[float(x), 0.0, 8.0] for x in range(0, 101, 10)])
        simplified = simplify_trajectory(points, tolerance_m=1.0, max_segment_length_m=25.0)
        np.testing.assert_allclose(simplified[0], points[0])
        np.testing.assert_allclose(simplified[-1], points[-1])
        self.assertTrue(np.all(np.linalg.norm(np.diff(simplified, axis=0), axis=1) <= 25.0))
        self.assertLess(len(simplified), len(points))

    def test_prepare_waypoints_trims_ground_climb(self) -> None:
        points = np.asarray(
            [
                [10.0, 20.0, 0.7],
                [10.0, 20.0, 3.0],
                [12.0, 20.0, 5.0],
                [20.0, 20.0, 8.0],
            ]
        )
        takeoff, waypoints = prepare_qgc_waypoints(
            points,
            takeoff_altitude_m=5.0,
            simplification_tolerance_m=0.0,
        )
        np.testing.assert_allclose(takeoff, [10.0, 20.0, 5.0])
        np.testing.assert_allclose(waypoints[0], points[2])
        np.testing.assert_allclose(waypoints[-1], points[-1])

    def test_plan_has_qgc_structure_and_relative_altitude_items(self) -> None:
        plan = build_qgc_plan(
            home_point_enu=(10.0, 20.0, 0.7),
            takeoff_point_enu=(10.0, 20.0, 5.0),
            waypoint_points_enu=np.asarray([[12.0, 20.0, 5.0], [20.0, 30.0, 8.0]]),
            origin_latitude_wgs84=31.0,
            origin_longitude_wgs84=121.0,
            origin_altitude_amsl_m=6.0,
            firmware="px4",
        )
        self.assertEqual(plan["fileType"], "Plan")
        self.assertEqual(plan["version"], 1)
        mission = plan["mission"]
        self.assertEqual(mission["firmwareType"], 12)
        self.assertEqual(mission["vehicleType"], 2)
        self.assertEqual(len(mission["items"]), 3)
        self.assertEqual(mission["items"][0]["command"], MAV_CMD_NAV_TAKEOFF)
        self.assertEqual(mission["items"][1]["command"], MAV_CMD_NAV_WAYPOINT)
        self.assertEqual(mission["items"][1]["frame"], MAV_FRAME_GLOBAL_RELATIVE_ALT)
        self.assertEqual(mission["items"][2]["params"][6], 8.0)

    def test_land_action_uses_last_waypoint_location(self) -> None:
        plan = build_qgc_plan(
            home_point_enu=(0.0, 0.0, 0.0),
            takeoff_point_enu=(0.0, 0.0, 5.0),
            waypoint_points_enu=np.asarray([[10.0, 20.0, 8.0]]),
            origin_latitude_wgs84=31.0,
            origin_longitude_wgs84=121.0,
            origin_altitude_amsl_m=6.0,
            end_action="land",
        )
        waypoint = plan["mission"]["items"][-2]
        land = plan["mission"]["items"][-1]
        self.assertEqual(land["command"], MAV_CMD_NAV_LAND)
        self.assertEqual(land["params"][4:6], waypoint["params"][4:6])
        self.assertEqual(land["params"][6], 0.0)

    def test_plan_save_and_autoload_filename(self) -> None:
        plan = build_qgc_plan(
            home_point_enu=(0.0, 0.0, 0.0),
            takeoff_point_enu=(0.0, 0.0, 5.0),
            waypoint_points_enu=np.asarray([[10.0, 0.0, 5.0]]),
            origin_latitude_wgs84=31.0,
            origin_longitude_wgs84=121.0,
            origin_altitude_amsl_m=6.0,
        )
        output_directory = Path.cwd()
        output = output_directory / f".test_{uuid.uuid4().hex}_AutoLoad7.plan"
        temporary = output.with_suffix(output.suffix + ".tmp")
        try:
            self.assertEqual(qgc_autoload_path(output_directory, 7).name, "AutoLoad7.plan")
            save_qgc_plan(plan, output)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, plan)
            self.assertFalse(temporary.exists())
        finally:
            output.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

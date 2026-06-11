"""Tests for drone_sdk.mission_planner (Module 8)."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pytest

from drone_sdk.mission_planner import (
    GeoPoint, GeofencePolygon, Mission, MissionItem,
    MissionItemType, MissionStatus, MissionValidator,
    PathPlanner, ValidationSeverity,
)


HOME = GeoPoint(12.9600, 77.4173, 902.0)


@pytest.fixture
def simple_mission():
    m = Mission(name="test_flight", home=HOME, cruise_speed_ms=5.0, max_altitude_agl=100.0)
    m.add(MissionItem.takeoff(0, altitude_agl=30.0))
    m.add(MissionItem.waypoint(1, GeoPoint(12.961, 77.418, 902.0), 30.0))
    m.add(MissionItem.waypoint(2, GeoPoint(12.962, 77.419, 902.0), 30.0))
    m.add(MissionItem.rtl(3))
    return m


@pytest.fixture
def square_fence():
    return GeofencePolygon.rectangle(HOME, width_m=2000.0, height_m=2000.0,
                                      is_inclusion=True)


@pytest.fixture
def validator(square_fence):
    return MissionValidator(
        max_altitude_agl=100.0,
        geofences=[square_fence],
        max_speed_ms=15.0,
        max_range_m=10_000.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
class TestGeoPoint:
    def test_distance_to_self_is_zero(self):
        p = GeoPoint(12.96, 77.42, 900.0)
        assert abs(p.distance_to(p)) < 0.01

    def test_distance_100m_north(self):
        p0 = GeoPoint(12.960, 77.420, 900.0)
        p1 = GeoPoint(12.961, 77.420, 900.0)
        d  = p0.distance_to(p1)
        assert 100 < d < 115

    def test_bearing_north_is_zero(self):
        p0 = GeoPoint(12.960, 77.420, 900.0)
        p1 = GeoPoint(12.961, 77.420, 900.0)
        b  = p0.bearing_to(p1)
        assert abs(b) < 1.0 or abs(b - 360.0) < 1.0

    def test_bearing_east_is_90(self):
        p0 = GeoPoint(12.960, 77.420, 900.0)
        p1 = GeoPoint(12.960, 77.430, 900.0)
        b  = p0.bearing_to(p1)
        assert abs(b - 90.0) < 2.0

    def test_ned_roundtrip(self):
        p    = GeoPoint(12.962, 77.422, 910.0)
        ned  = p.to_ned(HOME)
        back = GeoPoint.from_ned(ned[0], ned[1], ned[2], HOME)
        assert abs(back.lat - p.lat) < 1e-5
        assert abs(back.lon - p.lon) < 1e-5

    def test_ned_north_is_positive_x(self):
        p   = GeoPoint(HOME.lat + 0.001, HOME.lon, HOME.alt)
        ned = p.to_ned(HOME)
        assert ned[0] > 0   # north = positive x

    def test_ned_east_is_positive_y(self):
        p   = GeoPoint(HOME.lat, HOME.lon + 0.001, HOME.alt)
        ned = p.to_ned(HOME)
        assert ned[1] > 0   # east = positive y

    def test_altitude_above_home_is_negative_z(self):
        p   = GeoPoint(HOME.lat, HOME.lon, HOME.alt + 50.0)
        ned = p.to_ned(HOME)
        assert ned[2] < 0   # higher alt = negative NED z


# ══════════════════════════════════════════════════════════════════════════════
class TestMissionItem:
    def test_takeoff_type(self):
        item = MissionItem.takeoff(0, 30.0)
        assert item.item_type == MissionItemType.TAKEOFF
        assert item.altitude_agl == 30.0

    def test_waypoint_type(self):
        item = MissionItem.waypoint(1, HOME, 30.0)
        assert item.item_type == MissionItemType.WAYPOINT
        assert item.position is not None

    def test_rtl_type(self):
        item = MissionItem.rtl(2)
        assert item.item_type == MissionItemType.RTL

    def test_land_type(self):
        item = MissionItem.land(3)
        assert item.item_type == MissionItemType.LAND

    def test_loiter_type(self):
        item = MissionItem.loiter(4, HOME, 30.0, 60.0)
        assert item.item_type == MissionItemType.LOITER_TIME
        assert item.loiter_time == 60.0

    def test_to_dict_is_serialisable(self):
        item = MissionItem.waypoint(0, HOME, 30.0)
        d    = item.to_dict()
        json.dumps(d)   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
class TestMission:
    def test_count(self, simple_mission):
        assert simple_mission.count == 4

    def test_add_increments_count(self, simple_mission):
        n = simple_mission.count
        simple_mission.add(MissionItem.waypoint(n, HOME, 30.0))
        assert simple_mission.count == n + 1

    def test_remove_decrements_count(self, simple_mission):
        n = simple_mission.count
        simple_mission.remove(1)
        assert simple_mission.count == n - 1

    def test_reindex_after_remove(self, simple_mission):
        simple_mission.remove(1)
        for i, item in enumerate(simple_mission.items):
            assert item.index == i

    def test_waypoints_filter(self, simple_mission):
        wps = simple_mission.waypoints
        assert len(wps) == 2
        assert all(w.item_type == MissionItemType.WAYPOINT for w in wps)

    def test_total_distance_positive(self, simple_mission):
        assert simple_mission.total_distance_m > 0

    def test_estimated_duration_positive(self, simple_mission):
        assert simple_mission.estimated_duration_s > 0

    def test_iteration(self, simple_mission):
        items = list(simple_mission)
        assert len(items) == 4

    def test_len(self, simple_mission):
        assert len(simple_mission) == 4

    def test_repr(self, simple_mission):
        r = repr(simple_mission)
        assert "test_flight" in r

    def test_to_json_roundtrip(self, simple_mission, tmp_path):
        path = simple_mission.save(str(tmp_path / "mission.json"))
        m2   = Mission.load(str(path))
        assert m2.name  == simple_mission.name
        assert m2.count == simple_mission.count

    def test_to_qgc_plan_structure(self, simple_mission):
        plan = simple_mission.to_qgc_plan()
        assert "mission" in plan
        assert "items"   in plan["mission"]
        assert len(plan["mission"]["items"]) == 4

    def test_clear(self, simple_mission):
        simple_mission.clear()
        assert simple_mission.count == 0

    def test_insert_reindexes(self, simple_mission):
        new_item = MissionItem.waypoint(99, HOME, 25.0)
        simple_mission.insert(1, new_item)
        for i, item in enumerate(simple_mission.items):
            assert item.index == i


# ══════════════════════════════════════════════════════════════════════════════
class TestGeofencePolygon:
    def test_point_inside_square_fence(self, square_fence):
        inside = GeoPoint(HOME.lat + 0.002, HOME.lon + 0.002, HOME.alt)
        assert square_fence.contains(inside)

    def test_point_outside_square_fence(self, square_fence):
        outside = GeoPoint(HOME.lat + 1.0, HOME.lon + 1.0, HOME.alt)
        assert not square_fence.contains(outside)

    def test_exclusion_zone_logic(self):
        nfz = GeofencePolygon.rectangle(HOME, 500.0, 500.0, is_inclusion=False)
        inside = GeoPoint(HOME.lat + 0.0005, HOME.lon + 0.0005, HOME.alt)
        assert nfz.contains(inside)   # Point IS inside the exclusion polygon

    def test_altitude_check_within(self, square_fence):
        assert square_fence.is_altitude_ok(50.0)

    def test_altitude_check_above_ceiling(self, square_fence):
        assert not square_fence.is_altitude_ok(200.0)

    def test_area_positive(self, square_fence):
        assert square_fence.area_m2() > 0

    def test_area_approx_correct(self):
        fence = GeofencePolygon.rectangle(HOME, 1000.0, 1000.0)
        area  = fence.area_m2()
        assert abs(area - 1_000_000.0) / 1_000_000.0 < 0.05  # within 5%

    def test_circle_approx_n_vertices(self):
        fence = GeofencePolygon.circle_approx(HOME, 500.0, n_segments=32)
        assert len(fence.vertices) == 32

    def test_circle_centre_is_inside(self):
        fence = GeofencePolygon.circle_approx(HOME, 500.0)
        assert fence.contains(HOME)

    def test_circle_far_point_is_outside(self):
        fence = GeofencePolygon.circle_approx(HOME, 500.0)
        far   = GeoPoint(HOME.lat + 1.0, HOME.lon + 1.0, HOME.alt)
        assert not fence.contains(far)


# ══════════════════════════════════════════════════════════════════════════════
class TestMissionValidator:
    def test_valid_mission_passes(self, simple_mission, validator):
        r = validator.validate(simple_mission)
        assert r.is_valid, [str(e) for e in r.errors]

    def test_empty_mission_fails(self, validator):
        m = Mission("empty", HOME)
        r = validator.validate(m)
        assert not r.is_valid
        assert any(i.code == "EMPTY" for i in r.errors)

    def test_altitude_ceiling_violated(self, validator):
        m = Mission("high_alt", HOME, max_altitude_agl=100.0)
        m.add(MissionItem.takeoff(0, altitude_agl=200.0))
        m.add(MissionItem.rtl(1))
        r = validator.validate(m)
        assert any(i.code == "ALT_CEILING" for i in r.errors)

    def test_negative_altitude_fails(self, validator):
        m = Mission("neg_alt", HOME)
        m.add(MissionItem.takeoff(0, altitude_agl=-5.0))
        m.add(MissionItem.rtl(1))
        r = validator.validate(m)
        assert any(i.code == "NEG_ALT" for i in r.errors)

    def test_speed_limit_violated(self, validator):
        m = Mission("fast", HOME, cruise_speed_ms=20.0)
        m.add(MissionItem.takeoff(0, altitude_agl=30.0))
        m.add(MissionItem.rtl(1))
        r = validator.validate(m)
        assert any(i.code == "SPEED_LIMIT" for i in r.errors)

    def test_outside_inclusion_fence_fails(self):
        small_fence = GeofencePolygon.rectangle(HOME, 100.0, 100.0)
        v = MissionValidator(geofences=[small_fence])
        m = Mission("far", HOME)
        m.add(MissionItem.takeoff(0, altitude_agl=30.0))
        far_pt = GeoPoint(HOME.lat + 1.0, HOME.lon + 1.0, HOME.alt)
        m.add(MissionItem.waypoint(1, far_pt, 30.0))
        m.add(MissionItem.rtl(2))
        r = v.validate(m)
        assert any(i.code == "OUTSIDE_FENCE" for i in r.errors)

    def test_inside_exclusion_zone_fails(self):
        nfz = GeofencePolygon.rectangle(HOME, 500.0, 500.0, is_inclusion=False)
        v   = MissionValidator(geofences=[nfz])
        m   = Mission("nfz_test", HOME)
        m.add(MissionItem.takeoff(0, altitude_agl=30.0))
        inside = GeoPoint(HOME.lat + 0.0005, HOME.lon + 0.0005, HOME.alt)
        m.add(MissionItem.waypoint(1, inside, 30.0))
        m.add(MissionItem.rtl(2))
        r = v.validate(m)
        assert any(i.code == "IN_NFZ" for i in r.errors)

    def test_no_takeoff_warning(self, validator):
        m = Mission("no_takeoff", HOME)
        m.add(MissionItem.rtl(0))
        r = validator.validate(m)
        assert any(i.code == "NO_TAKEOFF" for i in r.warnings)

    def test_no_landing_warning(self, validator):
        m = Mission("no_land", HOME)
        m.add(MissionItem.takeoff(0, altitude_agl=30.0))
        m.add(MissionItem.waypoint(1, HOME, 30.0))
        r = validator.validate(m)
        assert any(i.code == "NO_LANDING" for i in r.warnings)

    def test_battery_warning(self):
        v = MissionValidator(battery_capacity_wh=5.0, power_draw_w=200.0)
        m = Mission("battery_test", HOME, cruise_speed_ms=5.0)
        m.add(MissionItem.takeoff(0, altitude_agl=30.0))
        for i in range(10):
            pt = GeoPoint(HOME.lat + i * 0.01, HOME.lon, HOME.alt)
            m.add(MissionItem.waypoint(i+1, pt, 30.0))
        m.add(MissionItem.rtl(11))
        r = v.validate(m)
        assert any(i.code == "BATTERY_RANGE" for i in r.warnings)

    def test_validation_result_repr(self, simple_mission, validator):
        r = validator.validate(simple_mission)
        assert "ValidationResult" in repr(r)

    def test_info_always_added(self, simple_mission, validator):
        r = validator.validate(simple_mission)
        infos = [i for i in r.issues if i.severity == ValidationSeverity.INFO]
        assert len(infos) >= 1


# ══════════════════════════════════════════════════════════════════════════════
class TestPathPlanner:
    def test_straight_line_start_end(self):
        wps   = [np.array([0,0,-10.0]), np.array([100,0,-10.0])]
        path  = PathPlanner.straight_line(wps, samples_per_segment=10)
        assert np.allclose(path[0], wps[0], atol=1e-9)

    def test_straight_line_shape(self):
        wps  = [np.array([0,0,0.0]), np.array([50,0,0.0]), np.array([100,50,0.0])]
        path = PathPlanner.straight_line(wps, samples_per_segment=20)
        assert path.shape[1] == 3
        assert path.shape[0] > 3

    def test_straight_line_single_waypoint(self):
        path = PathPlanner.straight_line([np.array([0,0,0.0])])
        assert len(path) == 1

    def test_minimum_snap_returns_tuple(self):
        wps  = [np.array([0,0,-10.0]), np.array([50,0,-10.0]), np.array([100,50,-10.0])]
        pos, t = PathPlanner.minimum_snap(wps, dt=0.05)
        assert pos.shape[1] == 3
        assert len(t) == len(pos)

    def test_minimum_snap_timestamps_monotone(self):
        wps = [np.array([0,0,-10.0]), np.array([50,0,-10.0])]
        _, t = PathPlanner.minimum_snap(wps, dt=0.1)
        assert np.all(np.diff(t) >= 0)

    def test_minimum_snap_single_waypoint(self):
        pos, t = PathPlanner.minimum_snap([np.array([0,0,0.0])])
        assert len(pos) == 1

    def test_minimum_snap_custom_times(self):
        wps   = [np.array([0,0,-10.0]), np.array([50,0,-10.0])]
        pos, t = PathPlanner.minimum_snap(wps, segment_times=[5.0], dt=0.1)
        assert abs(t[-1] - 5.0) < 0.2

    def test_dubins_returns_2d_path(self):
        start = (0.0, 0.0, 0.0)
        end   = (100.0, 50.0, math.pi/4)
        path  = PathPlanner.dubins(start, end, turn_radius=20.0)
        assert path.shape[1] == 2
        assert len(path) >= 2

    def test_dubins_start_position(self):
        start = (10.0, 20.0, 0.0)
        end   = (110.0, 20.0, 0.0)
        path  = PathPlanner.dubins(start, end)
        assert abs(path[0, 0] - 10.0) < 0.01
        assert abs(path[0, 1] - 20.0) < 0.01

"""Tests for drone_sdk.swarm_engine (Module 21)."""
from __future__ import annotations
import math
import numpy as np
import pytest

from drone_sdk.swarm_engine import (
    CommConfig, DefenseSim, DefenseScenario, FormationController,
    FormationGeometry, FormationType, JammingSource, SwarmCommunication,
    SwarmManager, SwarmMessage, SwarmVehicle,
)


def make_vehicle(vid, x=0.0, y=0.0, z=-5.0, is_leader=False):
    return SwarmVehicle(
        vehicle_id=vid,
        position=np.array([x, y, z]),
        velocity=np.zeros(3),
        heading=0.0,
        is_leader=is_leader,
    )


# ══════════════════════════════════════════════════════════════════════════════
class TestSwarmVehicle:
    def test_distance_to_self_zero(self):
        v = make_vehicle("d0", 0, 0)
        assert abs(v.distance_to(v)) < 1e-9

    def test_distance_to_other(self):
        v0 = make_vehicle("d0", 0, 0)
        v1 = make_vehicle("d1", 3, 4)
        assert abs(v0.distance_to(v1) - 5.0) < 1e-6

    def test_bearing_east(self):
        v0 = make_vehicle("d0", 0, 0)
        v1 = make_vehicle("d1", 0, 10)
        b  = v0.bearing_to(v1)
        assert abs(b - math.pi/2) < 0.01

    def test_not_stale_when_fresh(self):
        v = make_vehicle("d0")
        assert not v.is_stale


# ══════════════════════════════════════════════════════════════════════════════
class TestFormationGeometry:
    def test_line_returns_n_offsets(self):
        offsets = FormationGeometry.compute(FormationType.LINE, 4)
        assert len(offsets) == 4

    def test_v_shape_returns_n_offsets(self):
        offsets = FormationGeometry.compute(FormationType.V_SHAPE, 5)
        assert len(offsets) == 5

    def test_circle_returns_n_offsets(self):
        offsets = FormationGeometry.compute(FormationType.CIRCLE, 6)
        assert len(offsets) == 6

    def test_grid_returns_n_offsets(self):
        offsets = FormationGeometry.compute(FormationType.GRID, 9)
        assert len(offsets) == 9

    def test_diamond_returns_4(self):
        offsets = FormationGeometry.compute(FormationType.DIAMOND, 4)
        assert len(offsets) == 4

    def test_wedge_returns_n(self):
        offsets = FormationGeometry.compute(FormationType.WEDGE, 5)
        assert len(offsets) == 5

    def test_all_offsets_3d(self):
        for ft in FormationType:
            offsets = FormationGeometry.compute(ft, 4)
            for o in offsets:
                assert o.shape == (3,)

    def test_line_first_offset_zero(self):
        offsets = FormationGeometry.compute(FormationType.LINE, 4)
        assert np.allclose(offsets[0], [0, 0, 0])

    def test_rotate_to_heading_zero_unchanged(self):
        offset = np.array([1.0, 0.0, 0.0])
        rotated = FormationGeometry.rotate_to_heading(offset, 0.0)
        assert np.allclose(rotated, offset, atol=1e-9)

    def test_rotate_to_heading_90(self):
        offset  = np.array([1.0, 0.0, 0.0])
        rotated = FormationGeometry.rotate_to_heading(offset, math.pi/2)
        assert abs(rotated[0]) < 0.01
        assert abs(rotated[1] - 1.0) < 0.01

    def test_spacing_scales_offsets(self):
        o1 = FormationGeometry.compute(FormationType.LINE, 3, spacing=5.0)
        o2 = FormationGeometry.compute(FormationType.LINE, 3, spacing=10.0)
        assert np.linalg.norm(o2[1]) > np.linalg.norm(o1[1])


# ══════════════════════════════════════════════════════════════════════════════
class TestFormationController:
    @pytest.fixture
    def ctrl(self):
        return FormationController()

    @pytest.fixture
    def leader(self):
        v = make_vehicle("lead", 0, 0, -10, is_leader=True)
        v.velocity = np.array([2.0, 0.0, 0.0])
        return v

    @pytest.fixture
    def follower(self):
        return make_vehicle("follow", -6, 2, -10)

    def test_returns_3d_velocity(self, ctrl, leader, follower):
        cmd = ctrl.compute_velocity_command(leader, follower, np.array([-5.0, 0.0, 0.0]))
        assert cmd.shape == (3,)

    def test_velocity_finite(self, ctrl, leader, follower):
        cmd = ctrl.compute_velocity_command(leader, follower, np.array([-5.0, 0.0, 0.0]))
        assert np.all(np.isfinite(cmd))

    def test_speed_limited(self, ctrl, leader, follower):
        # Place follower very far → large error → should be clipped
        follower.position = np.array([1000.0, 1000.0, -10.0])
        cmd = ctrl.compute_velocity_command(leader, follower, np.zeros(3))
        assert np.linalg.norm(cmd) <= ctrl._vmax + 0.01

    def test_moves_toward_desired(self, ctrl, leader, follower):
        desired_offset = np.array([-5.0, 0.0, 0.0])
        desired_pos    = leader.position + desired_offset
        cmd = ctrl.compute_velocity_command(leader, follower, desired_offset)
        error_before = np.linalg.norm(follower.position - desired_pos)
        new_pos      = follower.position + cmd * 0.1
        error_after  = np.linalg.norm(new_pos - desired_pos)
        assert error_after < error_before

    def test_repulsion_from_nearby(self, ctrl, leader):
        f1 = make_vehicle("f1", 0.5, 0.0, -10)  # Very close
        f2 = make_vehicle("f2", 0.0, 0.0, -10)
        cmd = ctrl.compute_velocity_command(leader, f2, np.array([-5.0, 0.0, 0.0]),
                                             neighbors=[f1])
        assert np.linalg.norm(cmd) > 0


# ══════════════════════════════════════════════════════════════════════════════
class TestSwarmCommunication:
    @pytest.fixture
    def comm(self):
        return SwarmCommunication(CommConfig(packet_loss=0.0, range_m=1000.0), seed=42)

    @pytest.fixture
    def vehicles(self):
        return [make_vehicle(f"d{i}", i*10, 0) for i in range(3)]

    def test_send_delivers_message(self, comm, vehicles):
        ok = comm.send("d0", "d1", {"pos": [1, 2, 3]}, vehicles)
        assert ok

    def test_receive_returns_message(self, comm, vehicles):
        comm.send("d0", "d1", {"x": 42}, vehicles)
        msgs = comm.receive("d1")
        assert len(msgs) == 1
        assert msgs[0].payload["x"] == 42

    def test_receive_clears_inbox(self, comm, vehicles):
        comm.send("d0", "d1", {"x": 1}, vehicles)
        comm.receive("d1")
        assert comm.receive("d1") == []

    def test_broadcast_delivers_to_all(self, comm, vehicles):
        comm.send("d0", "", {"broadcast": True}, vehicles)
        for v in vehicles[1:]:
            msgs = comm.receive(v.vehicle_id)
            assert len(msgs) == 1

    def test_broadcast_not_to_sender(self, comm, vehicles):
        comm.send("d0", "", {"x": 1}, vehicles)
        assert comm.receive("d0") == []

    def test_out_of_range_dropped(self, vehicles):
        comm = SwarmCommunication(CommConfig(packet_loss=0.0, range_m=5.0), seed=0)
        # d0 at x=0, d2 at x=20 → out of range
        ok = comm.send("d0", "d2", {"x": 1}, vehicles)
        assert not ok

    def test_packet_loss(self, vehicles):
        comm = SwarmCommunication(CommConfig(packet_loss=1.0, range_m=1000.0), seed=0)
        ok   = comm.send("d0", "d1", {"x": 1}, vehicles)
        assert not ok

    def test_drop_rate_zero_no_loss(self, comm, vehicles):
        for _ in range(10):
            comm.send("d0", "d1", {}, vehicles)
        assert comm.drop_rate == 0.0

    def test_get_stats(self, comm, vehicles):
        comm.send("d0", "d1", {}, vehicles)
        stats = comm.get_stats()
        assert "total_sent"    in stats
        assert "total_dropped" in stats
        assert "drop_rate"     in stats

    def test_message_has_sender_id(self, comm, vehicles):
        comm.send("d0", "d1", {}, vehicles)
        msg = comm.receive("d1")[0]
        assert msg.sender_id == "d0"


# ══════════════════════════════════════════════════════════════════════════════
class TestSwarmManager:
    @pytest.fixture
    def manager(self):
        return SwarmManager(n_drones=4, formation=FormationType.V_SHAPE, spacing=5.0)

    def test_n_active(self, manager):
        assert manager.n_active == 4

    def test_leader_set(self, manager):
        assert manager.leader is not None
        assert manager.leader.is_leader

    def test_get_positions_returns_dict(self, manager):
        pos = manager.get_positions()
        assert len(pos) == 4
        for p in pos.values():
            assert p.shape == (3,)

    def test_tick_moves_followers(self, manager):
        pos_before = {k: v.copy() for k, v in manager.get_positions().items()}
        # Move leader to force follower movement
        manager.leader.velocity = np.array([1.0, 0.0, 0.0])
        manager.leader.position += np.array([2.0, 0.0, 0.0])
        for _ in range(10):
            manager.tick(dt=0.1)
        pos_after = manager.get_positions()
        # At least one follower should have moved
        moved = any(
            np.linalg.norm(pos_after[k] - pos_before[k]) > 0.01
            for k in pos_after if k != "drone_0"
        )
        assert moved

    def test_distance_matrix_shape(self, manager):
        D = manager.inter_drone_distances()
        assert D.shape == (4, 4)

    def test_distance_matrix_diagonal_zero(self, manager):
        D = manager.inter_drone_distances()
        assert np.allclose(np.diag(D), 0.0)

    def test_distance_matrix_symmetric(self, manager):
        D = manager.inter_drone_distances()
        assert np.allclose(D, D.T)

    def test_set_formation(self, manager):
        manager.set_formation(FormationType.CIRCLE)
        assert manager._formation == FormationType.CIRCLE

    def test_set_leader(self, manager):
        manager.set_leader("drone_2")
        assert manager.leader.vehicle_id == "drone_2"

    def test_add_vehicle(self, manager):
        v = make_vehicle("drone_extra", 100, 0)
        manager.add_vehicle(v)
        assert manager.n_active == 5

    def test_remove_vehicle(self, manager):
        ok = manager.remove_vehicle("drone_3")
        assert ok
        assert manager.n_active == 3

    def test_remove_nonexistent_returns_false(self, manager):
        assert manager.remove_vehicle("ghost_drone") is False

    def test_centroid_shape(self, manager):
        c = manager.get_swarm_centroid()
        assert c.shape == (3,)

    def test_centroid_within_swarm_bounds(self, manager):
        pos  = list(manager.get_positions().values())
        xs   = [p[0] for p in pos]
        c    = manager.get_swarm_centroid()
        assert min(xs) - 1 <= c[0] <= max(xs) + 1

    def test_velocity_commands_dict(self, manager):
        cmds = manager.get_velocity_commands()
        assert len(cmds) == 4


# ══════════════════════════════════════════════════════════════════════════════
class TestDefenseSim:
    @pytest.fixture
    def sim(self):
        return DefenseSim()

    @pytest.fixture
    def jammer(self):
        return JammingSource(position=np.array([0.0, 0.0, -50.0]), radius_m=300.0)

    def test_gps_available_no_jammers(self, sim):
        assert sim.gps_available(np.array([100.0, 0.0, -50.0]))

    def test_gps_denied_inside_jammer(self, sim, jammer):
        sim.add_jammer(jammer)
        assert not sim.gps_available(np.array([0.0, 0.0, -50.0]))

    def test_gps_available_outside_jammer(self, sim, jammer):
        sim.add_jammer(jammer)
        assert sim.gps_available(np.array([500.0, 0.0, -50.0]))

    def test_spoofing_detected(self, sim):
        spoofer = JammingSource(np.array([100.0, 0.0, -50.0]), radius_m=200.0)
        sim.add_spoofer(spoofer)
        assert sim.gps_spoofed(np.array([100.0, 0.0, -50.0]))

    def test_no_spoofing_without_spoofer(self, sim):
        assert not sim.gps_spoofed(np.array([0.0, 0.0, -50.0]))

    def test_spoof_position_changes_reading(self, sim):
        spoofer  = JammingSource(np.array([500.0, 0.0, 0.0]), radius_m=1000.0)
        sim.add_spoofer(spoofer)
        true_pos = np.array([0.0, 0.0, -50.0])
        spoofed  = sim.spoof_position(true_pos, spoofer)
        assert not np.allclose(true_pos, spoofed)

    def test_comm_not_jammed_no_jammers(self, sim):
        a = np.array([0.0, 0.0, -50.0])
        b = np.array([100.0, 0.0, -50.0])
        assert not sim.comm_jammed(a, b)

    def test_comm_jammed_inside_zone(self, sim):
        jammer = JammingSource(np.array([50.0, 0.0, -50.0]),
                                radius_m=200.0, frequency_mhz=915.0)
        sim.add_jammer(jammer)
        a = np.array([0.0, 0.0, -50.0])
        b = np.array([100.0, 0.0, -50.0])
        assert sim.comm_jammed(a, b, frequency_mhz=915.0)

    def test_no_adversary_returns_none(self, sim):
        result = sim.get_nearest_adversary(np.zeros(3))
        assert result is None

    def test_adversary_detected(self, sim):
        adv = make_vehicle("adv", 50, 0, -50)
        sim.add_adversary(adv)
        result = sim.get_nearest_adversary(np.array([0.0, 0.0, -50.0]))
        assert result is not None
        _, dist = result
        assert abs(dist - 50.0) < 0.1

    def test_threat_level_zero_no_threats(self, sim):
        level = sim.threat_level(np.array([1000.0, 1000.0, -50.0]))
        assert level == 0.0

    def test_threat_level_high_inside_jammer(self, sim):
        jammer = JammingSource(np.array([0.0, 0.0, -50.0]), radius_m=500.0)
        sim.add_jammer(jammer)
        level = sim.threat_level(np.array([0.0, 0.0, -50.0]))
        assert level > 0.3

    def test_threat_level_bounded_0_1(self, sim):
        jammer  = JammingSource(np.array([0.0, 0.0, 0.0]), radius_m=5000.0)
        spoofer = JammingSource(np.array([0.0, 0.0, 0.0]), radius_m=5000.0)
        adv     = make_vehicle("adv", 10, 0, 0)
        sim.add_jammer(jammer)
        sim.add_spoofer(spoofer)
        sim.add_adversary(adv)
        level = sim.threat_level(np.array([0.0, 0.0, 0.0]))
        assert 0.0 <= level <= 1.0

    def test_get_summary_keys(self, sim):
        s = sim.get_summary()
        assert "n_jammers"     in s
        assert "n_spoofers"    in s
        assert "n_adversaries" in s

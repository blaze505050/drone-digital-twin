"""Tests for flight_simulator.simulator — core simulation loop, stepping, and telemetry."""
import json
import numpy as np
import pytest

from drone_sdk.flight_simulator.simulator import (
    FlightSimulator,
    SimulatorConfig,
    SimulatorState,
    UAVType,
)


class TestSimulatorState:
    def test_json_serialization(self):
        """SimulatorState should serialize to valid JSON matching expected schema."""
        state = SimulatorState(
            time_s=1.5,
            pos_ned=np.array([1.0, 2.0, -10.0]),
            vel_ned=np.array([0.5, 0.0, -0.2]),
            vel_body=np.array([0.5, 0.0, -0.2]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            euler_deg=np.array([0.0, 0.0, 45.0]),
            omega_body=np.array([0.01, -0.02, 0.0]),
            motor_commands=np.array([0.5, 0.5, 0.5, 0.5]),
            motor_rpms=np.array([5000.0, 5000.0, 5000.0, 5000.0]),
            altitude_agl=10.0,
            airspeed_mps=0.5,
            battery_soc=0.98,
            uav_type="quadrotor",
            flight_phase="hover",
            wind_ned=np.array([1.0, 0.5, 0.0]),
            air_density=1.225,
            twin_health=0.99,
        )
        json_str = state.to_json()
        data = json.loads(json_str)

        assert data["t"] == 1.5
        assert data["alt"] == 10.0
        assert data["type"] == "quadrotor"
        assert len(data["pos"]) == 3
        assert len(data["quat"]) == 4
        assert len(data["motors"]) == 4
        assert data["twin_health"] == 0.99


class TestFlightSimulator:
    def test_quadrotor_step_without_nan(self):
        """Simulator should step 200 iterations (1 second) without NaNs or infs."""
        sim = FlightSimulator(SimulatorConfig(
            uav_type=UAVType.QUADROTOR,
            enable_wind=True,
            enable_controller=True,
        ))
        sim.set_target_position(np.array([0, 0, -5.0]), yaw=0.0)

        for _ in range(200):
            snap = sim.step(dt=0.005)
            assert not np.isnan(snap.pos_ned).any()
            assert not np.isinf(snap.pos_ned).any()
            assert not np.isnan(snap.quat).any()

        # Altitude should remain positive (above ground)
        assert snap.altitude_agl > 0.0

    def test_hexarotor_instantiation_and_step(self):
        """Hexarotor configuration should step cleanly and command 6 motors."""
        sim = FlightSimulator(SimulatorConfig(
            uav_type=UAVType.HEXAROTOR,
            enable_controller=False,
        ))
        assert sim.uav_model.num_motors == 6
        snap = sim.step(dt=0.005)
        assert not np.isnan(snap.pos_ned).any()

    def test_fixed_wing_step(self):
        """Fixed-wing configuration should step without numerical divergence."""
        sim = FlightSimulator(SimulatorConfig(
            uav_type=UAVType.FIXED_WING,
            enable_controller=False,
            enable_wind=False,
        ))
        # Provide manual throttle
        sim._manual_input["throttle"] = 0.6
        for _ in range(100):
            snap = sim.step(dt=0.005)
            assert not np.isnan(snap.pos_ned).any()

    def test_vtol_step(self):
        """VTOL tilt-rotor configuration should step through hover mode."""
        sim = FlightSimulator(SimulatorConfig(
            uav_type=UAVType.VTOL,
            enable_controller=False,
            enable_wind=False,
        ))
        for _ in range(100):
            snap = sim.step(dt=0.005)
            assert not np.isnan(snap.pos_ned).any()

"""Tests for drone_sdk.sensor_models (Module 9)."""
from __future__ import annotations
import math
import numpy as np
import pytest

from drone_sdk.sensor_models import (
    BarometerConfig, BarometerModel,
    ComplementaryFilter,
    GPSConfig, GPSModel,
    IMUConfig, IMUModel,
    LiDARConfig, LiDARModel,
    MagnetometerConfig, MagnetometerModel,
)

SEED = 42


class TestIMUModel:
    @pytest.fixture
    def imu(self):
        return IMUModel(seed=SEED)

    def test_returns_measurement(self, imu):
        m = imu.measure(np.array([0., 0., -9.81]), np.zeros(3), dt=0.005)
        assert m.accelerometer.shape == (3,)
        assert m.gyroscope.shape == (3,)

    def test_noise_is_small(self, imu):
        acc = np.array([0., 0., -9.81])
        ms  = [imu.measure(acc, np.zeros(3), 0.005) for _ in range(500)]
        vals = [m.accelerometer[2] for m in ms]
        assert abs(np.mean(vals) - (-9.81)) < 0.5
        assert np.std(vals) < 0.5

    def test_gyro_near_zero(self, imu):
        ms = [imu.measure(np.array([0., 0., -9.81]), np.zeros(3), 0.005) for _ in range(500)]
        assert np.std([m.gyroscope[2] for m in ms]) < 0.1

    def test_bias_drifts_over_time(self):
        imu = IMUModel(seed=7)
        b0  = imu._acc_bias.copy()
        for _ in range(1000):
            imu.measure(np.array([0., 0., -9.81]), np.zeros(3), 0.01)
        assert np.linalg.norm(imu._acc_bias - b0) > 0

    def test_reset_bias(self, imu):
        for _ in range(100):
            imu.measure(np.zeros(3), np.zeros(3), 0.01)
        imu.reset_bias()
        assert np.allclose(imu._acc_bias,  np.zeros(3))
        assert np.allclose(imu._gyro_bias, np.zeros(3))

    def test_temperature_nominal(self, imu):
        m = imu.measure(np.zeros(3), np.zeros(3), 0.005)
        assert abs(m.temperature - 25.0) < 5.0

    def test_to_state_update(self, imu):
        m   = imu.measure(np.array([0., 0., -9.81]), np.zeros(3), 0.005)
        upd = imu.to_state_update(m, "drone_0")
        assert upd.vehicle_id == "drone_0"
        assert upd.acceleration is not None
        assert upd.angular_velocity is not None

    def test_all_finite(self, imu):
        m = imu.measure(np.zeros(3), np.zeros(3), 0.005)
        assert np.all(np.isfinite(m.accelerometer))
        assert np.all(np.isfinite(m.gyroscope))

    def test_various_dt_all_finite(self):
        for dt in [0.02, 0.005, 0.001]:
            imu = IMUModel(seed=SEED)
            for _ in range(20):
                m = imu.measure(np.array([0., 0., -9.81]), np.zeros(3), dt)
                assert np.all(np.isfinite(m.accelerometer))


class TestGPSModel:
    @pytest.fixture
    def gps(self):
        return GPSModel(seed=SEED)

    def test_returns_measurement(self, gps):
        m = gps.measure(np.zeros(3), np.zeros(3))
        assert m.position_ned.shape == (3,)
        assert m.velocity_ned.shape == (3,)

    def test_position_noise_within_spec(self, gps):
        pos = np.array([100., 50., -20.])
        ms  = [gps.measure(pos, np.zeros(3)) for _ in range(200)]
        xs  = [m.position_ned[0] for m in ms]
        assert abs(np.mean(xs) - 100.) < 2.0
        assert np.std(xs) < 5.0

    def test_velocity_noise(self, gps):
        vel = np.array([5., 0., 0.])
        ms  = [gps.measure(np.zeros(3), vel) for _ in range(200)]
        vxs = [m.velocity_ned[0] for m in ms]
        assert abs(np.mean(vxs) - 5.) < 0.5
        assert np.std(vxs) < 1.0

    def test_fix_type_valid(self, gps):
        m = gps.measure(np.zeros(3), np.zeros(3))
        assert m.fix_type in (0, 3)

    def test_satellites_nonneg(self, gps):
        m = gps.measure(np.zeros(3), np.zeros(3))
        assert m.n_satellites >= 0

    def test_lat_lon_near_home(self, gps):
        m   = gps.measure(np.zeros(3), np.zeros(3))
        cfg = GPSConfig()
        assert abs(m.lat - cfg.home_lat) < 0.01
        assert abs(m.lon - cfg.home_lon) < 0.01

    def test_north_offset_increases_lat(self, gps):
        m = gps.measure(np.array([100., 0., 0.]), np.zeros(3))
        assert m.lat > GPSConfig().home_lat

    def test_to_state_update(self, gps):
        m   = gps.measure(np.array([10., 5., -20.]), np.zeros(3))
        upd = gps.to_state_update(m, "drone_0")
        assert upd.position is not None
        assert upd.gps_lat  is not None
        assert upd.altitude_agl is not None

    def test_urban_env_degrades_accuracy(self):
        c = GPSModel(GPSConfig(environment_factor=0.0), seed=SEED)
        u = GPSModel(GPSConfig(environment_factor=0.8), seed=SEED)
        sc = np.std([c.measure(np.zeros(3), np.zeros(3)).position_ned[0] for _ in range(50)])
        su = np.std([u.measure(np.zeros(3), np.zeros(3)).position_ned[0] for _ in range(50)])
        assert su > sc


class TestBarometerModel:
    @pytest.fixture
    def baro(self):
        return BarometerModel(seed=SEED)

    def test_returns_measurement(self, baro):
        m = baro.measure(25.0)
        assert isinstance(m.pressure_pa, float)
        assert isinstance(m.altitude_m,  float)

    def test_altitude_near_true(self, baro):
        alts = [baro.measure(25.0).altitude_m for _ in range(200)]
        assert abs(np.mean(alts) - 25.0) < 2.0

    def test_sea_level_pressure(self, baro):
        m = baro.measure(0.0)
        assert abs(m.pressure_pa - 101_325.0) < 500

    def test_higher_altitude_lower_pressure(self, baro):
        assert baro.measure(100.).pressure_pa < baro.measure(10.).pressure_pa

    def test_temperature_reasonable(self, baro):
        m = baro.measure(25.)
        assert 15. < m.temperature_c < 45.

    def test_quantisation(self):
        baro = BarometerModel(BarometerConfig(quantisation_pa=1.0), seed=SEED)
        m    = baro.measure(25.)
        assert abs(m.pressure_pa - round(m.pressure_pa)) < 0.01

    def test_near_zero_alt(self, baro):
        m = baro.measure(0.)
        assert abs(m.altitude_m) < 5.


class TestMagnetometerModel:
    @pytest.fixture
    def mag(self):
        return MagnetometerModel(seed=SEED)

    def test_returns_measurement(self, mag):
        m = mag.measure(0., 0., 0.)
        assert m.field_body.shape == (3,)
        assert isinstance(m.heading_deg, float)

    def test_heading_in_range(self, mag):
        for yaw in np.linspace(0, 2*math.pi, 20):
            m = mag.measure(0., 0., yaw)
            assert 0. <= m.heading_deg < 360.

    def test_heading_near_north_at_zero_yaw(self, mag):
        m = mag.measure(0., 0., 0.)
        assert m.heading_deg < 10. or m.heading_deg > 350.

    def test_heading_near_east_at_90deg(self, mag):
        m = mag.measure(0., 0., math.pi/2)
        assert 80. < m.heading_deg < 100.

    def test_heading_near_south_at_180deg(self, mag):
        m = mag.measure(0., 0., math.pi)
        assert 170. < m.heading_deg < 190.

    def test_consistent_heading(self, mag):
        ms = [mag.measure(0., 0., 1.0) for _ in range(50)]
        assert np.std([m.heading_deg for m in ms]) < 5.

    def test_hard_iron_changes_field(self):
        normal    = MagnetometerModel(seed=SEED)
        disturbed = MagnetometerModel(MagnetometerConfig(
            hard_iron=np.array([0.1, 0., 0.])), seed=SEED)
        m_n = normal.measure(0., 0., 0.)
        m_d = disturbed.measure(0., 0., 0.)
        assert not np.allclose(m_n.field_body, m_d.field_body, atol=0.05)

    def test_field_magnitude_earth_like(self, mag):
        m = mag.measure(0., 0., 0.)
        assert 0.1 < np.linalg.norm(m.field_body) < 1.5


class TestLiDARModel:
    @pytest.fixture
    def lidar(self):
        return LiDARModel(seed=SEED)

    def test_returns_measurement(self, lidar):
        m = lidar.measure(5.)
        assert isinstance(m.range_m, float)
        assert isinstance(m.valid,   bool)

    def test_valid_in_range(self, lidar):
        # Test multiple times — with high reflectivity at mid range must be valid
        results = [lidar.measure(3., surface_reflectivity=0.9) for _ in range(10)]
        assert any(m.valid for m in results)

    def test_invalid_below_min(self, lidar):
        m = lidar.measure(0.05)
        assert not m.valid

    def test_invalid_above_max(self, lidar):
        m = lidar.measure(15.)
        assert not m.valid

    def test_noise_small_at_close_range(self, lidar):
        ms    = [lidar.measure(3.) for _ in range(200)]
        valid = [m.range_m for m in ms if m.valid]
        if valid:
            assert abs(np.mean(valid) - 3.) < 0.3
            assert np.std(valid) < 0.2

    def test_snr_positive_in_range(self, lidar):
        m = lidar.measure(3., 0.8)
        if m.valid:
            assert m.snr > 0

    def test_noise_increases_with_range(self, lidar):
        near = [lidar.measure(1.)  for _ in range(200)]
        far  = [lidar.measure(10.) for _ in range(200)]
        v_near = [m.range_m for m in near if m.valid]
        v_far  = [m.range_m for m in far  if m.valid]
        if v_near and v_far:
            assert np.std(v_far) >= np.std(v_near)


class TestComplementaryFilter:
    @pytest.fixture
    def cf(self):
        return ComplementaryFilter(alpha=0.98, dt=0.005)

    def test_initial_state_zero(self, cf):
        assert cf._roll == cf._pitch == cf._alt == 0.

    def test_level_flight_converges(self, cf):
        acc = np.array([0., 0., -9.81])
        for _ in range(200):
            roll, pitch = cf.update_attitude(acc, np.zeros(3))
        assert abs(roll) < 0.1 and abs(pitch) < 0.1

    def test_roll_direction_correct(self, cf):
        # Gravity component in +Y → positive roll
        acc = np.array([0., 4.9, -8.5])
        for _ in range(300):
            roll, _ = cf.update_attitude(acc, np.zeros(3))
        assert roll > 0.

    def test_altitude_converges(self, cf):
        for _ in range(500):
            alt = cf.update_altitude(50., 0.)
        assert abs(alt - 50.) < 3.

    def test_altitude_blends_velocity(self, cf):
        for _ in range(50):
            alt = cf.update_altitude(0., 1., dt=0.1)
        assert alt > 0.5

    def test_reset(self, cf):
        for _ in range(100):
            cf.update_attitude(np.array([0., 0., -9.81]), np.zeros(3))
            cf.update_altitude(25., 0.)
        cf.reset()
        assert cf._roll == cf._pitch == cf._alt == 0.

    def test_high_accel_ignored(self, cf):
        r0 = cf._roll
        cf.update_attitude(np.array([20., 0., -9.81]), np.zeros(3))
        assert abs(cf._roll - r0) < math.radians(5.)


class TestSensorFusionIntegration:
    def test_full_pipeline_no_crash(self):
        from drone_sdk.state_manager import StateStore, StateFactory, VehicleConfig
        StateStore.destroy_all()
        try:
            store = StateStore.create("s_drone", VehicleConfig("s_drone"))
            store.update(StateFactory.create_initial("s_drone"))
            imu = IMUModel(seed=SEED)
            gps = GPSModel(seed=SEED)
            for _ in range(10):
                m_i = imu.measure(np.array([0., 0., -9.81]), np.zeros(3), 0.005)
                m_g = gps.measure(np.array([0., 0., -10.]), np.zeros(3))
                store.update_partial(imu.to_state_update(m_i, "s_drone"))
                store.update_partial(gps.to_state_update(m_g, "s_drone"))
            latest = store.get_latest()
            assert latest is not None
            assert math.isfinite(latest.ax)
        finally:
            StateStore.destroy_all()

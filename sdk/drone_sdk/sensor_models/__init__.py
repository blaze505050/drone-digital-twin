"""
drone_sdk.sensor_models
=======================
Physics-based sensor models with realistic noise, bias drift, and
calibration errors — Module 9 of the UAV Digital Twin Platform.

Every model follows IEEE Standard 952 / Allan variance notation.
This allows comparing simulated sensor performance against real datasheet
specs (e.g. ICM-42688-P, u-blox M9N, MS5611, HMC5983).

Sensors implemented
-------------------
IMUModel          — accelerometer + gyroscope (Allan variance noise, bias walk)
GPSModel          — position + velocity (DOP-scaled noise, fix degradation)
BarometerModel    — altitude from pressure (ISA model + quantisation)
MagnetometerModel — 3-axis compass (hard/soft iron calibration errors)
LiDARModel        — 1-D rangefinder (beam divergence, multi-return)
SensorFusion      — complementary filter combining IMU + GPS + baro

Use in SITL loop
----------------
    imu   = IMUModel()
    gps   = GPSModel()
    baro  = BarometerModel()

    for state in sim_loop():
        imu_meas  = imu.measure(state.acceleration_body(), state.angular_velocity_body(), dt)
        gps_meas  = gps.measure(state.position_ned(), state.velocity_ned())
        baro_meas = baro.measure(-state.z)   # altitude = -NED_z

The output DroneStateUpdate can be fed directly into a StateStore.

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from drone_sdk.state_manager import DataSource, DroneStateUpdate


# ─────────────────────────────────────────────────────────────────────────────
#  Noise helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gauss(sigma: float, shape=()) -> np.ndarray:
    """Zero-mean Gaussian random variable with given std-dev."""
    return np.random.normal(0.0, sigma, shape)


def _random_walk(current: np.ndarray, sigma_per_sqrt_hz: float, dt: float) -> np.ndarray:
    """Discrete random walk: Δb ~ N(0, sigma² dt)."""
    return current + np.random.normal(0.0, sigma_per_sqrt_hz * math.sqrt(dt), current.shape)


# ─────────────────────────────────────────────────────────────────────────────
#  Measurement dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUMeasurement:
    """Raw IMU output as measured by the sensor."""
    timestamp_mono:     float
    accelerometer:      np.ndarray   # [ax, ay, az] body frame, m/s²
    gyroscope:          np.ndarray   # [p, q, r] body frame, rad/s
    temperature:        float        # °C
    acc_bias:           np.ndarray   # current bias estimate (for diagnostics)
    gyro_bias:          np.ndarray


@dataclass
class GPSMeasurement:
    """Raw GPS/GNSS output."""
    timestamp_mono:  float
    position_ned:    np.ndarray    # [x, y, z] NED, metres
    velocity_ned:    np.ndarray    # [vx, vy, vz] NED, m/s
    lat:             float         # degrees
    lon:             float
    alt_msl:         float         # metres
    fix_type:        int           # 0=no, 2=2D, 3=3D, 5=RTK
    hdop:            float
    vdop:            float
    n_satellites:    int
    hacc:            float         # horizontal accuracy, m (1-sigma)
    vacc:            float         # vertical accuracy, m


@dataclass
class BarometerMeasurement:
    """Barometric altitude and pressure."""
    timestamp_mono:  float
    pressure_pa:     float     # Pascal
    altitude_m:      float     # m above start altitude
    temperature_c:   float


@dataclass
class MagnetometerMeasurement:
    """3-axis magnetic field measurement."""
    timestamp_mono:  float
    field_body:      np.ndarray   # [bx, by, bz] in body frame, Gauss
    heading_deg:     float        # magnetic heading (uncorrected for declination)


@dataclass
class LiDARMeasurement:
    """1-D rangefinder measurement."""
    timestamp_mono:  float
    range_m:         float     # metres
    snr:             float     # signal-to-noise ratio
    valid:           bool


# ─────────────────────────────────────────────────────────────────────────────
#  IMU Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUConfig:
    """
    Sensor configuration based on Allan variance characterisation.

    Defaults match ICM-42688-P (Invensense) — widely used in PX4 platforms.

    All spectral densities follow the convention:
        noise = density * sqrt(bandwidth)
    """
    # ── Accelerometer ────────────────────────────────────────────────────────
    acc_noise_density:    float = 70e-6      # g/√Hz  (ICM-42688: 70 µg/√Hz)
    acc_bias_instability: float = 20e-6      # g (bias instability @ flicker)
    acc_random_walk:      float = 0.004      # m/s/√hr (bias random walk)
    acc_scale_error:      float = 0.001      # fractional (0.1%)
    acc_cross_axis:       float = 0.005      # rad (misalignment)

    # ── Gyroscope ─────────────────────────────────────────────────────────────
    gyro_noise_density:   float = 0.015e-3   # °/s/√Hz → rad/s/√Hz
    gyro_bias_instability: float = 1.0e-3    # °/s
    gyro_random_walk:     float = 0.1e-3     # °/s/√hr
    gyro_scale_error:     float = 0.001
    gyro_cross_axis:      float = 0.005      # rad

    # ── Environment ───────────────────────────────────────────────────────────
    temperature_c:        float = 25.0
    g_sensitivity:        float = 0.1e-3     # gyro g-sensitivity °/s/g

    # ── Quantisation ─────────────────────────────────────────────────────────
    acc_resolution:       float = 0.0        # m/s² (0 = infinite)
    gyro_resolution:      float = 0.0        # rad/s

    @property
    def acc_noise_density_mps2_sqhz(self) -> float:
        return self.acc_noise_density * 9.80665  # g/√Hz → m/s²/√Hz

    @property
    def gyro_noise_density_rads_sqhz(self) -> float:
        return math.radians(self.gyro_noise_density * 1000)  # already rad/s/√Hz


class IMUModel:
    """Realistic IMU sensor model with Allan-variance-based noise.

    Simulates:
    - White noise (velocity random walk for accelerometer, ARW for gyro)
    - Bias instability (flicker noise)
    - Bias random walk (slowly drifting bias)
    - Scale factor error
    - Cross-axis coupling
    - g-sensitivity (gyro affected by linear acceleration)
    - Quantisation

    Usage::

        imu = IMUModel()
        meas = imu.measure(
            true_accel = np.array([0.1, 0.0, -9.81]),
            true_omega = np.array([0.0, 0.0, 0.01]),
            dt=0.005,
        )
        print(meas.accelerometer)   # noisy version of true_accel
    """

    def __init__(
        self,
        config: Optional[IMUConfig] = None,
        seed:   Optional[int]       = None,
    ) -> None:
        self._cfg = config or IMUConfig()
        if seed is not None:
            np.random.seed(seed)

        # Bias states (time-correlated)
        self._acc_bias  = np.zeros(3)
        self._gyro_bias = np.zeros(3)

    def measure(
        self,
        true_accel: np.ndarray,
        true_omega: np.ndarray,
        dt:         float,
    ) -> IMUMeasurement:
        """Generate a noisy IMU measurement from true values.

        Args:
            true_accel: True body-frame acceleration (m/s²). Include gravity.
            true_omega: True body-frame angular velocity (rad/s).
            dt:         Integration time step (seconds).

        Returns:
            IMUMeasurement with realistic noise applied.
        """
        cfg = self._cfg
        bw  = 1.0 / (2 * dt)   # Nyquist bandwidth for this dt

        # ── Bias random walk ──────────────────────────────────────────────────
        acc_walk  = cfg.acc_random_walk / 3600.0 * math.sqrt(dt)  # m/s/√hr → m/s²
        gyro_walk = math.radians(cfg.gyro_random_walk / 3600.0) * math.sqrt(dt)

        self._acc_bias  = _random_walk(self._acc_bias,  acc_walk,  1.0)
        self._gyro_bias = _random_walk(self._gyro_bias, gyro_walk, 1.0)

        # ── White noise ───────────────────────────────────────────────────────
        acc_white  = _gauss(cfg.acc_noise_density_mps2_sqhz * math.sqrt(bw), (3,))
        gyro_white = _gauss(cfg.gyro_noise_density_rads_sqhz * math.sqrt(bw), (3,))

        # ── Scale factor error ────────────────────────────────────────────────
        acc_scaled  = true_accel  * (1.0 + _gauss(cfg.acc_scale_error, (3,)))
        gyro_scaled = true_omega  * (1.0 + _gauss(cfg.gyro_scale_error, (3,)))

        # ── g-sensitivity (gyro picks up linear accel) ────────────────────────
        g_sens = cfg.g_sensitivity * np.linalg.norm(true_accel) / 9.80665

        # ── Compose measurement ───────────────────────────────────────────────
        acc_meas  = acc_scaled  + self._acc_bias  + acc_white
        gyro_meas = gyro_scaled + self._gyro_bias + gyro_white + g_sens

        # ── Quantisation ──────────────────────────────────────────────────────
        if cfg.acc_resolution > 0:
            acc_meas = np.round(acc_meas / cfg.acc_resolution) * cfg.acc_resolution
        if cfg.gyro_resolution > 0:
            gyro_meas = np.round(gyro_meas / cfg.gyro_resolution) * cfg.gyro_resolution

        return IMUMeasurement(
            timestamp_mono = time.monotonic(),
            accelerometer  = acc_meas,
            gyroscope      = gyro_meas,
            temperature    = cfg.temperature_c + _gauss(0.5),
            acc_bias       = self._acc_bias.copy(),
            gyro_bias      = self._gyro_bias.copy(),
        )

    def to_state_update(
        self,
        meas: IMUMeasurement,
        vehicle_id: str,
    ) -> DroneStateUpdate:
        upd = DroneStateUpdate(
            vehicle_id     = vehicle_id,
            source         = DataSource.MANUAL,
            timestamp_wall = time.time(),
            timestamp_mono = meas.timestamp_mono,
        )
        upd.acceleration     = meas.accelerometer
        upd.angular_velocity = meas.gyroscope
        return upd

    def reset_bias(self) -> None:
        """Reset bias state (simulate power-on re-calibration)."""
        self._acc_bias  = np.zeros(3)
        self._gyro_bias = np.zeros(3)


# ─────────────────────────────────────────────────────────────────────────────
#  GPS Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPSConfig:
    """GPS sensor configuration. Defaults match u-blox M9N (typical UAV GPS)."""
    update_rate_hz:      float = 5.0
    base_hacc_m:         float = 1.5     # Horizontal accuracy (1-sigma) m
    base_vacc_m:         float = 2.5     # Vertical accuracy
    velocity_noise_ms:   float = 0.1     # m/s
    hdop_nominal:        float = 0.9
    vdop_nominal:        float = 1.2
    n_satellites_nominal: int  = 14
    fix_probability:     float = 1.0     # Probability of having a fix
    home_lat:            float = 47.397742
    home_lon:            float = 8.545594
    home_alt:            float = 488.0

    # Multipath / environment degradation (0.0=clear sky, 1.0=urban canyon)
    environment_factor:  float = 0.0


class GPSModel:
    """Realistic GPS/GNSS model with DOP-scaled noise and satellite dropout.

    Usage::

        gps = GPSModel()
        meas = gps.measure(
            position_ned = np.array([10.0, 5.0, -20.0]),
            velocity_ned = np.array([1.0, 0.5, 0.0]),
        )
    """

    def __init__(
        self,
        config: Optional[GPSConfig] = None,
        seed:   Optional[int]       = None,
    ) -> None:
        self._cfg = config or GPSConfig()
        if seed is not None:
            np.random.seed(seed)
        self._last_fix_time = 0.0
        self._fix_type = 3

    def measure(
        self,
        position_ned: np.ndarray,
        velocity_ned: np.ndarray,
    ) -> GPSMeasurement:
        """Generate a GPS measurement from the true NED position.

        Args:
            position_ned: True position [x, y, z] NED, metres from home.
            velocity_ned: True velocity [vx, vy, vz] NED, m/s.

        Returns:
            GPSMeasurement with DOP-scaled noise.
        """
        cfg = self._cfg

        # Degradation from environment
        env_mult = 1.0 + cfg.environment_factor * 5.0
        hacc = cfg.base_hacc_m * env_mult * cfg.hdop_nominal
        vacc = cfg.base_vacc_m * env_mult * cfg.vdop_nominal
        hdop = cfg.hdop_nominal * env_mult
        vdop = cfg.vdop_nominal * env_mult

        # Satellite count (varies slightly)
        n_sats = max(4, int(cfg.n_satellites_nominal + _gauss(2.0)))

        # Fix type degradation in poor environment
        fix = 3 if np.random.random() < cfg.fix_probability else 0

        # Add noise to NED position
        noise_h = _gauss(hacc, (2,))
        noise_v = _gauss(vacc)
        pos_noisy = position_ned.copy()
        pos_noisy[0] += noise_h[0]
        pos_noisy[1] += noise_h[1]
        pos_noisy[2] += noise_v

        # Add noise to velocity
        vel_noisy = velocity_ned + _gauss(cfg.velocity_noise_ms, (3,))

        # Convert NED offset to geodetic
        lat, lon, alt = self._ned_to_geodetic(pos_noisy, cfg)

        return GPSMeasurement(
            timestamp_mono = time.monotonic(),
            position_ned   = pos_noisy,
            velocity_ned   = vel_noisy,
            lat            = lat,
            lon            = lon,
            alt_msl        = alt,
            fix_type       = fix,
            hdop           = round(hdop, 2),
            vdop           = round(vdop, 2),
            n_satellites   = n_sats,
            hacc           = round(hacc, 2),
            vacc           = round(vacc, 2),
        )

    def to_state_update(
        self,
        meas: GPSMeasurement,
        vehicle_id: str,
    ) -> DroneStateUpdate:
        upd = DroneStateUpdate(
            vehicle_id     = vehicle_id,
            source         = DataSource.MANUAL,
            timestamp_wall = time.time(),
            timestamp_mono = meas.timestamp_mono,
        )
        upd.position        = meas.position_ned
        upd.velocity        = meas.velocity_ned
        upd.gps_lat         = meas.lat
        upd.gps_lon         = meas.lon
        upd.gps_alt_msl     = meas.alt_msl
        upd.gps_fix_type    = meas.fix_type
        upd.gps_satellites  = meas.n_satellites
        upd.gps_hdop        = meas.hdop
        upd.gps_vdop        = meas.vdop
        upd.altitude_agl    = -meas.position_ned[2]  # NED z → AGL
        return upd

    @staticmethod
    def _ned_to_geodetic(
        ned: np.ndarray,
        cfg: GPSConfig,
    ) -> Tuple[float, float, float]:
        dlat = ned[0] / 111_111.0
        dlon = ned[1] / (111_111.0 * math.cos(math.radians(cfg.home_lat)))
        return (
            cfg.home_lat + dlat,
            cfg.home_lon + dlon,
            cfg.home_alt - ned[2],
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Barometer Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BarometerConfig:
    """Barometer configuration. Defaults match MS5611 (Measurement Specialties)."""
    noise_pa:          float = 0.65    # Pa (1-sigma white noise)
    bias_pa:           float = 0.0     # Initial bias
    drift_pa_per_s:    float = 0.001   # Long-term drift
    quantisation_pa:   float = 0.012   # LSB resolution ≈ ~10 cm alt
    temperature_c:     float = 25.0
    reference_alt_m:   float = 0.0     # Altitude at power-on


class BarometerModel:
    """MS5611-like barometer with noise, drift, and temperature compensation.

    Usage::

        baro = BarometerModel()
        meas = baro.measure(altitude_agl=25.0)
        print(f"Measured altitude: {meas.altitude_m:.2f} m")
    """

    # ISA constants
    _P0  = 101_325.0   # Pa (sea-level pressure)
    _T0  = 288.15      # K  (sea-level temperature)
    _L   = 0.0065      # K/m (temperature lapse rate)
    _R   = 8.314462    # J/(mol·K)
    _M   = 0.0289644   # kg/mol
    _g   = 9.80665

    def __init__(
        self,
        config: Optional[BarometerConfig] = None,
        seed:   Optional[int]             = None,
    ) -> None:
        self._cfg   = config or BarometerConfig()
        self._bias  = self._cfg.bias_pa
        self._t0    = time.monotonic()
        if seed is not None:
            np.random.seed(seed)

    def measure(self, altitude_agl: float) -> BarometerMeasurement:
        """Generate a barometer measurement for the given AGL altitude.

        Args:
            altitude_agl: True altitude above ground level (metres).

        Returns:
            BarometerMeasurement.
        """
        cfg = self._cfg
        alt_msl = altitude_agl + cfg.reference_alt_m

        # True pressure from ISA formula
        true_pressure = self._P0 * (
            1.0 - self._L * alt_msl / self._T0
        ) ** (self._g * self._M / (self._R * self._L))

        # Drift
        elapsed = time.monotonic() - self._t0
        drift   = cfg.drift_pa_per_s * elapsed

        # Total measured pressure
        noise        = float(_gauss(cfg.noise_pa))
        measured_p   = true_pressure + self._bias + drift + noise
        measured_p   = round(measured_p / cfg.quantisation_pa) * cfg.quantisation_pa

        # Convert back to altitude
        alt_measured = (
            self._T0 / self._L
        ) * (1.0 - (measured_p / self._P0) ** (self._R * self._L / (self._g * self._M)))
        alt_measured -= cfg.reference_alt_m

        temp = cfg.temperature_c + float(_gauss(0.3))

        return BarometerMeasurement(
            timestamp_mono = time.monotonic(),
            pressure_pa    = measured_p,
            altitude_m     = alt_measured,
            temperature_c  = temp,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Magnetometer Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MagnetometerConfig:
    """Magnetometer config. Defaults match HMC5983."""
    field_strength_gauss: float = 0.5         # Total field strength (typical)
    inclination_deg:      float = -20.0        # Magnetic dip angle (Bengaluru ≈ -20°)
    declination_deg:      float = 1.5          # Magnetic declination
    noise_gauss:          float = 1e-4         # Measurement noise (1-sigma)
    hard_iron:            np.ndarray = field(
        default_factory=lambda: np.zeros(3)    # Hard iron offset (Gauss)
    )
    soft_iron:            np.ndarray = field(
        default_factory=lambda: np.eye(3)      # Soft iron matrix
    )


class MagnetometerModel:
    """3-axis magnetometer with hard/soft iron calibration errors.

    Usage::

        mag = MagnetometerModel()
        meas = mag.measure(roll=0.0, pitch=0.0, yaw=1.57)
        print(f"Heading: {meas.heading_deg:.1f}°")
    """

    def __init__(
        self,
        config: Optional[MagnetometerConfig] = None,
        seed:   Optional[int]               = None,
    ) -> None:
        self._cfg = config or MagnetometerConfig()
        if seed is not None:
            np.random.seed(seed)

        # Pre-compute NED earth field vector
        cfg = self._cfg
        incl = math.radians(cfg.inclination_deg)
        decl = math.radians(cfg.declination_deg)
        H    = cfg.field_strength_gauss * math.cos(incl)
        B    = cfg.field_strength_gauss * math.sin(incl)
        # NED: [H cos(decl), H sin(decl), B]
        self._earth_ned = np.array([
            H * math.cos(decl),
            H * math.sin(decl),
            B,
        ])

    def measure(
        self,
        roll:  float,
        pitch: float,
        yaw:   float,
    ) -> MagnetometerMeasurement:
        """Generate a magnetometer measurement for the given Euler orientation.

        Args:
            roll:  Roll angle, radians.
            pitch: Pitch angle, radians.
            yaw:   Yaw angle, radians (true heading).

        Returns:
            MagnetometerMeasurement.
        """
        # Rotation matrix NED → body (ZYX Euler)
        R = self._euler_to_dcm(roll, pitch, yaw)

        # Transform earth field to body frame
        true_body = R @ self._earth_ned

        # Apply soft iron and hard iron calibration errors
        cfg    = self._cfg
        meas_b = cfg.soft_iron @ true_body + cfg.hard_iron

        # Add noise
        meas_b += _gauss(cfg.noise_gauss, (3,))

        # Compute heading from measured field (flat earth assumption)
        heading = math.degrees(math.atan2(-meas_b[1], meas_b[0])) % 360.0

        return MagnetometerMeasurement(
            timestamp_mono = time.monotonic(),
            field_body     = meas_b,
            heading_deg    = heading,
        )

    @staticmethod
    def _euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX Euler angles to direction cosine matrix (NED → body)."""
        cr, sr = math.cos(roll),  math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw),   math.sin(yaw)
        return np.array([
            [ cp*cy,            cp*sy,           -sp    ],
            [ sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp ],
            [ cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp ],
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiDARConfig:
    """1-D rangefinder configuration. Defaults match TFmini Plus."""
    update_rate_hz:   float = 100.0
    min_range_m:      float = 0.1
    max_range_m:      float = 12.0
    noise_m:          float = 0.02     # 1-sigma measurement noise (m)
    beam_divergence:  float = 0.003    # radians (half-angle)
    snr_nominal:      float = 20.0     # dB
    snr_min_valid:    float = 5.0      # Below this = invalid reading


class LiDARModel:
    """1-D LiDAR rangefinder model for altitude sensing.

    Handles:
    - Range-dependent noise (farther targets = noisier)
    - Minimum/maximum range gates
    - SNR-based validity
    - Out-of-range returns (noise)

    Usage::

        lidar = LiDARModel()
        meas  = lidar.measure(true_range=5.0, surface_reflectivity=0.7)
    """

    def __init__(
        self,
        config: Optional[LiDARConfig] = None,
        seed:   Optional[int]         = None,
    ) -> None:
        self._cfg = config or LiDARConfig()
        if seed is not None:
            np.random.seed(seed)

    def measure(
        self,
        true_range:           float,
        surface_reflectivity: float = 0.8,
    ) -> LiDARMeasurement:
        """Generate a LiDAR measurement.

        Args:
            true_range:           True distance to surface (metres).
            surface_reflectivity: Lambertian reflectivity [0, 1].

        Returns:
            LiDARMeasurement. valid=False if outside measurement envelope.
        """
        cfg = self._cfg

        # Range-dependent SNR: highest at near range, falls off with r²
        if true_range > 0 and surface_reflectivity > 0:
            snr = cfg.snr_nominal * surface_reflectivity * (cfg.max_range_m / max(true_range, 0.1)) ** 2
            snr = max(0.0, snr)
        else:
            snr = 0.0

        valid = (
            cfg.min_range_m <= true_range <= cfg.max_range_m
            and snr >= cfg.snr_min_valid
        )

        if valid:
            # Noise increases with range (range walk noise)
            noise_sigma = cfg.noise_m * (1.0 + true_range / cfg.max_range_m)
            measured    = true_range + float(_gauss(noise_sigma))
            measured    = max(cfg.min_range_m, min(cfg.max_range_m, measured))
        else:
            measured = cfg.max_range_m + float(_gauss(cfg.noise_m * 3))

        return LiDARMeasurement(
            timestamp_mono = time.monotonic(),
            range_m        = measured,
            snr            = snr,
            valid          = valid,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Sensor Fusion — complementary filter
# ─────────────────────────────────────────────────────────────────────────────

class ComplementaryFilter:
    """Simple complementary filter fusing IMU accelerometer with GPS/baro.

    Altitude fusion:
        alt_fused = α * (alt_fused + vz * dt) + (1-α) * alt_baro

    Attitude fusion (using accelerometer as low-frequency reference):
        roll/pitch = α * (gyro integration) + (1-α) * (acc-derived)

    This is the same filter used in many commercial flight controllers
    (Betaflight, ArduPilot) at their core before the full EKF.

    Usage::

        cf = ComplementaryFilter(alpha=0.98, dt=0.005)
        roll, pitch = cf.update_attitude(accel, gyro)
        alt = cf.update_altitude(baro_alt, vz, dt)
    """

    def __init__(self, alpha: float = 0.98, dt: float = 0.005) -> None:
        self._alpha = alpha
        self._dt    = dt
        self._roll  = 0.0
        self._pitch = 0.0
        self._alt   = 0.0

    def update_attitude(
        self,
        accel: np.ndarray,   # [ax, ay, az] m/s²
        gyro:  np.ndarray,   # [p, q, r] rad/s
    ) -> Tuple[float, float]:
        """Fuse accelerometer tilt estimate with gyro integration.

        Returns:
            (roll, pitch) in radians.
        """
        dt = self._dt
        # Gyro integration
        self._roll  += gyro[0] * dt
        self._pitch += gyro[1] * dt

        # Accelerometer-derived roll/pitch (only valid at low accelerations)
        acc_norm = np.linalg.norm(accel)
        if 0.5 * 9.81 < acc_norm < 1.5 * 9.81:  # within 50% of gravity
            acc_roll  = math.atan2(accel[1], math.sqrt(accel[0]**2 + accel[2]**2))
            acc_pitch = math.atan2(-accel[0], math.sqrt(accel[1]**2 + accel[2]**2))
            # Complementary blend
            self._roll  = self._alpha * self._roll  + (1 - self._alpha) * acc_roll
            self._pitch = self._alpha * self._pitch + (1 - self._alpha) * acc_pitch

        return self._roll, self._pitch

    def update_altitude(
        self,
        baro_alt: float,
        vz:       float,
        dt:       Optional[float] = None,
    ) -> float:
        """Fuse barometer with vertical velocity integral.

        Args:
            baro_alt: Barometer altitude (m).
            vz:       Vertical velocity from IMU integration (m/s, up positive).
            dt:       Time step (uses instance default if None).

        Returns:
            Fused altitude estimate (m).
        """
        step = dt if dt is not None else self._dt
        # Predict
        self._alt += vz * step
        # Correct
        self._alt = self._alpha * self._alt + (1 - self._alpha) * baro_alt
        return self._alt

    def reset(self) -> None:
        self._roll = self._pitch = self._alt = 0.0

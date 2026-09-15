"""
drone_sdk.flight_simulator.environment
=======================================
Atmospheric, Wind, and Gravity Environment Models.

Provides:
  1. ISA Standard Atmosphere (temperature, pressure, density vs altitude)
  2. Dryden Turbulence Model (MIL-F-8785C) for realistic wind gusts
  3. Steady wind field with altitude gradient
  4. WGS-84 latitude-dependent gravity

References:
  - International Standard Atmosphere (ISA), ISO 2533:1975.
  - MIL-F-8785C, "Flying Qualities of Piloted Airplanes", 1980.
  - Dryden wind turbulence model, MIL-STD-1797A.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  ISA Standard Atmosphere
# ─────────────────────────────────────────────────────────────────────────────

# ISA constants
ISA_T0 = 288.15       # Sea level temperature (K)
ISA_P0 = 101325.0     # Sea level pressure (Pa)
ISA_RHO0 = 1.225      # Sea level density (kg/m³)
ISA_LAPSE = -0.0065   # Temperature lapse rate (K/m) in troposphere
ISA_R = 287.058        # Specific gas constant for dry air (J/(kg·K))
ISA_G0 = 9.80665       # Standard gravity (m/s²)


@dataclass
class AtmosphereState:
    """Atmospheric conditions at a given altitude."""
    temperature_k: float
    pressure_pa: float
    density_kgm3: float
    speed_of_sound_mps: float
    altitude_m: float


def isa_atmosphere(altitude_msl_m: float) -> AtmosphereState:
    """ISA Standard Atmosphere model for the troposphere (0–11 km).

    Args:
        altitude_msl_m: Altitude above mean sea level (m).

    Returns:
        AtmosphereState with temperature, pressure, density, and speed of sound.
    """
    h = max(0.0, min(altitude_msl_m, 11000.0))

    T = ISA_T0 + ISA_LAPSE * h
    P = ISA_P0 * (T / ISA_T0) ** (-ISA_G0 / (ISA_LAPSE * ISA_R))
    rho = P / (ISA_R * T)

    # Speed of sound: a = √(γ·R·T), γ = 1.4 for air
    a = math.sqrt(1.4 * ISA_R * T)

    return AtmosphereState(
        temperature_k=T,
        pressure_pa=P,
        density_kgm3=rho,
        speed_of_sound_mps=a,
        altitude_m=altitude_msl_m,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  WGS-84 Gravity Model
# ─────────────────────────────────────────────────────────────────────────────

def wgs84_gravity(latitude_deg: float = 0.0, altitude_m: float = 0.0) -> float:
    """WGS-84 gravity model (Somigliana formula).

    Args:
        latitude_deg: Geodetic latitude (degrees).
        altitude_m: Altitude above ellipsoid (m).

    Returns:
        Local gravitational acceleration (m/s²).
    """
    lat_rad = math.radians(latitude_deg)
    sin2_lat = math.sin(lat_rad) ** 2

    # Somigliana formula (normal gravity on the ellipsoid)
    g_0 = 9.7803253359 * (1.0 + 0.00193185265241 * sin2_lat) / math.sqrt(1.0 - 0.00669437999014 * sin2_lat)

    # Free-air correction (altitude above ellipsoid)
    g = g_0 * (1.0 - 2.0 * altitude_m / 6378137.0)

    return max(9.0, g)  # Clamp to avoid nonsensical values


# ─────────────────────────────────────────────────────────────────────────────
#  Wind Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindField:
    """Configurable wind model with steady component and Dryden turbulence.

    Attributes:
        steady_wind_ned: Constant wind vector in NED (m/s).
        wind_gradient_factor: Altitude-dependent wind scaling (log profile).
        turbulence_intensity: Dryden turbulence intensity (light=0.5, moderate=1.5, severe=3.0).
        enabled: Whether wind is active.
    """
    steady_wind_ned: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    wind_gradient_factor: float = 0.15   # Power-law exponent for altitude profile
    turbulence_intensity: float = 1.0    # σ_w (m/s) at reference altitude
    reference_altitude_m: float = 6.0    # Height for log wind profile
    enabled: bool = True

    # Internal Dryden state (first-order Markov)
    _turb_state: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        self.steady_wind_ned = np.asarray(self.steady_wind_ned, dtype=np.float64)

    def get_wind_ned(self, altitude_agl: float, dt: float = 0.01) -> np.ndarray:
        """Compute total wind vector (steady + turbulence) at given altitude.

        Args:
            altitude_agl: Altitude above ground (m, positive up).
            dt: Timestep for turbulence filter update.

        Returns:
            Wind velocity in NED frame (m/s).
        """
        if not self.enabled:
            return np.zeros(3, dtype=np.float64)

        # Steady wind with altitude profile (power law)
        h = max(0.5, altitude_agl)
        h_ref = max(1.0, self.reference_altitude_m)
        wind_scale = (h / h_ref) ** self.wind_gradient_factor
        steady = self.steady_wind_ned * wind_scale

        # Dryden turbulence (simplified first-order filter)
        if self.turbulence_intensity > 0.01:
            sigma = self.turbulence_intensity
            # Length scales (MIL-F-8785C for low altitude)
            Lu = max(10.0, h / 0.177 + 0.177 * h)
            Lv = Lu
            Lw = max(1.0, h)

            V_ref = max(1.0, np.linalg.norm(steady) + 5.0)

            # Time constants
            tau_u = Lu / V_ref
            tau_v = Lv / V_ref
            tau_w = Lw / V_ref

            # First-order Markov update with white noise driving
            noise = np.random.randn(3) * sigma
            for i, tau in enumerate([tau_u, tau_v, tau_w]):
                alpha_filt = dt / max(dt, tau + dt)
                self._turb_state[i] += alpha_filt * (noise[i] - self._turb_state[i])

            return steady + self._turb_state
        else:
            return steady.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Combined environment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Environment:
    """Complete flight environment model."""
    wind: WindField = field(default_factory=WindField)
    latitude_deg: float = 28.6139     # Default: New Delhi, India
    ground_altitude_msl_m: float = 0.0

    def get_conditions(self, pos_ned: np.ndarray, dt: float = 0.01) -> Tuple[float, float, np.ndarray]:
        """Get atmosphere density, gravity, and wind at current position.

        Args:
            pos_ned: Position in NED frame (m). z < 0 means above ground.
            dt: Timestep for wind turbulence update.

        Returns:
            (air_density, gravity, wind_ned)
        """
        altitude_agl = max(0.0, -pos_ned[2])
        altitude_msl = self.ground_altitude_msl_m + altitude_agl

        atmo = isa_atmosphere(altitude_msl)
        gravity = wgs84_gravity(self.latitude_deg, altitude_msl)
        wind_ned = self.wind.get_wind_ned(altitude_agl, dt)

        return atmo.density_kgm3, gravity, wind_ned

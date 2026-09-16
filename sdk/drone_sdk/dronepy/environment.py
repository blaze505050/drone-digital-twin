"""
dronepy.environment
===================
Atmospheric, Wind, and Environmental Conditions for DronePy Simulations.
Reuses the ISA 1976 and Dryden turbulence models from drone_sdk.flight_simulator.environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from drone_sdk.flight_simulator.environment import (
    ISA_G0,
    ISA_LAPSE,
    ISA_P0,
    ISA_R,
    ISA_RHO0,
    ISA_T0,
    AtmosphereState,
    isa_atmosphere,
    wgs84_gravity,
)


@dataclass
class Wind:
    """Wind velocity field model in NED coordinates."""
    steady_wind_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    wind_gradient_per_m: float = 0.0  # Increase in horizontal wind speed per metre altitude
    gust_amplitude: float = 0.0
    gust_duration_s: float = 0.0
    gust_start_s: float = 0.0
    gust_direction_ned: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    turbulence_model: Optional[DrydenTurbulence] = None

    def __post_init__(self) -> None:
        self.steady_wind_ned = np.asarray(self.steady_wind_ned, dtype=np.float64)
        self.gust_direction_ned = np.asarray(self.gust_direction_ned, dtype=np.float64)
        norm = np.linalg.norm(self.gust_direction_ned)
        if norm > 1e-6:
            self.gust_direction_ned /= norm

    @classmethod
    def constant(
        cls,
        speed: Optional[float] = None,
        heading_deg: float = 0.0,
        vertical_mps: float = 0.0,
        north: Optional[float] = None,
        east: Optional[float] = None,
        down: Optional[float] = None,
    ) -> "Wind":
        """Constant wind field blowing towards heading_deg (0=North, 90=East) or specified by components."""
        if north is not None or east is not None or down is not None:
            ned = np.array([float(north or 0.0), float(east or 0.0), float(down or 0.0)], dtype=np.float64)
        else:
            spd = float(speed or 0.0)
            rad = math.radians(heading_deg)
            ned = np.array([spd * math.cos(rad), spd * math.sin(rad), -vertical_mps], dtype=np.float64)
        return cls(steady_wind_ned=ned)

    @classmethod
    def linear_profile(
        cls,
        base_speed: float,
        gradient_per_m: float = 0.02,
        heading_deg: float = 0.0,
    ) -> "Wind":
        """Wind speed increasing linearly with altitude."""
        rad = math.radians(heading_deg)
        ned = np.array([base_speed * math.cos(rad), base_speed * math.sin(rad), 0.0], dtype=np.float64)
        return cls(steady_wind_ned=ned, wind_gradient_per_m=gradient_per_m)

    @classmethod
    def gust(
        cls,
        base_speed: float = 0.0,
        gust_amplitude: Optional[float] = None,
        magnitude: Optional[float] = None,
        gust_duration: Optional[float] = None,
        duration: Optional[float] = None,
        start_time: float = 10.0,
        heading_deg: float = 0.0,
        direction_deg: Optional[float] = None,
    ) -> "Wind":
        """Steady wind with a "1 - cosine" discrete gust."""
        h_deg = direction_deg if direction_deg is not None else heading_deg
        rad = math.radians(h_deg)
        ned = np.array([base_speed * math.cos(rad), base_speed * math.sin(rad), 0.0], dtype=np.float64)
        amp = float(magnitude if magnitude is not None else (gust_amplitude if gust_amplitude is not None else 5.0))
        dur = float(duration if duration is not None else (gust_duration if gust_duration is not None else 3.0))
        return cls(
            steady_wind_ned=ned,
            gust_amplitude=amp,
            gust_duration_s=dur,
            gust_start_s=start_time,
            gust_direction_ned=np.array([math.cos(rad), math.sin(rad), 0.0]),
        )

    def get_wind(self, altitude_agl: float = 0.0, time: float = 0.0) -> np.ndarray:
        """Evaluate wind vector [North, East, Down] in m/s."""
        return self.velocity_at(np.array([0.0, 0.0, -altitude_agl]), t=time)

    @classmethod
    def turbulence(cls, intensity: float = 1.0, altitude_m: float = 10.0, base_wind_mps: float = 3.0) -> "Wind":
        """Dryden continuous atmospheric turbulence field (MIL-F-8785C)."""
        dt = DrydenTurbulence(altitude_m=altitude_m, intensity=intensity)
        return cls(steady_wind_ned=np.array([base_wind_mps, 0.0, 0.0]), turbulence_model=dt)

    def velocity_at(self, pos_ned: np.ndarray, t: float = 0.0, dt: float = 0.01) -> np.ndarray:
        """Compute instantaneous wind velocity [v_north, v_east, v_down] in m/s."""
        alt_m = max(0.0, -float(pos_ned[2]))
        wind = self.steady_wind_ned.copy()

        # Altitude gradient
        if self.wind_gradient_per_m > 0 and alt_m > 0:
            h_speed = np.linalg.norm(wind[:2])
            if h_speed > 1e-6:
                scale = 1.0 + self.wind_gradient_per_m * alt_m
                wind[:2] *= scale

        # Discrete gust (1 - cos profile)
        if self.gust_amplitude > 0 and self.gust_duration_s > 0:
            if self.gust_start_s <= t <= (self.gust_start_s + self.gust_duration_s):
                tau = (t - self.gust_start_s) / self.gust_duration_s
                gust_mag = 0.5 * self.gust_amplitude * (1.0 - math.cos(2.0 * math.pi * tau))
                wind += self.gust_direction_ned * gust_mag

        # Dryden turbulence
        if self.turbulence_model is not None:
            # Assumes 10 m/s nominal forward airspeed
            turb = self.turbulence_model.step(10.0, max(0.5, alt_m), dt)
            wind += turb

        return wind


@dataclass
class Environment:
    """Atmospheric environment and gravity model for flight simulation."""
    altitude_msl_m: float = 0.0
    temperature_k: float = ISA_T0
    pressure_pa: float = ISA_P0
    density_kgm3: float = ISA_RHO0
    speed_of_sound_mps: float = 340.294
    gravity_mps2: float = ISA_G0
    latitude_deg: float = 45.0
    wind: Wind = field(default_factory=Wind)

    def __post_init__(self) -> None:
        if isinstance(self.wind, (list, tuple, np.ndarray)):
            self.wind = Wind(steady_wind_ned=np.asarray(self.wind, dtype=np.float64))

    @classmethod
    def standard_atmosphere(
        cls,
        altitude: float = 0.0,
        wind: Optional[Union[Wind, np.ndarray, List[float]]] = None,
        latitude_deg: float = 45.0,
    ) -> "Environment":
        """Create an environment based on the ISA 1976 standard atmosphere."""
        isa = isa_atmosphere(altitude)
        g = wgs84_gravity(latitude_deg, altitude)

        if wind is None:
            w_obj = Wind()
        elif isinstance(wind, Wind):
            w_obj = wind
        else:
            w_obj = Wind(steady_wind_ned=np.asarray(wind, dtype=np.float64))

        return cls(
            altitude_msl_m=altitude,
            temperature_k=isa.temperature_k,
            pressure_pa=isa.pressure_pa,
            density_kgm3=isa.density_kgm3,
            speed_of_sound_mps=isa.speed_of_sound_mps,
            gravity_mps2=g,
            latitude_deg=latitude_deg,
            wind=w_obj,
        )

    @classmethod
    def custom(
        cls,
        altitude: float = 0.0,
        temperature: float = 288.15,
        pressure: float = 101325.0,
        density: Optional[float] = None,
        wind: Optional[Union[Wind, np.ndarray, List[float]]] = None,
        gravity: float = ISA_G0,
    ) -> "Environment":
        """Create a custom environment with user-specified thermodynamics."""
        rho = density if density is not None else pressure / (ISA_R * temperature)
        a = math.sqrt(1.4 * ISA_R * temperature)

        if wind is None:
            w_obj = Wind()
        elif isinstance(wind, Wind):
            w_obj = wind
        else:
            w_obj = Wind(steady_wind_ned=np.asarray(wind, dtype=np.float64))

        return cls(
            altitude_msl_m=altitude,
            temperature_k=temperature,
            pressure_pa=pressure,
            density_kgm3=rho,
            speed_of_sound_mps=a,
            gravity_mps2=gravity,
            wind=w_obj,
        )

    def air_density_at(self, altitude_m: float) -> float:
        """Evaluate air density at a given local altitude above MSL."""
        if abs(altitude_m - self.altitude_msl_m) < 1.0:
            return self.density_kgm3
        return isa_atmosphere(altitude_m).density_kgm3

    def density_at(self, altitude_m: float) -> float:
        """Alias for air_density_at."""
        return self.air_density_at(altitude_m)

    @property
    def sea_level_pressure_pa(self) -> float:
        return ISA_P0

    @property
    def sea_level_temperature_k(self) -> float:
        return ISA_T0

    @property
    def sea_level_density_kgm3(self) -> float:
        return ISA_RHO0

    def wind_at(self, pos_ned: np.ndarray, t: float = 0.0, dt: float = 0.01) -> np.ndarray:
        """Query total wind vector [v_n, v_e, v_d] at position and time."""
        return self.wind.velocity_at(pos_ned, t, dt)

    def get_wind_ned(self, altitude_agl: float = 0.0, time: float = 0.0) -> np.ndarray:
        """Query total wind vector at altitude AGL and time."""
        return self.wind.velocity_at(np.array([0.0, 0.0, -altitude_agl]), t=time)

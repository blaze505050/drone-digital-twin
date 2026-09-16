# DronePy API Reference Guide

`dronepy` is a RocketPy-style multirotor UAV engineering simulation, stochastic analysis, and digital twin synchronization framework built upon the UAV Digital Twin platform.

```python
from dronepy import (
    Drone,
    Environment,
    Wind,
    Flight,
    Mission,
    Waypoint,
    FlightEvent,
    Scenario,
    Distribution,
    MonteCarlo,
    Sweep,
    TwinComparison,
    DigitalTwinSynchronizer,
    DigitalTwinCalibrator,
    FlightReplay,
    SafetyGateway,
    LaptopController,
)
```

---

## 1. Vehicle Abstraction: `Drone`

The `Drone` class represents the physical multirotor airframe, mass properties, motor group, propeller set, aerodynamics model, and battery pack.

### Factory Constructors
- `Drone.quadcopter(mass=1.5, arm_length=0.25, ...)`: Standard 4-rotor X-configuration quadcopter.
- `Drone.hexacopter(mass=2.5, arm_length=0.35, ...)`: Standard 6-rotor radial hexacopter.
- `Drone.octocopter(mass=4.0, arm_length=0.45, ...)`: Standard 8-rotor radial octocopter.

### Attributes and Properties
- `mass` / `mass_kg`: Total vehicle mass including any active payload (kg). Both getter and setter.
- `arm_length`: Distance from CG to motor center (m).
- `inertia_tensor`: 3x3 symmetric mass moment of inertia tensor about the vehicle CG ($kg \cdot m^2$).
- `center_of_gravity`: 3-element vector $[x_{cg}, y_{cg}, z_{cg}]$ in the body FRD frame (m).
- `motors`: `MotorGroup` containing individual `Motor` instances.
- `propellers`: List of `Propeller`, `UIUCPropeller`, or `BEMTPropeller` instances.
- `aerodynamics`: Configured `AerodynamicsModel` (`AnalyticalDrag`, `OpenFOAMAeroDatabase`, or `NeuralAeroModel`).
- `battery`: High-fidelity `PackECMModel` equivalent-circuit battery model.

### Key Methods
- `add_payload(mass_kg=0.4, offset_m=np.array([x, y, z]))`: Adds payload mass at a specific offset from the body origin. Dynamically updates the total mass, shifts the center of gravity, and recalculates the inertia tensor using the parallel-axis theorem.
- `release_payload(mass_released=None)`: Drops payload mass (or partial mass), linearly relaxing the inertia tensor back to nominal base levels and restoring nominal CG.
- `simulate(duration=30.0, dt=0.002, environment=None, ...)`: Convenience shortcut executing a 6-DOF simulation run and returning a `FlightResult`.
- `copy()`: Returns an independent deep copy of the `Drone` instance.

---

## 2. Atmosphere & Weather: `Environment` and `Wind`

### `Environment`
Represents the thermodynamic atmosphere and gravitational field.
- `Environment.standard_atmosphere(altitude=0.0, wind=None, latitude_deg=45.0)`: Builds an atmosphere based on the International Standard Atmosphere (ISA 1976 ISO 2533) coupled with WGS-84 Somigliana gravity.
- `Environment.custom(altitude=0.0, temperature=288.15, pressure=101325.0, density=1.225, ...)`: User-specified thermodynamic state.
- `density_at(altitude_m)`: Returns local air density $\rho$ at specified altitude MSL.
- `get_wind_ned(altitude_agl, time=0.0)`: Returns 3D wind velocity vector $[v_n, v_e, v_d]$ at specified AGL altitude and time.

### `Wind`
- `Wind.constant(speed=5.0, heading_deg=45.0, vertical_mps=0.0)`: Constant wind blowing towards heading (or specify `north`, `east`, `down`).
- `Wind.linear_profile(base_speed=3.0, gradient_per_m=0.02, heading_deg=0.0)`: Atmospheric shear profile increasing linearly with altitude.
- `Wind.gust(base_speed=0.0, magnitude=5.0, duration=3.0, start_time=10.0, direction_deg=0.0)`: Discrete "1 - cosine" gust perturbation.
- `Wind.turbulence(intensity=1.0, altitude_m=10.0)`: Continuous Dryden wind turbulence model conforming to MIL-F-8785C.

---

## 3. Actuator Models: `Motor` and `Propeller`

### `Motor`
- `max_rpm`: Maximum achievable rotation speed (RPM).
- `time_constant`: First-order response lag $\tau_m$ (s):
  $$\frac{d\Omega}{dt} = \frac{\Omega_{target} - \Omega}{\tau_m}$$
- `thrust_coefficient`: Static thrust scaling coefficient $k_t$ where $T = k_t \cdot (\Omega/1000)^2$.
- `voltage`: Nominal DC bus voltage.
- `step(command=1.0, bus_voltage=15.2, dt=0.002, cmd_rpm=None)`: Advances motor dynamic state, applies voltage scaling, and returns `(thrust_n, torque_nm, current_a, power_w)`.
- `failure(rpm_limit=None)` / `fail()`: Injects immediate catastrophic motor shutdown or RPM cap.
- `recover()`: Restores failed motor to operational status.
- `set_efficiency(factor)`: Multiplies motor output by degradation factor $[0.0, 1.0]$.

### `Propeller`
- `Propeller(diameter_in=10.0, pitch_in=4.5)`: Parametric quadratic propeller thrust and torque.
- `UIUCPropeller(propeller_name="apc_10x4.5")`: Wind-tunnel polar lookup tables from UIUC propeller database.
- `BEMTPropeller(diameter_in=10.0, pitch_in=4.5)`: High-fidelity iterative Blade Element Momentum Theory solver integrating sectional lift and drag with Prandtl tip-loss corrections.

---

## 4. Aerodynamics Models: `AerodynamicsModel`

All aerodynamic models inherit from `AerodynamicsModel` and implement `compute_forces_and_moments(v_air_body, omega_body, air_density)`:
- `AnalyticalDrag(cd_x=0.85, cd_y=0.85, cd_z=1.25)`: Directional quadratic parasitic drag and rotational damping.
- `OpenFOAMAeroDatabase()`: Table-lookup across multi-angle of attack ($\alpha$) and sideslip ($\beta$) wind-tunnel / CFD databases.
- `NeuralAeroModel()`: Physics-Informed Neural Network (PINN) surrogate model trained on Navier-Stokes solutions.

---

## 5. Mission & Flight Events: `Mission`, `FlightEvent`, `Scenario`

### `Mission` & `Waypoint`
- `mission.takeoff(altitude=10.0, speed=2.0)`: Climb to target AGL altitude.
- `mission.goto(north=50.0, east=25.0, down=-10.0, speed=8.0)`: Local 3D waypoint navigation.
- `mission.hover(duration=5.0)`: Loiter in place for specified seconds.
- `mission.rtl(return_altitude=15.0)`: Return to launch coordinates and transition to land.
- `mission.land(speed=1.5)`: Controlled descent and touch down.
- `mission.to_core_mission()`: Converts to platform-native `drone_sdk.mission_planner.Mission`.

### `Scenario` Builder
- `scenario.at(t).motor_failure(motor_index=0)`: Complete shutdown of designated motor at time $t$.
- `scenario.at(t).motor_efficiency(motor_index=0, efficiency=0.7)`: 30% degradation of designated motor at time $t$.
- `scenario.at(t).wind_change(speed_mps=8.0, direction_deg=90.0)`: Dynamic wind shift at time $t$.
- `scenario.at(t).payload_release()`: Cargo drop at time $t$.
- `scenario.at(t).gps_loss()`: GPS fix loss at time $t$.

---

## 6. Simulation & Telemetry: `Flight` and `FlightResult`

### `Flight`
Executes full 6-DOF nonlinear rigid-body dynamics forward in time.
```python
flight = Flight(
    drone=drone,
    environment=env,
    mission=mission,
    events=scenario,
    duration=30.0,
    dt=0.002,
)
result = flight.result
```

### `FlightResult`
Contains complete decimated time-series telemetry:
- `time`: Time array in seconds.
- `pos_ned`: $(N, 3)$ array of North-East-Down position (m).
- `altitude_agl`: $(N,)$ array of altitude above ground (m).
- `vel_ned`: $(N, 3)$ array of inertial velocities (m/s).
- `airspeed`: $(N,)$ magnitude of relative airspeed (m/s).
- `euler_deg`: $(N, 3)$ Euler angles [roll, pitch, yaw] in degrees.
- `quaternion`: $(N, 4)$ Hamilton attitude quaternions $[q_w, q_x, q_y, q_z]$.
- `omega_body`: $(N, 3)$ body angular velocity $[p, q, r]$ (rad/s).
- `motor_rpms`: $(N, M)$ array of individual motor rotation speeds (RPM).
- `battery_voltage`, `battery_current`, `battery_power`, `battery_energy_wh`, `battery_soc`: Complete electrical telemetry.
- `to_dataframe()`: Converts time-series into a Pandas DataFrame.
- `to_csv(filepath)` / `to_hdf5(filepath)`: Saves telemetry to disk.
- `plot_trajectory()`, `plot_attitude()`, `plot_velocity()`, `plot_motor_rpm()`, `plot_power()`, `plot_dashboard()`: High-quality engineering visualizations.

---

## 7. Stochastic Monte Carlo & Dispersion: `MonteCarlo` and `Distribution`

Quantifies flight dispersion under environmental and vehicle parameter uncertainties.

```python
mc = MonteCarlo(drone=drone, num_simulations=100, seed=42)
mc.add_parameter("mass", Distribution.normal(mean=1.5, std_dev=0.05))
mc.add_parameter("cd", Distribution.uniform(low=0.75, high=0.95))
mc.add_parameter("wind_speed", Distribution.triangular(low=0.0, mode=3.0, high=8.0))

mc_result = mc.run(duration=20.0, parallel=True)
summary = mc_result.summary()
print(f"CEP50: {summary['cep50_m']:.2f} m | CEP95: {summary['cep95_m']:.2f} m")

# Visualize landing dispersion and 3D trajectory envelopes
mc_result.plot_dispersion()
mc_result.plot_envelopes()
```

---

## 8. Digital Twin Synchronization & Calibration

### `TwinComparison`
Directly compares physical reality (or flight logs) against parallel digital twin predictions:
```python
comp = TwinComparison.compare(real=flight_real, twin=flight_twin)
print(f"Position RMSE: {comp.rmse_position*100:.1f} cm")
print(f"Velocity RMSE: {comp.rmse_velocity:.2f} m/s")
print(f"Max Position Error: {comp.max_position_error*100:.1f} cm")
comp.plot()
```

### `DigitalTwinCalibrator`
Performs bounded parameter estimation to match a digital twin model to real telemetry while enforcing persistent excitation checks and physical safety bounds:
```python
calibrator = DigitalTwinCalibrator(max_condition_number=100.0, max_parameter_shift_pct=15.0)
calibrated_drone, report = calibrator.calibrate(drone, flight_data)
print(report.summary())
```

---

## 9. Safety Gateway & Laptop Flight Control

Ensures commands are validated before reaching actuators and prevents accidental physical flight.

```python
gateway = SafetyGateway()
# Default mode is strictly SIMULATION_ONLY
assert gateway.mode == SystemExecutionMode.SIMULATION_ONLY

# Laptop keyboard/gamepad controller routes through gateway
controller = LaptopController(safety_gateway=gateway)
gateway.arm(pin_code=1234)

# Update stick commands from keyboard
controller.update_from_keyboard({"w": True, "shift": True})
report = controller.process_and_validate(current_pos, current_vel, current_euler)
```

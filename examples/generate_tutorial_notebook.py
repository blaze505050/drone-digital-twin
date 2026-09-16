"""
Generates examples/DronePy_Complete_Tutorial.ipynb
Covering all 18 engineering steps with rich markdown and executable cells.
"""
import json
from pathlib import Path

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# Title & Overview
add_md("""# 🚁 DronePy: RocketPy-style Multirotor Flight Dynamics & Digital Twin
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blaze505050/drone-digital-twin/blob/main/examples/DronePy_Complete_Tutorial.ipynb)

**DronePy** is a high-fidelity, modular 6-DOF multirotor UAV flight simulation, stochastic dispersion analysis, and digital twin synchronization framework built directly on top of the 31-subsystem UAV Digital Twin platform.

### Key Capabilities
- **RocketPy-Style Declarative API**: `from dronepy import Drone, Environment, Flight, MonteCarlo`
- **6-DOF Rigid-Body Dynamics**: Full non-linear quaternion equations of motion (NED/FRD conventions)
- **Actuator Dynamics**: First-order motor response lag, voltage-droop sensitivity, and failure injection
- **Aero Modeling**: Analytical drag, OpenFOAM CFD polar databases, and PINN neural surrogates
- **Atmospheric Physics**: ISA 1976 standard atmosphere, Dryden continuous turbulence, and 1-cosine gusts
- **Discrete Event Scheduling**: In-flight payload drops (recomputing mass, CG, and inertia) and motor failures
- **Stochastic Monte Carlo & Dispersion**: Landing footprints with CEP50 & CEP95 confidence circles
- **Digital Twin Synchronization**: Twin vs. Reality tracking residuals and online system identification
- **Safety Gateway**: Arming interlocks, geofence cylinders, and laptop keyboard/gamepad flight control
""")

# Step 1
add_md("""---
## Step 1: Environment Setup & Package Imports
We import the top-level declarative DronePy API and numerical utilities. If running in Google Colab, the cell installs DronePy automatically.""")
add_code("""# Setup Google Colab environment if executing in cloud
try:
    import dronepy
except ImportError:
    !pip install -q git+https://github.com/blaze505050/drone-digital-twin.git
    import dronepy

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Import the DronePy RocketPy-style API
import dronepy
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

print(f"Loaded DronePy version: {dronepy.__version__}")
""")

# Step 2
add_md("""---
## Step 2: Vehicle Geometry & Mass Definition
Define multirotors using built-in factories (`.quadcopter()`, `.hexacopter()`, `.octocopter()`) or custom geometry.""")
add_code("""# Create standard 1.5 kg X-configuration Quadcopter
quad = Drone.quadcopter(mass=1.5, arm_length=0.25)
print(f"Vehicle: {quad.name} | Mass: {quad.mass:.2f} kg | Motors: {quad.num_motors}")
print(f"Inertia Tensor Ixx={quad.inertia_tensor[0,0]:.4f}, Iyy={quad.inertia_tensor[1,1]:.4f}, Izz={quad.inertia_tensor[2,2]:.4f} kg·m²")

# Inspect motor mount coordinates in body FRD frame (X-forward, Y-right, Z-down)
for idx, pos in enumerate(quad.motor_positions):
    print(f"  Motor {idx+1}: Body pos = [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m")
""")

# Step 3
add_md("""---
## Step 3: Aerodynamic Models: Analytical, CFD & PINN Surrogate
DronePy supports modular aerodynamic providers implementing `AerodynamicsModel`:
1. `AnalyticalDrag`: Directional quadratic parasitic drag
2. `OpenFOAMAeroDatabase`: High-fidelity polar tables from CFD
3. `NeuralAeroModel`: Physics-Informed Neural Network (PINN) surrogate""")
add_code("""from dronepy import AnalyticalDrag, OpenFOAMAeroDatabase, NeuralAeroModel

# Directional quadratic parasitic drag model
aero_analytical = AnalyticalDrag(cd_x=0.85, cd_y=0.85, cd_z=1.25, area_frontal_m2=0.025)

# High-fidelity OpenFOAM CFD database lookup
aero_cfd = OpenFOAMAeroDatabase()

# Fast Physics-Informed Neural Network (PINN) surrogate
aero_pinn = NeuralAeroModel()

# Compare drag forces at 12 m/s relative airspeed
v_test = np.array([12.0, 0.0, 0.0])  # Forward flight
omega_test = np.zeros(3)

f_ana, _ = aero_analytical.compute_forces_and_moments(v_test, omega_test)
f_cfd, _ = aero_cfd.compute_forces_and_moments(v_test, omega_test)
f_pinn, _ = aero_pinn.compute_forces_and_moments(v_test, omega_test)

print(f"Analytical Drag Force: {f_ana[0]:.2f} N")
print(f"OpenFOAM CFD Force:    {f_cfd[0]:.2f} N")
print(f"PINN Surrogate Force:  {f_pinn[0]:.2f} N")
""")

# Step 4
add_md("""---
## Step 4: Actuator Physics: BEMT, Response Lag & Voltage Droop
Model brushless DC motors with first-order mechanical lag $\\tau_m$ and Blade Element Momentum Theory (BEMT).""")
add_code("""from dronepy import Motor, Propeller, BEMTPropeller

# Create a high-fidelity BEMT propeller
prop_bemt = BEMTPropeller(diameter_in=10.0, pitch_in=4.5)
thrust_bemt, torque_bemt, power_bemt = prop_bemt.compute_thrust_and_torque(rpm=7500.0)

print(f"BEMT Propeller at 7500 RPM:")
print(f"  Thrust: {thrust_bemt:.3f} N | Torque: {torque_bemt:.4f} N·m | Power: {power_bemt:.1f} W")

# Test motor spin-up step response
motor = Motor(max_rpm=9500.0, time_constant=0.035, voltage=16.0)
dt = 0.005
t_steps = np.arange(0.0, 0.20, dt)
rpms = []

for t in t_steps:
    motor.step(command=1.0, dt=dt, bus_voltage=15.2)  # Full throttle under 15.2V
    rpms.append(motor.current_rpm)

print(f"Motor RPM reached {rpms[-1]:.0f} RPM after 200 ms (tau = {motor.time_constant*1000:.0f} ms)")
""")

# Step 5
add_md("""---
## Step 5: Atmospheric Environment & Dynamic Wind Fields
Model altitude-dependent thermodynamic properties via ISA 1976 and wind perturbations (steady gradient, 1-cosine gusts, and Dryden turbulence).""")
add_code("""# Create an environment with a steady wind and a discrete 1-cosine gust
wind = Wind.gust(
    base_speed=3.0,
    magnitude=6.0,
    duration=2.0,
    start_time=5.0,
    direction_deg=45.0,  # Northeast
)

env = Environment.standard_atmosphere(altitude=100.0, wind=wind)

print(f"ISA Atmosphere at 100m MSL:")
print(f"  Pressure: {env.pressure_pa:.1f} Pa | Temp: {env.temperature_k:.2f} K | Density: {env.density_kgm3:.3f} kg/m³")
print(f"  Wind at t=4.0s (pre-gust):  {env.get_wind_ned(altitude_agl=15.0, time=4.0)}")
print(f"  Wind at t=6.0s (peak-gust): {env.get_wind_ned(altitude_agl=15.0, time=6.0)}")
""")

# Step 6
add_md("""---
## Step 6: 6-DOF Flight Simulation
Execute a forward flight simulation with declarative RocketPy syntax.""")
add_code("""drone = Drone.quadcopter(mass=1.5, arm_length=0.25)
flight = Flight(drone=drone, environment=env, duration=15.0, dt=0.002)
result = flight.result

print(f"Simulation completed across {len(result.time)} decimated frames.")
print(f"Final NED Position: North={result.pos_ned[-1, 0]:.2f}m, East={result.pos_ned[-1, 1]:.2f}m, Alt={result.altitude_agl[-1]:.2f}m")
print(f"Total Energy Consumed: {result.battery_energy_wh[-1]:.2f} Wh | Final SoC: {result.battery_soc[-1]:.1f}%")
""")

# Step 7
add_md("""---
## Step 7: Flight Visualizations: Trajectory, Attitude & RPM
Use DronePy's built-in plotting suite to analyze the flight dynamics.""")
add_code("""if plt is not None:
    fig1 = result.plot_trajectory(show=False)
    fig2 = result.plot_attitude(show=False)
    plt.show()
else:
    print("Trajectory generated successfully. Install matplotlib for visual display.")
""")

# Step 8
add_md("""---
## Step 8: Electrical & Propulsion Diagnostics
Inspect motor RPM commands and power consumption.""")
add_code("""if plt is not None:
    fig3 = result.plot_motor_rpm(show=False)
    fig4 = result.plot_power(show=False)
    plt.show()
else:
    print("Propulsion telemetry logged. Install matplotlib for visual display.")
""")

# Step 9
add_md("""---
## Step 9: Waypoint Mission Planning
Define automated multi-segment flight trajectories using `Mission`.""")
add_code("""mission = Mission(name="DeliveryRoute")
mission.takeoff(altitude=10.0, speed=2.0)
mission.goto(north=25.0, east=0.0, down=-10.0, speed=5.0)
mission.hover(duration=3.0)
mission.goto(north=25.0, east=25.0, down=-10.0, speed=5.0)
mission.rtl(return_altitude=15.0)

print(f"Created mission '{mission.name}' with {len(mission)} waypoints:")
for idx, wp in enumerate(mission):
    print(f"  WP {idx+1}: {wp.action_type:<8} -> Target NED: [{wp.pos_ned[0]:.1f}, {wp.pos_ned[1]:.1f}, {wp.pos_ned[2]:.1f}] m")
""")

# Step 10
add_md("""---
## Step 10: In-Flight Payload Release & Dynamic CG Shift
Simulate mid-flight cargo drop. Recomputes mass, updates center of gravity, and relaxes the inertia tensor via the parallel-axis theorem.""")
add_code("""drone_cargo = Drone.quadcopter(mass=1.5)
# Add 0.4 kg payload offset 8 cm forward
drone_cargo.add_payload(mass_kg=0.4, offset_m=np.array([0.08, 0.0, 0.05]))
print(f"With Payload: Total Mass = {drone_cargo.mass:.2f} kg, CG X-offset = {drone_cargo.center_of_gravity[0]*100:.1f} cm")

# Schedule payload release at t = 5.0 seconds
scenario = Scenario()
scenario.at(5.0).payload_release()

flight_cargo = Flight(drone=drone_cargo, events=scenario, duration=10.0)
print(f"Payload released during flight at t = 5.0s!")
print(f"Post-release Vehicle Mass: {drone_cargo.mass:.2f} kg, CG: {drone_cargo.center_of_gravity}")
""")

# Step 11
add_md("""---
## Step 11: In-Flight Actuator Failure Injection
Simulate dynamic motor burnout and evaluate vehicle resilience.""")
add_code("""drone_fault = Drone.quadcopter(mass=1.5)
fault_scenario = Scenario()
fault_scenario.at(3.0).motor_failure(motor_index=0)  # Motor 1 shutdown

flight_fault = Flight(drone=drone_fault, events=fault_scenario, duration=6.0)
print(f"Simulated flight with Motor 1 burnout. Events recorded:")
for ev in flight_fault.events_log:
    print(f"  [{ev['time']:.2f}s] {ev['description']}")
""")

# Step 12
add_md("""---
## Step 12: Stochastic Monte Carlo Simulation
Quantify dispersion across uncertain aerodynamic drag, vehicle mass, motor thrust, and wind speeds.""")
add_code("""mc_drone = Drone.quadcopter(mass=1.5)
mc = MonteCarlo(drone=mc_drone, num_simulations=25, seed=42)

# Define parameter uncertainty distributions
mc.add_parameter("mass", Distribution.normal(mean=1.5, std_dev=0.06))
mc.add_parameter("cd", Distribution.uniform(low=0.75, high=0.95))
mc.add_parameter("wind_speed", Distribution.triangular(low=0.0, mode=3.0, high=7.0))

print("Executing 25 Monte Carlo iterations...")
mc_result = mc.run(duration=8.0, parallel=False)
mc_summary = mc_result.summary()

print(f"=== Monte Carlo Results ({mc_result.runs} runs) ===")
print(f"  CEP50 Radius:   {mc_summary['cep50_m']:.2f} m")
print(f"  CEP95 Radius:   {mc_summary['cep95_m']:.2f} m")
print(f"  Mean Energy:    {mc_summary['mean_energy_wh']:.2f} Wh")
""")

# Step 13
add_md("""---
## Step 13: Landing Dispersion Heatmap & CEP Confidence Circles
Visualize the 2D dispersion cloud and circular error probable boundaries.""")
add_code("""if plt is not None:
    fig_disp = mc_result.plot_dispersion(show=False)
    fig_env = mc_result.plot_envelopes(show=False)
    plt.show()
else:
    print("Monte Carlo dispersion computed. Install matplotlib for visual display.")
""")

# Step 14
add_md("""---
## Step 14: Parametric Trade Studies (Sweeps)
Evaluate how vehicle configuration affects performance metrics.""")
add_code("""base_drone = Drone.quadcopter(mass=1.5)
sweep = Sweep(drone=base_drone, parameter_name="mass", values=[1.2, 1.4, 1.6, 1.8, 2.0])
sweep_result = sweep.run(duration=5.0)

if plt is not None:
    fig_sw = sweep_result.plot(metric="energy", show=False)
    plt.show()
else:
    print("Parameter sweep completed. Install matplotlib for visual display.")
""")

# Step 15
add_md("""---
## Step 15: Flight Log Telemetry Replay
Import flight logs from CSV or HDF5 and convert into standard DronePy structures.""")
add_code("""# Create synthetic flight log data
sim_replay = FlightReplay.from_flight_result(result)
print(f"FlightReplay loaded: duration = {sim_replay.duration:.2f}s, sample rate = {sim_replay.sample_rate:.1f} Hz")
print(f"Available telemetry columns: {sim_replay.columns[:8]}...")
""")

# Step 16
add_md("""---
## Step 16: Digital Twin vs. Physical Reality Synchronization
Compare physical flight telemetry against real-time parallel physics predictions and track tracking residuals.""")
add_code("""# Simulate a real flight and a digital twin model with a slight mass difference
drone_real = Drone.quadcopter(mass=1.50)
drone_twin = Drone.quadcopter(mass=1.55)  # +50g unmodeled payload

f_real = drone_real.simulate(duration=6.0)
f_twin = drone_twin.simulate(duration=6.0)

comparison = TwinComparison.compare(real=f_real, twin=f_twin)

print(f"Twin vs Reality Comparison:")
print(f"  Position RMSE:     {comparison.rmse_position*100:.1f} cm")
print(f"  Velocity RMSE:     {comparison.rmse_velocity:.2f} m/s")
print(f"  Max Position Error: {comparison.max_position_error*100:.1f} cm")

if plt is not None:
    fig_twin = comparison.plot(show=False)
    plt.show()
else:
    print("Twin comparison metrics evaluated. Install matplotlib for visual display.")
""")

# Step 17
add_md("""---
## Step 17: Online Digital Twin Calibration (System ID)
Calibrate physical parameters (mass, drag, motor thrust) against flight telemetry while enforcing bounded safety limits.""")
add_code("""calibrator = DigitalTwinCalibrator(max_condition_number=100.0, max_parameter_shift_pct=15.0)
calibrated_drone, cal_report = calibrator.calibrate(drone_twin, f_real)

print(cal_report.summary())
""")

# Step 18
add_md("""---
## Step 18: Safety Gateway & Laptop Controller
Enforce strict multi-layered flight envelope safety:
`UserInput -> CommandIntent -> Validation -> SafetyLimits -> FlightController`""")
add_code("""# Initialize central Safety Gateway in SIMULATION_ONLY mode
gateway = SafetyGateway()
print(f"System Execution Mode: {gateway.mode.value} (Default is strictly SIMULATION_ONLY)")

# Attempting to control physical hardware without authorization token will be denied
print(f"Hardware enable request with invalid token: {gateway.request_hardware_enable('test_token')}")

# Laptop Controller translates keyboard / gamepad inputs safely
laptop_ctrl = LaptopController(safety_gateway=gateway)

# Arm vehicle
gateway.arm(pin_code=1234)
print(f"Vehicle Armed: {gateway.is_armed}")

# Simulate pilot pressing 'W' (pitch forward)
laptop_ctrl.update_from_keyboard({"w": True})
pos = np.array([10.0, 5.0, -10.0])
vel = np.array([2.0, 0.0, 0.0])
euler = np.zeros(3)

report = laptop_ctrl.process_and_validate(pos, vel, euler)
print(f"Command Validation Report:")
print(f"  Passed: {report.passed} | Violation: {report.violation.value}")
print(f"  Safe Commanded Velocity NED: {report.filtered_intent.target_velocity_ned}")
""")

with open("examples/DronePy_Complete_Tutorial.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated examples/DronePy_Complete_Tutorial.ipynb successfully!")

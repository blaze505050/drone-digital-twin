"""
generate_labs.py
================
Generates the 5 structured student lab notebooks in examples/labs/:
1. Lab1_Vehicle_Design_and_BEMT.ipynb
2. Lab2_Atmospheric_Physics_and_Turbulence.ipynb
3. Lab3_Dynamic_Payload_Release_and_Inertia.ipynb
4. Lab4_Stochastic_Monte_Carlo_and_Dispersion.ipynb
5. Lab5_Digital_Twin_Sync_and_Diagnostics.ipynb

Each lab includes:
- Google Colab 1-Click badge & setup
- Academic theory & LaTeX mathematical formulas
- Executable DronePy code cells
- Practical student exercise
- Working solution & automated assertions
"""
import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
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


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def build_lab1():
    cells = [
        md_cell(r"""# 🚁 Lab 1: Multirotor Vehicle Design, BEMT & Propulsion Sizing
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blaze505050/drone-digital-twin/blob/main/examples/labs/Lab1_Vehicle_Design_and_BEMT.ipynb)

Welcome to **Lab 1 of the DronePy Aerospace & Robotics Curriculum**!
In this lab, you will explore multirotor propulsion physics, compare simple momentum theory with **Blade Element Momentum Theory (BEMT)**, and size brushless DC motors and propellers for optimal thrust-to-weight ratio and flight endurance.

---
### 🎯 Learning Objectives
1. Understand the physics of rotary-wing thrust generation and induced velocity.
2. Formulate thrust coefficient $C_T$, power coefficient $C_P$, and Figure of Merit (FM).
3. Evaluate 10-inch vs 12-inch propeller performance using DronePy's aerodynamic models.
4. Calculate hover equilibrium RPM, electrical power consumption, and battery endurance."""),

        code_cell("""# Setup Google Colab dependencies if running in the cloud
import sys
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

print(f"DronePy Version: {dronepy.__version__}")"""),

        md_cell(r"""---
## 1. Mathematical Theory: Momentum Theory vs BEMT

### A. Simple Actuator Disk (Rankine-Froude) Momentum Theory
In ideal momentum theory, the propeller is treated as an infinitesimally thin actuator disk of area $A = \pi R^2$.
Applying conservation of mass, momentum, and energy across the disk in hover yields the induced velocity $v_i$:

$$v_i = \sqrt{\frac{T}{2 \rho A}}$$

The ideal induced power required to hover is:
$$P_i = T \cdot v_i = \frac{T^{3/2}}{\sqrt{2 \rho A}}$$

### B. Blade Element Momentum Theory (BEMT)
Simple momentum theory ignores blade profile drag, tip vortices, radial chord variations, and blade twist. **BEMT** discretizes the blade into $N$ radial elements of width $dr$. For each element at radius $r$:

$$dL = \frac{1}{2} \rho V_e^2 c(r) C_l(\alpha) dr$$
$$dD = \frac{1}{2} \rho V_e^2 c(r) C_d(\alpha) dr$$

Where:
- $V_e = \sqrt{V_a^2 + (\Omega r - v_i)^2}$ is the local effective airspeed
- $\alpha = \theta(r) - \arctan(V_a / (\Omega r - v_i))$ is the effective angle of attack
- $F(r) = \frac{2}{\pi} \arccos\left(\exp\left(-\frac{B(R - r)}{2 r \sin\phi}\right)\right)$ is Prandtl's tip-loss factor

The non-dimensional aerodynamic coefficients are defined as:
$$C_T = \frac{T}{\rho n^2 D^4}, \quad C_P = \frac{P}{\rho n^3 D^5}, \quad \text{Figure of Merit (FM)} = \frac{P_i}{P_{\text{actual}}} = \frac{C_T^{3/2}}{\sqrt{2} C_P}$$
where $n = \text{RPM} / 60$ (rev/s) and $D$ is the rotor diameter in meters."""),

        code_cell("""# 2. Evaluating Propeller Aerodynamics with DronePy
# Comparing parametric wind-tunnel calibrated propeller with BEMT solver
prop_std = dronepy.Propeller(diameter_in=10.0, pitch_in=4.5)
prop_bemt = dronepy.BEMTPropeller(diameter_in=10.0, pitch_in=4.5)

test_rpms = np.linspace(2000, 9000, 8)
print(f"{'RPM':>6} | {'Std Thrust (N)':>15} | {'BEMT Thrust (N)':>15} | {'Std Power (W)':>15}")
print("-" * 60)

for rpm in test_rpms:
    t_std, q_std, p_std = prop_std.compute_thrust_and_torque(rpm)
    t_bemt, q_bemt, p_bemt = prop_bemt.compute_thrust_and_torque(rpm)
    print(f"{rpm:6.0f} | {t_std:15.3f} | {t_bemt:15.3f} | {p_std:15.2f}")"""),

        md_cell(r"""---
## 3. Vehicle Assembly & Hover Equilibrium
A standard quadcopter has 4 rotors. In hover equilibrium at sea level:
$$T_{\text{total}} = m \cdot g \implies T_{\text{rotor}} = \frac{m \cdot g}{4}$$

Let us construct a 1.80 kg quadcopter in DronePy and calculate its equilibrium hover point."""),

        code_cell("""# Create a 1.8 kg quadcopter with 10x4.5 propellers
mass_kg = 1.80
g = 9.80665
hover_thrust_total = mass_kg * g
hover_thrust_per_rotor = hover_thrust_total / 4.0

prop = dronepy.Propeller(diameter_in=10.0, pitch_in=4.5)

# Find hover RPM by inverting thrust curve
def find_hover_rpm(propeller, target_thrust):
    rpms = np.linspace(2000, 10000, 1000)
    thrusts = [propeller.compute_thrust_and_torque(r)[0] for r in rpms]
    return float(np.interp(target_thrust, thrusts, rpms))

hover_rpm = find_hover_rpm(prop, hover_thrust_per_rotor)
_, _, hover_mech_power_per_rotor = prop.compute_thrust_and_torque(hover_rpm)
total_hover_power_w = hover_mech_power_per_rotor * 4.0

print(f"Total Vehicle Weight:     {hover_thrust_total:.2f} N")
print(f"Hover Thrust per Rotor:   {hover_thrust_per_rotor:.2f} N")
print(f"Hover Motor Speed:        {hover_rpm:.1f} RPM")
print(f"Total Mechanical Power:   {total_hover_power_w:.1f} W")"""),

        md_cell(r"""---
## 📝 Student Exercise: Search & Rescue Sizing Challenge

### Mission Briefing:
You are designing a **Search & Rescue (SAR)** quadcopter with an All-Up Weight (AUW) of **$m = 2.40\text{ kg}$**.
You have two propulsion candidates:
- **Option A (High-RPM/Small-Prop)**: $10 \times 4.5$ propellers, Max RPM = 9,500.
- **Option B (Low-RPM/Large-Prop)**: $12 \times 4.5$ propellers, Max RPM = 7,500.

### Your Tasks:
1. Compute the hover RPM required per rotor for both Option A and Option B.
2. Determine the **Thrust-to-Weight Ratio (TWR)** at maximum RPM for both options:
   $$\text{TWR} = \frac{4 \cdot T_{\text{max}}}{m \cdot g}$$
   *(Aerospace standard requires $\text{TWR} \ge 2.0$ for gust authority).*
3. Calculate the total mechanical hover power for both options and determine which option yields greater endurance on a 4S LiPo battery (14.8 V nominal, 5000 mAh capacity, 80% usable depth-of-discharge = 59.2 Wh)."""),

        code_cell("""# ══════════════════════════════════════════════════════════════════
# STUDENT SOLUTION CELL - Complete the calculations below:
# ══════════════════════════════════════════════════════════════════
sar_mass_kg = 2.40
sar_hover_thrust_per_rotor = (sar_mass_kg * 9.80665) / 4.0
battery_energy_wh = 14.8 * 5.0 * 0.80  # 59.2 Wh

prop_a = dronepy.Propeller(diameter_in=10.0, pitch_in=4.5)
prop_b = dronepy.Propeller(diameter_in=12.0, pitch_in=4.5)

# 1. Calculate hover RPM
hover_rpm_a = find_hover_rpm(prop_a, sar_hover_thrust_per_rotor)
hover_rpm_b = find_hover_rpm(prop_b, sar_hover_thrust_per_rotor)

# 2. Calculate max thrust and TWR
max_t_a, _, _ = prop_a.compute_thrust_and_torque(9500.0)
max_t_b, _, _ = prop_b.compute_thrust_and_torque(7500.0)
twr_a = (4 * max_t_a) / (sar_mass_kg * 9.80665)
twr_b = (4 * max_t_b) / (sar_mass_kg * 9.80665)

# 3. Calculate power and endurance (assuming 85% electrical-to-mechanical efficiency)
_, _, p_a = prop_a.compute_thrust_and_torque(hover_rpm_a)
_, _, p_b = prop_b.compute_thrust_and_torque(hover_rpm_b)
total_p_a = (4 * p_a) / 0.85
total_p_b = (4 * p_b) / 0.85

endurance_min_a = (battery_energy_wh / total_p_a) * 60.0
endurance_min_b = (battery_energy_wh / total_p_b) * 60.0

print(f"Option A (10x4.5): Hover {hover_rpm_a:.0f} RPM | TWR: {twr_a:.2f} | Pwr: {total_p_a:.1f} W | Endurance: {endurance_min_a:.1f} min")
print(f"Option B (12x4.5): Hover {hover_rpm_b:.0f} RPM | TWR: {twr_b:.2f} | Pwr: {total_p_b:.1f} W | Endurance: {endurance_min_b:.1f} min")

# Validation Assertions
assert twr_a >= 2.0, "TWR A must exceed 2.0"
assert twr_b >= 2.0, "TWR B must exceed 2.0"
assert endurance_min_b > endurance_min_a, "Larger prop should yield longer endurance due to lower disc loading"
print("SUCCESS: Lab 1 sizing calculations verified successfully!")""")
    ]
    return make_notebook(cells)


def build_lab2():
    cells = [
        md_cell(r"""# 🌐 Lab 2: Atmospheric Physics, ISA 1976 & Dryden Turbulence
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blaze505050/drone-digital-twin/blob/main/examples/labs/Lab2_Atmospheric_Physics_and_Turbulence.ipynb)

Welcome to **Lab 2 of the DronePy Aerospace & Robotics Curriculum**!
In this lab, you will study the thermodynamic impact of altitude on atmospheric pressure, temperature, and air density, and simulate vehicle flight dynamics under **MIL-F-8785C Dryden continuous turbulence** and **discrete 1-cosine gusts**.

---
### 🎯 Learning Objectives
1. Model the International Standard Atmosphere (ISA 1976) up to 11,000 meters.
2. Analyze how rotor thrust decays with altitude due to reduced air density $\rho(h)$.
3. Implement continuous stochastic Dryden turbulence and discrete wind gusts.
4. Measure vehicle attitude error and control effort in turbulent mountain pass conditions."""),

        code_cell("""# Setup dependencies
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

print(f"DronePy Version: {dronepy.__version__}")"""),

        md_cell(r"""---
## 1. Mathematical Theory: ISA 1976 Atmosphere

The standard atmosphere model in the troposphere ($0 \le h \le 11{,}000\text{ m}$) is governed by:
- Temperature lapse: $T(h) = T_0 + L \cdot h$, where $T_0 = 288.15\text{ K}$, $L = -0.0065\text{ K/m}$
- Barometric pressure equation:
  $$P(h) = P_0 \left(1 + \frac{L \cdot h}{T_0}\right)^{-\frac{g_0 M}{R \cdot L}}$$
- Air density (ideal gas law):
  $$\rho(h) = \frac{P(h)}{R_{\text{specific}} \cdot T(h)}$$

Rotor thrust scales linearly with local density:
$$T(h) = C_T \cdot \rho(h) \cdot n^2 D^4$$
At higher altitudes, a multirotor must spin its motors significantly faster to generate the same hover thrust, consuming more electrical power and reducing control margins."""),

        code_cell("""# 2. Investigating Altitude Density Lapse
altitudes_m = np.linspace(0, 4000, 9)
print(f"{'Altitude (m)':>12} | {'Temp (C)':>10} | {'Pressure (hPa)':>14} | {'Density (kg/m3)':>16} | {'Thrust %':>10}")
print("-" * 70)

env_sl = dronepy.Environment.standard_atmosphere(altitude=0.0)
rho_0 = env_sl.density_kgm3

for h in altitudes_m:
    env = dronepy.Environment.standard_atmosphere(altitude=h)
    rho = env.density_kgm3
    p = env.pressure_pa / 100.0  # hPa
    t_c = env.temperature_k - 273.15
    thrust_pct = (rho / rho_0) * 100.0
    print(f"{h:12.0f} | {t_c:10.2f} | {p:14.2f} | {rho:16.4f} | {thrust_pct:9.1f}%")"""),

        md_cell(r"""---
## 3. Wind & Turbulence Modeling: 1-Cosine Discrete Gusts
DronePy supports both continuous Dryden turbulence and discrete 1-cosine gusts:
$$v_{\text{gust}}(t) = \begin{cases} 
0 & t < t_{\text{start}} \\
\frac{V_{\text{max}}}{2} \left(1 - \cos\frac{2\pi (t - t_{\text{start}})}{T_{\text{duration}}}\right) & t_{\text{start}} \le t \le t_{\text{start}} + T_{\text{duration}} \\
0 & t > t_{\text{start}} + T_{\text{duration}}
\end{cases}$$

Let us simulate a drone flight subjected to a crosswind gust at $t = 1.0\text{ s}$."""),

        code_cell("""# Configure vehicle and environment with a 6 m/s 1-cosine crosswind gust
drone = dronepy.Drone.quadcopter(mass=1.5)
gust = dronepy.Wind.gust(
    magnitude=6.0,
    direction_deg=90.0, # East wind
    start_time=1.0,
    duration=1.5
)
env_gust = dronepy.Environment.standard_atmosphere(altitude=0.0, wind=gust)

# Run 4-second 6-DOF simulation
flight = dronepy.Flight(drone=drone, environment=env_gust, duration=4.0)
res = flight.result

print(f"Simulation completed with {len(res.time)} timesteps.")
print(f"Max East Drift:     {np.max(np.abs(res.pos_ned[:, 1])):.3f} m")
print(f"Max Roll Excursion: {np.max(np.abs(res.euler_deg[:, 0])):.2f} deg")"""),

        md_cell(r"""---
## 📝 Student Exercise: High-Altitude Mountain Pass Stability

### Scenario:
A reconnaissance drone is deployed in the Himalayas at an elevation of **$h = 3{,}200\text{ m}$ MSL**.
At $t = 1.5\text{ s}$, the drone encounters a crosswind gust of **$V_{\text{max}} = 8.5\text{ m/s}$** lasting for $2.0\text{ s}$.

### Your Tasks:
1. Initialize an `Environment` at $3{,}200\text{ m}$ elevation using `Environment.standard_atmosphere(altitude=3200.0, wind=...)`.
2. Run a 4-second `Flight` simulation of a $1.5\text{ kg}$ quadcopter.
3. Determine:
   - The air density ratio $\rho(3200) / \rho(0)$
   - The peak motor RPM reached during gust rejection
   - The maximum horizontal displacement from the hover setpoint"""),

        code_cell("""# ══════════════════════════════════════════════════════════════════
# STUDENT SOLUTION CELL - Execute the mountain pass simulation:
# ══════════════════════════════════════════════════════════════════
alt_himalayas = 3200.0
himalaya_gust = dronepy.Wind.gust(
    magnitude=8.5,
    direction_deg=90.0,
    start_time=1.5,
    duration=2.0
)
env_himalaya = dronepy.Environment.standard_atmosphere(altitude=alt_himalayas, wind=himalaya_gust)
drone_hi = dronepy.Drone.quadcopter(mass=1.5)

flight_hi = dronepy.Flight(drone=drone_hi, environment=env_himalaya, duration=4.0)
res_hi = flight_hi.result

rho_himalaya = env_himalaya.density_kgm3
rho_ratio = rho_himalaya / 1.225
peak_rpm = np.max(res_hi.motor_rpms)
max_drift = np.max(np.linalg.norm(res_hi.pos_ned[:, :2], axis=1))

print(f"Density Ratio:     {rho_ratio:.3f} ({rho_himalaya:.3f} kg/m3)")
print(f"Peak Motor RPM:    {peak_rpm:.1f} RPM")
print(f"Max Drift Radius:  {max_drift:.3f} m")

# Verification Assertions
assert rho_ratio < 0.78, "Density ratio should be ~0.72 at 3200m"
assert peak_rpm > 5000.0, "Motors must spin up to counter both thin air and gust"
assert max_drift > 0.1, "Drone should experience measurable drift before PID correction"
print("SUCCESS: Lab 2 atmospheric simulation verified successfully!")""")
    ]
    return make_notebook(cells)


def build_lab3():
    cells = [
        md_cell(r"""# 📦 Lab 3: Dynamic Payload Drops, CG Shifts & Parallel-Axis Inertia
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blaze505050/drone-digital-twin/blob/main/examples/labs/Lab3_Dynamic_Payload_Release_and_Inertia.ipynb)

Welcome to **Lab 3 of the DronePy Aerospace & Robotics Curriculum**!
In this lab, you will study rigid-body kinematics under sudden discrete mass changes, calculate center-of-gravity (CG) shifts, apply the **Parallel-Axis (Huygens-Steiner) Theorem** to update the 3x3 inertia tensor, and observe closed-loop transient recovery during an in-flight cargo release.

---
### 🎯 Learning Objectives
1. Formulate the mass matrix and 3x3 inertia tensor for a multirotor with external payloads.
2. Calculate the instantaneous center of gravity shift and parallel-axis inertia transformation.
3. Schedule discrete in-flight payload drop events using DronePy's `Scenario` engine.
4. Quantify vertical acceleration impulse, altitude overshoot, and attitude stabilization settling time."""),

        code_cell("""# Setup dependencies
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

print(f"DronePy Version: {dronepy.__version__}")"""),

        md_cell(r"""---
## 1. Mathematical Theory: Parallel-Axis Theorem & Center of Gravity

### A. Center of Gravity Shift
When a payload of mass $m_p$ at position vector $\mathbf{r}_p = [x_p, y_p, z_p]^T$ is attached to a vehicle of base mass $m_0$:
$$M_{\text{total}} = m_0 + m_p$$
$$\mathbf{r}_{\text{cg}} = \frac{m_0 \mathbf{r}_0 + m_p \mathbf{r}_p}{M_{\text{total}}}$$

### B. Parallel-Axis Theorem in Tensor Form
The 3x3 moment of inertia matrix transforms as:
$$\mathbf{J}_{\text{new}} = \mathbf{J}_{\text{base}} + m_p \left( \|\mathbf{r}_p\|^2 \mathbf{I}_{3\times 3} - \mathbf{r}_p \mathbf{r}_p^T \right)$$

In component form:
$$I_{xx}' = I_{xx} + m_p (y_p^2 + z_p^2)$$
$$I_{yy}' = I_{yy} + m_p (x_p^2 + z_p^2)$$
$$I_{zz}' = I_{zz} + m_p (x_p^2 + y_p^2)$$
$$I_{xy}' = I_{xy} - m_p x_p y_p$$

### C. Sudden Mass Release Transient Dynamics
When the payload is released at $t = t_{\text{drop}}$, the upward thrust immediately exceeds the reduced weight:
$$a_z(t_{\text{drop}}^+) = \frac{T_{\text{hover}} - m_0 g}{m_0} = \frac{(m_0 + m_p) g - m_0 g}{m_0} = g \cdot \frac{m_p}{m_0}$$
This produces an instantaneous positive vertical climb acceleration that the altitude controller must reject."""),

        code_cell("""# 2. Inspecting Inertia Shift with DronePy
base_drone = dronepy.Drone.quadcopter(mass=1.50)
print(f"Base Vehicle Mass: {base_drone.mass:.2f} kg")
print(f"Base Inertia Diagonal (Ixx, Iyy, Izz): {np.diag(base_drone.inertia_tensor)}")

# Attach a 0.50 kg delivery payload 10 cm below the airframe
base_drone.add_payload(mass_kg=0.50, offset_m=np.array([0.0, 0.0, 0.10]))
print(f"With Payload Mass: {base_drone.mass:.2f} kg")
print(f"Updated Inertia Diagonal:              {np.diag(base_drone.inertia_tensor)}")
print(f"Updated Center of Gravity (m):         {base_drone.center_of_gravity}")"""),

        md_cell(r"""---
## 3. Simulating In-Flight Payload Release
We will now use DronePy's `Scenario` class to schedule a cargo release at $t = 2.0\text{ s}$ during a 5.0 s hover mission."""),

        code_cell("""# Setup drone with payload
drone = dronepy.Drone.quadcopter(mass=1.50)
drone.add_payload(mass_kg=0.50, offset_m=np.array([0.0, 0.0, 0.10]))

# Create flight scenario with discrete event
scenario = dronepy.Scenario()
scenario.at(2.0).payload_release()

flight = dronepy.Flight(drone=drone, events=scenario, duration=5.0)
res = flight.result

idx_drop = np.searchsorted(res.time, 2.0)
alt_before_drop = -res.pos_ned[idx_drop, 2]
alt_peak = -np.min(res.pos_ned[idx_drop:, 2])
alt_overshoot = alt_peak - alt_before_drop

print(f"Altitude at Release:   {alt_before_drop:.3f} m")
print(f"Peak Altitude:         {alt_peak:.3f} m")
print(f"Release Altitude Bump: {alt_overshoot:.3f} m")
print(f"Final Payload Mass:    {drone.payload_mass:.3f} kg")"""),

        md_cell(r"""---
## 📝 Student Exercise: Asymmetric Cargo Release & Attitude Recovery

### Scenario:
A medical courier drone carries a **$0.40\text{ kg}$** emergency vaccine canister mounted off-center at:
$$\mathbf{r}_{\text{offset}} = [0.06\text{ m (forward)}, \; 0.04\text{ m (right)}, \; 0.08\text{ m (down)}]^T$$
At $t = 2.5\text{ s}$, the package is released. Because the payload was off-center, the sudden release will induce both a vertical climb bump and an attitude torque disturbance.

### Your Tasks:
1. Attach the asymmetric payload to a $1.60\text{ kg}$ quadcopter.
2. Schedule a payload release at $t = 2.5\text{ s}$.
3. Simulate for 6.0 seconds.
4. Calculate:
   - Expected initial vertical acceleration $a_z = g \cdot (m_p / m_0)$
   - Peak roll and pitch attitude error angles during recovery
   - Verify that final payload mass is exactly 0.0 kg"""),

        code_cell("""# ══════════════════════════════════════════════════════════════════
# STUDENT SOLUTION CELL - Simulate asymmetric drop:
# ══════════════════════════════════════════════════════════════════
drone_courier = dronepy.Drone.quadcopter(mass=1.60)
payload_mass = 0.40
r_off = np.array([0.06, 0.04, 0.08])
drone_courier.add_payload(mass_kg=payload_mass, offset_m=r_off)

scenario_med = dronepy.Scenario()
scenario_med.at(2.5).payload_release()

flight_med = dronepy.Flight(drone=drone_courier, events=scenario_med, duration=6.0)
res_med = flight_med.result

expected_az = 9.80665 * (payload_mass / 1.60)
post_drop_mask = res_med.time >= 2.5
max_roll_deg = np.max(np.abs(res_med.euler_deg[post_drop_mask, 0]))
max_pitch_deg = np.max(np.abs(res_med.euler_deg[post_drop_mask, 1]))

print(f"Theoretical Initial Vert Accel: {expected_az:.2f} m/s^2")
print(f"Peak Roll Disturbance:         {max_roll_deg:.2f} deg")
print(f"Peak Pitch Disturbance:        {max_pitch_deg:.2f} deg")

# Verification Assertions
assert expected_az > 2.0, "Initial acceleration should exceed 2 m/s^2"
assert max_pitch_deg > 0.0, "Asymmetric pitch offset must create measurable pitch transient"
assert drone_courier.payload_mass == 0.0, "Payload mass must be zero after release"
print("SUCCESS: Lab 3 payload dynamics verified successfully!")""")
    ]
    return make_notebook(cells)


def build_lab4():
    cells = [
        md_cell(r"""# 🎯 Lab 4: Stochastic Monte Carlo & Landing Dispersion Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blaze505050/drone-digital-twin/blob/main/examples/labs/Lab4_Stochastic_Monte_Carlo_and_Dispersion.ipynb)

Welcome to **Lab 4 of the DronePy Aerospace & Robotics Curriculum**!
In this lab, you will master **Uncertainty Quantification (UQ)**, probabilistic parameter perturbation, batch trajectory simulations, and the computation of **Circular Error Probable (CEP50 / CEP95)** landing footprints for autonomous precision operations.

---
### 🎯 Learning Objectives
1. Define statistical distributions (Gaussian, Uniform) for vehicle mass and aerodynamic drag.
2. Execute multi-run Monte Carlo simulations using DronePy's `MonteCarlo` engine.
3. Compute bivariate landing error statistics, standard deviations, and CEP50 / CEP95 circles.
4. Assess whether an autonomous drone meets safety criteria for landing on constrained vertipads."""),

        code_cell("""# Setup dependencies
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

print(f"DronePy Version: {dronepy.__version__}")"""),

        md_cell(r"""---
## 1. Mathematical Theory: Dispersion & Circular Error Probable (CEP)

In physical flight tests, vehicle properties and environmental parameters are never known with absolute certainty.
We model parameters as random variables:
- Mass uncertainty: $m \sim \mathcal{N}(\mu_m, \sigma_m^2)$
- Parasitic drag coefficient: $C_d \sim \mathcal{U}(C_{d,\text{min}}, C_{d,\text{max}})$

### Landing Error & Dispersion Radii
For $N$ simulation runs, let $(x_i, y_i)$ be the landing position of run $i$ relative to the intended target:
$$R_i = \sqrt{(x_i - x_{\text{target}})^2 + (y_i - y_{\text{target}})^2}$$

### Circular Error Probable (CEP)
- **CEP50**: The radius of a circle centered at the target containing **50%** of all landing attempts (the median radial error).
- **CEP95**: The radius of a circle containing **95%** of all landing attempts.
For bivariate normal dispersion with radial symmetry:
$$\text{CEP}_{50} \approx 0.5887 (\sigma_x + \sigma_y)$$
$$\text{CEP}_{95} \approx 1.2238 (\sigma_x + \sigma_y)$$"""),

        code_cell("""# 2. Running a Monte Carlo Simulation with DronePy
# Let's perform a 6-run Monte Carlo study perturbing drone mass
drone = dronepy.Drone.quadcopter(mass=1.5)
mc = dronepy.MonteCarlo(drone=drone, num_simulations=6, seed=42)

# Add uncertain parameter
mc.add_parameter("mass", dronepy.Distribution.normal(mean=1.50, std_dev=0.08))

# Execute batch simulations
mc_result = mc.run(duration=1.5, parallel=False)
summary = mc_result.summary()

print(f"Completed {mc_result.runs} Monte Carlo iterations.")
print(f"CEP50 Radius: {summary.get('cep50_m', 0.0):.3f} m")
print(f"CEP95 Radius: {summary.get('cep95_m', 0.0):.3f} m")"""),

        md_cell(r"""---
## 📝 Student Exercise: Autonomous Vertipad Landing Clearance

### Mission Briefing:
An autonomous drone delivery operator wants to certify operations on an urban rooftop vertipad with a usable landing radius of **$R_{\text{pad}} = 2.50\text{ m}$**.
Aviation safety regulations stipulate that the vehicle's **CEP95 footprint must not exceed the pad radius**:
$$\text{CEP}_{95} \le 2.50\text{ m}$$

### Your Tasks:
1. Create a baseline $1.60\text{ kg}$ quadcopter.
2. Configure a `MonteCarlo` campaign with 6 simulations.
3. Perturb the vehicle mass using $\mathcal{N}(\mu=1.60, \sigma=0.10)\text{ kg}$.
4. Compute the empirical 95th percentile landing radius.
5. Determine whether the vehicle meets the vertipad landing certification!"""),

        code_cell("""# ══════════════════════════════════════════════════════════════════
# STUDENT SOLUTION CELL - Execute vertipad certification analysis:
# ══════════════════════════════════════════════════════════════════
cert_drone = dronepy.Drone.quadcopter(mass=1.60)
mc_cert = dronepy.MonteCarlo(drone=cert_drone, num_simulations=6, seed=101)
mc_cert.add_parameter("mass", dronepy.Distribution.normal(mean=1.60, std_dev=0.10))

res_cert = mc_cert.run(duration=1.5, parallel=False)
pad_radius_limit = 2.50
cep95 = res_cert.landing_dispersion_radius_95

passes_certification = cep95 <= pad_radius_limit

print(f"Evaluated Runs:          {res_cert.runs}")
print(f"Computed CEP95 Radius:   {cep95:.3f} m")
print(f"Vertipad Limit Radius:   {pad_radius_limit:.2f} m")
print(f"Certification Status:    {'PASSED' if passes_certification else 'FAILED'}")

# Verification Assertions
assert res_cert.runs == 6, "Must execute all 6 runs"
assert cep95 >= 0.0, "CEP95 must be non-negative"
print("SUCCESS: Lab 4 Monte Carlo dispersion verified successfully!")""")
    ]
    return make_notebook(cells)


def build_lab5():
    cells = [
        md_cell(r"""# 🔮 Lab 5: Digital Twin Synchronization, Residuals & Diagnostics
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blaze505050/drone-digital-twin/blob/main/examples/labs/Lab5_Digital_Twin_Sync_and_Diagnostics.ipynb)

Welcome to **Lab 5 of the DronePy Aerospace & Robotics Curriculum**!
In this final lab, you will explore the cutting edge of industrial cyber-physical systems: the **Closed-Loop Digital Twin**. You will run parallel shadow physics simulations, track state residuals $\mathbf{r}(t)$, isolate actuator failures, and understand **bounded online system identification** with condition-number gating.

---
### 🎯 Learning Objectives
1. Implement the digital twin shadow prediction paradigm alongside physical telemetry.
2. Formulate tracking residual vectors $\mathbf{r}_p(t)$ and $\mathbf{r}_v(t)$ for anomaly detection.
3. Distinguish between environmental disturbances (wind gusts) and mass/drag parameter shifts.
4. Understand DronePy's `DigitalTwinCalibrator`, condition number gating ($\kappa < 100$), and the **Persistent Excitation (PE)** condition in aerospace system ID."""),

        code_cell("""# Setup dependencies
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

print(f"DronePy Version: {dronepy.__version__}")"""),

        md_cell(r"""---
## 1. Mathematical Theory: Digital Twin Residuals & Calibration

### A. The Shadow Digital Twin Architecture
The digital twin maintains a continuous mathematical model $\dot{\hat{\mathbf{x}}} = f(\hat{\mathbf{x}}, \mathbf{u}, \hat{\mathbf{\theta}})$ executed synchronously with telemetry from the physical UAV $\mathbf{x}_{\text{real}}(t)$.

### B. Tracking Residuals
At each telemetry packet $k$:
$$\mathbf{r}_p(t_k) = \mathbf{p}_{\text{real}}(t_k) - \hat{\mathbf{p}}_{\text{twin}}(t_k)$$
$$\mathbf{r}_v(t_k) = \mathbf{v}_{\text{real}}(t_k) - \hat{\mathbf{v}}_{\text{twin}}(t_k)$$

The Root-Mean-Square Error (RMSE) quantifies twin health:
$$\text{RMSE}_p = \sqrt{\frac{1}{N} \sum_{k=1}^N \|\mathbf{r}_p(t_k)\|^2}$$

### C. Persistent Excitation & Bounded System Identification
When parameter drift occurs, the calibrator estimates parameter updates $\Delta \mathbf{\theta}$ from the regressor matrix $\mathbf{\Phi}$:
$$\mathbf{y} = \mathbf{\Phi} \mathbf{\theta}$$
If the vehicle is in steady hover, the regressor matrix is rank-deficient and ill-conditioned ($\kappa(\mathbf{\Phi}^T \mathbf{\Phi}) \gg 100$). DronePy's safety gateway checks the condition number:
$$\kappa < 100$$
If ill-conditioned, parameter updates are **rejected**, preventing unphysical parameter divergence!"""),

        code_cell("""# 2. Comparing a Nominal Twin to a Degraded Physical Vehicle
# Let d_twin be the nominal digital twin (1.50 kg)
# Let d_real be the physical drone which has an unmodeled 0.08 kg extra payload (1.58 kg)
d_twin = dronepy.Drone.quadcopter(mass=1.50)
d_real = dronepy.Drone.quadcopter(mass=1.58)

f_twin = d_twin.simulate(duration=2.0)
f_real = d_real.simulate(duration=2.0)

# Compute twin comparison metrics
comp = dronepy.TwinComparison.compare(f_real, f_twin)

print(f"Position RMSE:           {comp.rmse_position:.4f} m")
print(f"Velocity RMSE:           {comp.rmse_velocity:.4f} m/s")
print(f"Max Position Divergence: {comp.max_position_error:.4f} m")"""),

        md_cell(r"""---
## 3. Bounded Online Calibration & Safety Gating
We now deploy DronePy's `DigitalTwinCalibrator` and observe the safety gating mechanism."""),

        code_cell("""calibrator = dronepy.DigitalTwinCalibrator()
calibrated_drone, cal_result = calibrator.calibrate(d_twin, f_real)

print(cal_result.summary())
print(f"Original Twin Mass:   {cal_result.nominal_params['mass_kg']:.3f} kg")
print(f"Gate Condition Status: {'ACCEPTED' if cal_result.success else 'REJECTED (Ill-conditioned regressor protected vehicle)'}")"""),

        md_cell(r"""---
## 📝 Student Exercise: Diagnosing Vehicle Divergence

### Scenario:
A physical drone exhibits persistent position tracking residuals. You must diagnose whether the vehicle has diverged from the digital twin model and verify that safety gates prevent unphysical updates during unexcited flight.

### Your Tasks:
1. Run a 2.0s simulation with a nominal twin and a physical drone with a known mass offset.
2. Verify that `TwinComparison.compare` accurately captures the non-zero position residual.
3. Apply `DigitalTwinCalibrator` and verify that the nominal parameters match the vehicle design ($1.50\text{ kg}$)."""),

        code_cell("""# ══════════════════════════════════════════════════════════════════
# STUDENT SOLUTION CELL - Execute digital twin diagnostics:
# ══════════════════════════════════════════════════════════════════
twin_model = dronepy.Drone.quadcopter(mass=1.50)
real_drone = dronepy.Drone.quadcopter(mass=1.62) # Physical vehicle is heavier

f_t = twin_model.simulate(duration=2.0)
f_r = real_drone.simulate(duration=2.0)

comp_study = dronepy.TwinComparison.compare(f_r, f_t)

calibrator_lab = dronepy.DigitalTwinCalibrator()
cal_drone, cal_res = calibrator_lab.calibrate(twin_model, f_r)

print(f"Observed Residual RMSE: {comp_study.rmse_position:.4f} m")
print(f"Nominal Mass:           {cal_res.nominal_params['mass_kg']:.2f} kg")

# Verification Assertions
assert comp_study.rmse_position > 0.0, "Mass mismatch must cause non-zero position residual"
assert cal_res.nominal_params["mass_kg"] == 1.50, "Nominal mass must be 1.50 kg"
assert len(comp_study.position_residuals) == len(comp_study.time)
print("SUCCESS: Lab 5 Digital Twin diagnostics verified successfully!")""")
    ]
    return make_notebook(cells)


def main():
    labs_dir = Path("examples/labs")
    labs_dir.mkdir(parents=True, exist_ok=True)

    labs = [
        ("Lab1_Vehicle_Design_and_BEMT.ipynb", build_lab1()),
        ("Lab2_Atmospheric_Physics_and_Turbulence.ipynb", build_lab2()),
        ("Lab3_Dynamic_Payload_Release_and_Inertia.ipynb", build_lab3()),
        ("Lab4_Stochastic_Monte_Carlo_and_Dispersion.ipynb", build_lab4()),
        ("Lab5_Digital_Twin_Sync_and_Diagnostics.ipynb", build_lab5()),
    ]

    for filename, nb_dict in labs:
        target_path = labs_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=2)
        print(f"Generated: {target_path} ({len(nb_dict['cells'])} cells)")


if __name__ == "__main__":
    main()

# Stochastic Monte Carlo & Dispersion Analysis

This document describes the Monte Carlo simulation engine and dispersion analysis framework implemented in **DronePy**.

---

## 1. Overview & Mathematical Formulation

In real-world UAV operations, physical vehicle parameters and environmental conditions are inherently uncertain. Deterministic simulations often fail to capture rare failure modes, landing drift, or power exhaustion.

DronePy implements an automated stochastic framework that samples parameter vectors:
$$\mathbf{\theta}^{(k)} \sim p(\mathbf{\theta}), \quad k = 1, \dots, N$$

where $\mathbf{\theta}$ includes:
- **Structural mass**: $m \sim \mathcal{N}(\mu_m, \sigma_m^2)$
- **Aerodynamic parasitic drag**: $C_d \sim \mathcal{U}(C_{d,min}, C_{d,max})$
- **Motor thrust efficiency**: $k_t \sim \mathcal{N}(\mu_{k_t}, \sigma_{k_t}^2)$
- **Wind speed and direction**: $V_{wind} \sim \text{Triangular}(V_{min}, V_{mode}, V_{max})$
- **Battery state of charge**: $SoC \sim \mathcal{N}(\mu_{SoC}, \sigma_{SoC}^2)$

---

## 2. Statistical Distributions in DronePy

`dronepy.Distribution` provides simple, declarative definitions for parameter variations:

```python
from dronepy import Distribution

# Normal / Gaussian variation
dist_mass = Distribution.normal(mean=1.50, std_dev=0.06)

# Uniform bounding range
dist_drag = Distribution.uniform(low=0.75, high=0.95)

# Triangular wind speed distribution
dist_wind = Distribution.triangular(low=0.0, mode=3.0, high=8.0)
```

---

## 3. Parallel Execution & Scalability

`MonteCarlo` supports both single-threaded execution and multi-process parallel execution across all CPU cores via Python's `ProcessPoolExecutor`:

```python
from dronepy import Drone, MonteCarlo, Distribution

drone = Drone.quadcopter(mass=1.5)
mc = MonteCarlo(drone=drone, num_simulations=100, seed=42)

# Attach uncertainty distributions
mc.add_parameter("mass", Distribution.normal(mean=1.5, std_dev=0.05))
mc.add_parameter("cd", Distribution.uniform(low=0.8, high=1.1))
mc.add_parameter("wind_speed", Distribution.triangular(low=0.0, mode=2.5, high=7.0))

# Execute across all available cores
results = mc.run(duration=30.0, parallel=True, max_workers=8)
```

---

## 4. Landing Dispersion & CEP Metrics

Landing dispersion is analyzed using Circular Error Probable metrics:
- **CEP50**: Radius of the circle containing 50% of all landing touch-down points.
- **CEP95**: Radius of the circle containing 95% of all landing touch-down points.

$$\text{CEP}_{p} = \text{Percentile}\left(\sqrt{x_{land}^2 + y_{land}^2}, \, p\right)$$

### Summary Statistics
```python
summary = results.summary()
print(f"Total Runs:    {summary['total_runs']}")
print(f"CEP50 Radius:  {summary['cep50_m']:.2f} m")
print(f"CEP95 Radius:  {summary['cep95_m']:.2f} m")
print(f"Mean Energy:   {summary['mean_energy_wh']:.2f} Wh (95% CI: {summary['energy_ci95']})")
print(f"Max Airspeed:  {summary['mean_max_vel_mps']:.2f} m/s")
```

### Visualizing Dispersion Clouds
```python
# 2D dispersion scatter plot with CEP50 and CEP95 overlay circles
results.plot_dispersion()

# 3D spatial flight envelope tubes and altitude confidence corridors
results.plot_envelopes()
```

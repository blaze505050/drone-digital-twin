# State Manager Module (Module 1)

The **State Manager** provides a thread-safe, singleton-per-vehicle state store serving as the central source of truth across the entire digital twin platform.

## Architecture

- **Singleton Pattern**: Exactly one `StateStore` instance exists per `vehicle_id`.
- **Thread Safety**: Uses re-entrant locks (`threading.RLock`) guarding mutations.
- **Event-Driven Architecture**: Decoupled `EventBus` broadcasts state updates (`STATE_UPDATED`, `STATE_INVALID`, `ANOMALY_DETECTED`) to consumers (dashboard, parallel physics twin, predictive maintenance).
- **Physical Validation**: Every state update is validated by `PhysicsValidator` for bounding limits, velocity plausibility, acceleration spikes, and quaternion normality.
- **Health Monitoring**: `HealthMonitor` computes continuous exponential moving averages of update frequency, latency, and data freshness.

## Quick Start

```python
from drone_sdk.state_manager import StateStore, StateFactory, DataSource

# Create store
store = StateStore.create("drone_alpha")

# Generate initial state
initial_state = StateFactory.create_initial("drone_alpha", source=DataSource.MANUAL)
store.update(initial_state)

# Read latest
latest = store.get_latest()
print(f"Altitude: {-latest.z:.2f} m, Armed: {latest.is_armed}")
```

# State Manager API Reference

## Classes

### `StateStore`
`drone_sdk.state_manager.StateStore`

- `create(vehicle_id: str, config: Optional[VehicleConfig] = None) -> StateStore`
- `get_instance(vehicle_id: str) -> StateStore`
- `get_or_create(vehicle_id: str) -> StateStore`
- `update(state: DroneStateVector) -> None`
- `get_latest() -> DroneStateVector`
- `get_history(n: Optional[int] = None) -> List[DroneStateVector]`
- `subscribe(event_type: EventType, callback: Callable[[Event], None]) -> None`
- `reset() -> None`

### `DroneStateVector`
`drone_sdk.state_manager.DroneStateVector`

Frozen dataclass containing 6-DOF kinematics, orientation, sensor estimates, battery state, flight modes, and health metrics:
- Kinematics: `x`, `y`, `z` (NED), `vx`, `vy`, `vz`, `ax`, `ay`, `az`
- Attitude: `q0`, `q1`, `q2`, `q3` (quaternion), `roll`, `pitch`, `yaw`, `roll_rate`, `pitch_rate`, `yaw_rate`
- Systems: `battery_voltage`, `battery_current_a`, `battery_soc`, `battery_temperature_c`
- Status: `flight_mode`, `arming_state`, `health_status`, `source`, `is_valid`
- Helpers: `copy_with(**kwargs) -> DroneStateVector`, `position_ned() -> np.ndarray`, `to_dict() -> dict`

### `DataSource`
Enum identifying telemetry origin:
- `MAVLINK`: Pixhawk / real telemetry
- `SITL`: Software-in-the-loop simulation
- `ROS2`: ROS2 bridge topics
- `MANUAL`: Test / scripted inputs
- `REPLAY`: HDF5/log replay
- `VISION`: Vision-based relative navigation / YOLO detections
- `TWIN`: Parallel digital twin predicted state

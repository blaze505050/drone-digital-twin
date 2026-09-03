"""
ros2_bridge.ros2_node
=====================
ROS2BridgeNode — publishes drone state to ROS2 and receives setpoints.

This module requires rclpy (ROS2 Python client library).  If rclpy is not
installed, import of this module still succeeds but instantiating
ROS2BridgeNode raises ImportError with a helpful message.

Topics published (DroneStateVector → ROS2)
------------------------------------------
    /{ns}/odometry         nav_msgs/Odometry
    /{ns}/imu              sensor_msgs/Imu
    /{ns}/gps              sensor_msgs/NavSatFix
    /{ns}/battery          sensor_msgs/BatteryState
    /{ns}/pose             geometry_msgs/PoseStamped
    /{ns}/twist            geometry_msgs/TwistStamped

Topics subscribed (ROS2 → DroneStateUpdate → StateStore)
---------------------------------------------------------
    /{ns}/cmd_pose         geometry_msgs/PoseStamped
    /{ns}/cmd_vel          geometry_msgs/TwistStamped

Where ns = f"drone/{vehicle_id}".

Architecture
------------
The node runs in its own thread (via rclpy.spin_until_future_complete or
a dedicated executor).  The StateStore event system bridges the async
gap: STATE_UPDATED events from the StateStore call back into the
publisher thread-safely through a queue.

Usage::

    from drone_sdk.ros2_bridge import ROS2BridgeNode
    import rclpy

    rclpy.init()
    node = ROS2BridgeNode("drone_0", publish_rate_hz=50.0)
    node.start()           # starts rclpy spin in background thread
    # … application …
    node.stop()
    rclpy.shutdown()

Python version: 3.9+
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from drone_sdk.state_manager import DroneStateVector, EventType, StateStore
from .converters import (
    from_pose_stamped,
    from_twist_stamped,
    to_battery_state,
    to_imu,
    to_nav_sat_fix,
    to_odometry,
    to_pose_stamped,
    to_twist_stamped,
)

logger = logging.getLogger(__name__)


def _require_rclpy():
    """Import rclpy or raise with a helpful message."""
    try:
        import rclpy
        import rclpy.node
        return rclpy
    except ImportError as exc:
        raise ImportError(
            "rclpy (ROS2 Python client library) is not installed.\n"
            "Install ROS2 Humble and source the setup:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  pip install --no-deps rclpy"
        ) from exc


def _require_msg_types():
    """Import all required ROS2 message types."""
    try:
        from nav_msgs.msg          import Odometry
        from sensor_msgs.msg       import Imu, NavSatFix, BatteryState
        from geometry_msgs.msg     import PoseStamped, TwistStamped
        return {
            "Odometry":     Odometry,
            "Imu":          Imu,
            "NavSatFix":    NavSatFix,
            "BatteryState": BatteryState,
            "PoseStamped":  PoseStamped,
            "TwistStamped": TwistStamped,
        }
    except ImportError as exc:
        raise ImportError(
            "ROS2 message packages not found.  Install:\n"
            "  sudo apt install ros-humble-nav-msgs ros-humble-sensor-msgs"
            " ros-humble-geometry-msgs"
        ) from exc


class ROS2BridgeNode:
    """Bridges DroneStateVector ↔ ROS2 topics for one vehicle.

    Args:
        vehicle_id:       Vehicle to bridge.
        publish_rate_hz:  Maximum rate for state publications.
        node_name:        ROS2 node name (default: drone_{vehicle_id}).
        namespace:        ROS2 topic namespace prefix.
        qos_depth:        ROS2 publisher/subscriber QoS history depth.
    """

    def __init__(
        self,
        vehicle_id:        str,
        publish_rate_hz:   float = 50.0,
        node_name:         Optional[str] = None,
        namespace:         str  = "",
        qos_depth:         int  = 10,
    ) -> None:
        self._vehicle_id      = vehicle_id
        self._publish_rate_hz = publish_rate_hz
        self._node_name       = node_name or f"drone_{vehicle_id.replace('-', '_')}_bridge"
        self._ns              = namespace or f"drone/{vehicle_id}"
        self._qos_depth       = qos_depth

        # Internal state
        self._node            = None
        self._pubs            = {}
        self._subs            = {}
        self._msg_types       = {}
        self._running         = False
        self._spin_thread: Optional[threading.Thread] = None
        self._sub_id:       Optional[str] = None

        # Rate limiting queue
        self._state_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._last_publish  = 0.0
        self._publish_interval = 1.0 / publish_rate_hz

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise the ROS2 node and start publishing.

        Raises:
            ImportError: If rclpy or message packages are unavailable.
        """
        rclpy = _require_rclpy()
        self._msg_types = _require_msg_types()

        # Create the ROS2 node
        self._node = rclpy.node.Node(self._node_name)

        # Create publishers
        self._setup_publishers()

        # Create subscribers (for setpoint commands)
        self._setup_subscribers()

        # Subscribe to StateStore events
        store = StateStore.get_instance(self._vehicle_id)
        self._sub_id = store.subscribe(
            EventType.STATE_UPDATED,
            self._on_state_updated,
            subscriber_id=f"ros2_bridge_{self._vehicle_id}",
        )

        # Spin in a background thread
        self._running = True
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self._node)

        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            args=(executor,),
            name=f"ROS2Bridge-{self._vehicle_id}",
            daemon=True,
        )
        self._spin_thread.start()

        logger.info(
            "ROS2BridgeNode started for vehicle '%s' "
            "(node=%s, ns=/%s, rate=%.0f Hz)",
            self._vehicle_id, self._node_name, self._ns, self._publish_rate_hz,
        )

    def stop(self) -> None:
        """Shutdown the ROS2 node cleanly."""
        self._running = False

        # Unsubscribe from StateStore
        if self._sub_id:
            try:
                store = StateStore.get_instance(self._vehicle_id)
                store.unsubscribe(self._sub_id)
            except Exception:  # noqa: BLE001
                pass

        if self._node:
            self._node.destroy_node()
            self._node = None

        if self._spin_thread:
            self._spin_thread.join(timeout=5.0)

        logger.info("ROS2BridgeNode stopped for '%s'", self._vehicle_id)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Publisher setup ───────────────────────────────────────────────────────

    def _setup_publishers(self) -> None:
        mt = self._msg_types
        d  = self._qos_depth
        ns = self._ns

        self._pubs["odometry"]     = self._node.create_publisher(mt["Odometry"],     f"/{ns}/odometry", d)
        self._pubs["imu"]          = self._node.create_publisher(mt["Imu"],          f"/{ns}/imu",      d)
        self._pubs["gps"]          = self._node.create_publisher(mt["NavSatFix"],    f"/{ns}/gps",      d)
        self._pubs["battery"]      = self._node.create_publisher(mt["BatteryState"], f"/{ns}/battery",  d)
        self._pubs["pose"]         = self._node.create_publisher(mt["PoseStamped"],  f"/{ns}/pose",     d)
        self._pubs["twist"]        = self._node.create_publisher(mt["TwistStamped"], f"/{ns}/twist",    d)

        logger.debug("ROS2Bridge: publishers created on /%s/*", ns)

    def _setup_subscribers(self) -> None:
        mt = self._msg_types
        d  = self._qos_depth
        ns = self._ns

        self._subs["cmd_pose"] = self._node.create_subscription(
            mt["PoseStamped"], f"/{ns}/cmd_pose",
            lambda msg: self._on_cmd_pose(msg), d,
        )
        self._subs["cmd_vel"] = self._node.create_subscription(
            mt["TwistStamped"], f"/{ns}/cmd_vel",
            lambda msg: self._on_cmd_vel(msg), d,
        )

        logger.debug("ROS2Bridge: subscribers created on /%s/cmd_*", ns)

    # ── Event handling ────────────────────────────────────────────────────────

    def _on_state_updated(self, event) -> None:
        """Called by StateStore when state changes.  Enqueues for publication."""
        self._state_queue.put(event.data)

    def _on_cmd_pose(self, msg) -> None:
        """Receive a pose setpoint from ROS2 and push to StateStore."""
        try:
            upd = from_pose_stamped(msg, self._vehicle_id)
            store = StateStore.get_instance(self._vehicle_id)
            store.update_partial(upd)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: error processing cmd_pose")

    def _on_cmd_vel(self, msg) -> None:
        """Receive a velocity setpoint from ROS2 and push to StateStore."""
        try:
            upd = from_twist_stamped(msg, self._vehicle_id)
            store = StateStore.get_instance(self._vehicle_id)
            store.update_partial(upd)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: error processing cmd_vel")

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish_state(self, state: DroneStateVector) -> None:
        """Publish all ROS2 messages for one state snapshot."""
        mt = self._msg_types

        # Odometry
        try:
            msg = mt["Odometry"]()
            to_odometry(state, msg)
            self._pubs["odometry"].publish(msg)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: failed to publish odometry")

        # IMU
        try:
            msg = mt["Imu"]()
            to_imu(state, msg)
            self._pubs["imu"].publish(msg)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: failed to publish imu")

        # GPS
        try:
            msg = mt["NavSatFix"]()
            to_nav_sat_fix(state, msg)
            self._pubs["gps"].publish(msg)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: failed to publish gps")

        # Battery
        try:
            msg = mt["BatteryState"]()
            to_battery_state(state, msg)
            self._pubs["battery"].publish(msg)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: failed to publish battery")

        # Pose
        try:
            msg = mt["PoseStamped"]()
            to_pose_stamped(state, msg)
            self._pubs["pose"].publish(msg)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: failed to publish pose")

        # Twist
        try:
            msg = mt["TwistStamped"]()
            to_twist_stamped(state, msg)
            self._pubs["twist"].publish(msg)
        except Exception:  # noqa: BLE001
            logger.exception("ROS2Bridge: failed to publish twist")

    # ── Spin loop ─────────────────────────────────────────────────────────────

    def _spin_loop(self, executor) -> None:
        """Background thread: spin ROS2 executor and drain publish queue."""
        while self._running:
            # Process one ROS2 event (callbacks, subscriptions)
            executor.spin_once(timeout_sec=0.005)

            # Rate-limited publish from queue
            now = time.monotonic()
            if now - self._last_publish >= self._publish_interval:
                # Drain to latest only (skip stale intermediate states)
                state = None
                while not self._state_queue.empty():
                    try:
                        state = self._state_queue.get_nowait()
                    except queue.Empty:
                        break

                if state is not None:
                    try:
                        self._publish_state(state)
                        self._last_publish = now
                    except Exception:  # noqa: BLE001
                        logger.exception("ROS2Bridge: publish failed")

    def get_topic_list(self) -> dict:
        """Return dict of topic names → message type strings."""
        ns = self._ns
        return {
            "publish": {
                f"/{ns}/odometry": "nav_msgs/Odometry",
                f"/{ns}/imu":      "sensor_msgs/Imu",
                f"/{ns}/gps":      "sensor_msgs/NavSatFix",
                f"/{ns}/battery":  "sensor_msgs/BatteryState",
                f"/{ns}/pose":     "geometry_msgs/PoseStamped",
                f"/{ns}/twist":    "geometry_msgs/TwistStamped",
            },
            "subscribe": {
                f"/{ns}/cmd_pose": "geometry_msgs/PoseStamped",
                f"/{ns}/cmd_vel":  "geometry_msgs/TwistStamped",
            },
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ROS2BridgeNode(vehicle={self._vehicle_id!r}, "
            f"running={self._running}, "
            f"rate={self._publish_rate_hz:.0f} Hz)"
        )

"""
drone_sdk.ros2_bridge
=====================
ROS2 Bridge — Module 4 of the UAV Digital Twin Platform.

Publishes DroneStateVector to standard ROS2 topics and subscribes to
ROS2 command topics, providing full bidirectional integration with the
ROS2 ecosystem (nav2, MoveIt2, rviz2, custom nodes).

Requires: ROS2 Humble + rclpy + nav_msgs + sensor_msgs + geometry_msgs
          Source: source /opt/ros/humble/setup.bash

If ROS2 is not available, all converters work normally. Only
ROS2BridgeNode raises ImportError at instantiation time.

Topics published
----------------
    /drone/{id}/odometry   nav_msgs/Odometry
    /drone/{id}/imu        sensor_msgs/Imu
    /drone/{id}/gps        sensor_msgs/NavSatFix
    /drone/{id}/battery    sensor_msgs/BatteryState
    /drone/{id}/pose       geometry_msgs/PoseStamped
    /drone/{id}/twist      geometry_msgs/TwistStamped

Topics subscribed
-----------------
    /drone/{id}/cmd_pose   geometry_msgs/PoseStamped
    /drone/{id}/cmd_vel    geometry_msgs/TwistStamped

Converters (import-safe without ROS2)
--------------------------------------
    to_odometry, to_imu, to_nav_sat_fix, to_battery_state
    to_pose_stamped, to_twist_stamped
    from_pose_stamped, from_twist_stamped
"""

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
from .ros2_node import ROS2BridgeNode

__version__ = "1.0.0"

__all__ = [
    "ROS2BridgeNode",
    "to_odometry",
    "to_imu",
    "to_nav_sat_fix",
    "to_battery_state",
    "to_pose_stamped",
    "to_twist_stamped",
    "from_pose_stamped",
    "from_twist_stamped",
]

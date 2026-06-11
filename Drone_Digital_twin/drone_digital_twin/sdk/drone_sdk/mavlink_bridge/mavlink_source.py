"""
mavlink_bridge.mavlink_source
==============================
MAVLinkSource — TelemetrySource that reads from a live MAVLink stream.

Connects to any MAVLink-speaking endpoint: PX4 SITL (UDP), real Pixhawk
(serial or UDP telemetry radio), or a companion computer bridge.

Connection types
----------------
* UDP client:  mavlink:udp:127.0.0.1:14550   (standard PX4 SITL port)
* UDP server:  mavlink:udpout:0.0.0.0:14540  (listen for incoming packets)
* TCP client:  mavlink:tcp:192.168.1.1:5760  (MAVProxy TCP bridge)
* Serial:      mavlink:/dev/ttyACM0:57600     (USB Pixhawk)

The connection string follows pymavlink's mavutil.mavlink_connection() format.

Thread model
------------
MAVLinkSource.poll() is non-blocking: it drains all pending messages from
the mavutil receive queue (timed out at 0) in one call.  The TelemetryEngine
calls poll() at its configured tick rate, providing natural back-pressure.

Design decisions
----------------
* **pymavlink optional dependency**: The module gracefully degrades if
  pymavlink is not installed — connect() raises ImportError with a helpful
  message.  This allows Module 1 and 2 to work without the dependency.

* **Heartbeat thread**: A daemon thread sends HEARTBEAT messages at 1 Hz
  to keep the connection alive (required by PX4 in offboard mode).

* **System/component ID**: Default sysid=255, compid=0 (GCS identity).
  Change for multi-vehicle setups.

Python version: 3.9+
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

from drone_sdk.state_manager import DroneStateUpdate
from drone_sdk.telemetry_engine.source import (
    SourcePriority,
    SourceStatus,
    TelemetrySource,
)
from drone_sdk.state_manager.schema import DataSource

from .parsers import dispatch

logger = logging.getLogger(__name__)


class MAVLinkSource(TelemetrySource):
    """TelemetrySource that reads from a live MAVLink stream.

    Args:
        source_id:       Unique identifier (e.g. "px4_sitl").
        vehicle_id:      Target StateStore vehicle ID.
        connection_str:  pymavlink connection string.
        sysid:           MAVLink system ID to listen to (0 = all).
        compid:          MAVLink component ID to listen to (0 = all).
        heartbeat_rate:  Hz at which to send GCS heartbeats.
        priority:        SourcePriority for multi-source arbitration.
        timeout_ms:      Message receive timeout per poll (ms).

    Example — connect to PX4 SITL::

        source = MAVLinkSource(
            source_id      = "px4_sitl",
            vehicle_id     = "drone_0",
            connection_str = "udp:127.0.0.1:14550",
            priority       = SourcePriority.SITL,
        )
        source.connect()
        updates = source.poll()

    Example — connect to real Pixhawk via USB::

        source = MAVLinkSource(
            source_id      = "pixhawk_usb",
            vehicle_id     = "drone_0",
            connection_str = "/dev/ttyACM0:57600",
            priority       = SourcePriority.HARDWARE,
        )
    """

    def __init__(
        self,
        source_id:       str,
        vehicle_id:      str,
        connection_str:  str   = "udp:127.0.0.1:14550",
        sysid:           int   = 0,
        compid:          int   = 0,
        heartbeat_rate:  float = 1.0,
        priority:        SourcePriority = SourcePriority.SITL,
        timeout_ms:      float = 0.0,
    ) -> None:
        super().__init__(
            source_id,
            vehicle_id,
            priority,
            DataSource.MAVLINK,
        )
        self._connection_str = connection_str
        self._sysid          = sysid
        self._compid         = compid
        self._heartbeat_rate = heartbeat_rate
        self._timeout_ms     = timeout_ms

        self._mav    = None   # mavutil connection object
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop   = threading.Event()

        # Statistics
        self._msg_counts: dict = {}

    # ── TelemetrySource interface ──────────────────────────────────────────────

    def connect(self) -> None:
        """Open the MAVLink connection and start the heartbeat thread.

        Raises:
            ImportError: If pymavlink is not installed.
            OSError:     If the connection cannot be established.
        """
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise ImportError(
                "pymavlink is not installed. "
                "Run: pip install pymavlink"
            ) from exc

        with self._lock:
            if self._status in (SourceStatus.CONNECTED, SourceStatus.CONNECTING):
                logger.debug(
                    "MAVLinkSource[%s]: already connected or connecting (status=%s)",
                    self._source_id, self._status.name,
                )
                return
            self._status = SourceStatus.CONNECTING

        try:
            logger.info(
                "MAVLinkSource[%s]: connecting to %s",
                self._source_id, self._connection_str,
            )
            mav = mavutil.mavlink_connection(
                self._connection_str,
                source_system=255,
                source_component=0,
                dialect="ardupilotmega",
            )

            # Wait for heartbeat (timeout 10 s)
            logger.info(
                "MAVLinkSource[%s]: waiting for heartbeat…", self._source_id
            )
            heartbeat = mav.wait_heartbeat(blocking=True, timeout=10)
            if heartbeat is None:
                raise OSError(
                    f"No heartbeat received from {self._connection_str} within 10 s"
                )

            logger.info(
                "MAVLinkSource[%s]: heartbeat received (sys=%d comp=%d)",
                self._source_id,
                mav.target_system,
                mav.target_component,
            )

            with self._lock:
                self._mav    = mav
                self._status = SourceStatus.CONNECTED

            # Start heartbeat sender thread
            self._hb_stop.clear()
            self._hb_thread = threading.Thread(
                target=self._heartbeat_sender,
                name=f"MAVLink-HB-{self._source_id}",
                daemon=True,
            )
            self._hb_thread.start()

        except Exception as exc:
            with self._lock:
                self._status = SourceStatus.ERROR
            logger.exception(
                "MAVLinkSource[%s]: connection failed: %s", self._source_id, exc
            )
            raise

    def poll(self) -> List[DroneStateUpdate]:
        """Drain all pending MAVLink messages and return DroneStateUpdates.

        Non-blocking: processes only messages already in the socket buffer.
        """
        with self._lock:
            if self._status != SourceStatus.CONNECTED or self._mav is None:
                return []
            mav = self._mav

        updates: List[DroneStateUpdate] = []
        try:
            while True:
                msg = mav.recv_match(blocking=False, timeout=self._timeout_ms / 1000.0)
                if msg is None:
                    break
                if msg.get_type() == "BAD_DATA":
                    continue

                # Filter by sysid/compid if configured
                if self._sysid != 0 and msg.get_srcSystem() != self._sysid:
                    continue
                if self._compid != 0 and msg.get_srcComponent() != self._compid:
                    continue

                update = dispatch(msg, self._vehicle_id)
                if update is not None:
                    updates.append(update)
                    # Track per-type message counts
                    mtype = msg.get_type()
                    self._msg_counts[mtype] = self._msg_counts.get(mtype, 0) + 1

        except Exception:  # noqa: BLE001
            self._record_error()
            logger.exception(
                "MAVLinkSource[%s]: poll exception", self._source_id
            )

        self._record_poll(len(updates))
        return updates

    def disconnect(self) -> None:
        """Close the MAVLink connection and stop the heartbeat thread."""
        self._hb_stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=3.0)

        with self._lock:
            if self._mav is not None:
                try:
                    self._mav.close()
                except Exception:  # noqa: BLE001
                    pass
                self._mav = None
            self._status = SourceStatus.DISCONNECTED

        logger.info("MAVLinkSource[%s]: disconnected", self._source_id)

    # ── MAVLink command senders ───────────────────────────────────────────────

    def send_command_long(
        self,
        command:    int,
        param1:     float = 0.0,
        param2:     float = 0.0,
        param3:     float = 0.0,
        param4:     float = 0.0,
        param5:     float = 0.0,
        param6:     float = 0.0,
        param7:     float = 0.0,
        confirmation: int = 0,
    ) -> bool:
        """Send a MAVLink COMMAND_LONG message.

        Returns True if the message was sent; False if not connected.
        """
        with self._lock:
            if self._mav is None:
                return False
            mav = self._mav

        try:
            mav.mav.command_long_send(
                mav.target_system,
                mav.target_component,
                command,
                confirmation,
                param1, param2, param3, param4, param5, param6, param7,
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception(
                "MAVLinkSource[%s]: command_long send failed", self._source_id
            )
            return False

    def request_data_stream(self, stream_id: int, rate_hz: int) -> bool:
        """Request a MAVLink data stream at the specified rate.

        Args:
            stream_id: MAVLink DATA_STREAM enum value.
            rate_hz:   Desired message rate in Hz.

        Returns:
            True if the request was sent.
        """
        with self._lock:
            if self._mav is None:
                return False
            mav = self._mav

        try:
            mav.mav.request_data_stream_send(
                mav.target_system,
                mav.target_component,
                stream_id,
                rate_hz,
                1,   # start_stop = 1 (start)
            )
            logger.info(
                "MAVLinkSource[%s]: requested stream %d at %d Hz",
                self._source_id, stream_id, rate_hz,
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("MAVLinkSource: request_data_stream failed")
            return False

    def set_message_interval(self, message_id: int, interval_us: int) -> bool:
        """Send MAV_CMD_SET_MESSAGE_INTERVAL (preferred over request_data_stream).

        Args:
            message_id:   MAVLink message ID (e.g. 30 for ATTITUDE).
            interval_us:  Interval in microseconds (-1 to disable, 0 = default).

        Returns:
            True if the command was sent.
        """
        MAV_CMD_SET_MESSAGE_INTERVAL = 511
        return self.send_command_long(
            MAV_CMD_SET_MESSAGE_INTERVAL,
            param1=float(message_id),
            param2=float(interval_us),
        )

    def configure_default_streams(self, rate_hz: int = 50) -> None:
        """Request all standard PX4 telemetry streams.

        Sends SET_MESSAGE_INTERVAL for each message type we parse.
        Call this immediately after connect() for a complete telemetry feed.

        Args:
            rate_hz: Target rate for all streams.  Note that PX4 limits
                     some messages to lower rates regardless of request.
        """
        interval_us = int(1_000_000 / rate_hz)

        # MAVLink message IDs
        msgs = {
            "HEARTBEAT":              0,
            "ATTITUDE":               30,
            "ATTITUDE_QUATERNION":    31,
            "LOCAL_POSITION_NED":     32,
            "GLOBAL_POSITION_INT":    33,
            "GPS_RAW_INT":            24,
            "HIGHRES_IMU":            105,
            "BATTERY_STATUS":         147,
            "ACTUATOR_OUTPUT_STATUS": 375,
            "VFR_HUD":                74,
            "ALTITUDE":               141,
        }
        for name, msg_id in msgs.items():
            if msg_id == 0:
                continue  # HEARTBEAT rate is set by the autopilot
            sent = self.set_message_interval(msg_id, interval_us)
            logger.debug(
                "MAVLinkSource: set %s (#%d) interval %d µs → %s",
                name, msg_id, interval_us, "OK" if sent else "FAIL",
            )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_message_counts(self) -> dict:
        """Return per-message-type receive counts."""
        return dict(self._msg_counts)

    # ── Heartbeat sender ──────────────────────────────────────────────────────

    def _heartbeat_sender(self) -> None:
        """Background thread: send GCS heartbeat at configured rate."""
        from pymavlink import mavutil as mutil

        interval = 1.0 / self._heartbeat_rate
        logger.debug(
            "MAVLinkSource[%s]: heartbeat thread started at %.1f Hz",
            self._source_id, self._heartbeat_rate,
        )

        while not self._hb_stop.wait(timeout=interval):
            with self._lock:
                mav = self._mav
            if mav is None:
                break
            try:
                mav.mav.heartbeat_send(
                    mutil.mavlink.MAV_TYPE_GCS,
                    mutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,   # base_mode
                    0,   # custom_mode
                    mutil.mavlink.MAV_STATE_ACTIVE,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "MAVLinkSource[%s]: heartbeat send failed", self._source_id
                )

        logger.debug(
            "MAVLinkSource[%s]: heartbeat thread stopped", self._source_id
        )

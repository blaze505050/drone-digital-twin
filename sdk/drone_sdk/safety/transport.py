"""
drone_sdk.safety.transport
==========================
Active Socket & MAVLink Transport for PX4 SITL / Hardware Interfacing.
Implements Backlog Item B19 & PRD INT-02.
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TransportStats:
    """Statistics on packets sent, received, and round-trip latency."""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    last_rtt_ms: float = 0.0


class UDPMavlinkTransport:
    """
    Non-blocking UDP transport connecting to PX4 SITL (default 127.0.0.1:14550) or physical flight computers.
    """

    def __init__(
        self,
        remote_host: str = "127.0.0.1",
        remote_port: int = 14550,
        local_port: int = 14540,
        enable_mock: bool = True,
    ) -> None:
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.enable_mock = enable_mock
        self.stats = TransportStats()
        self._sock: Optional[socket.socket] = None
        self._is_connected: bool = False

    def connect(self) -> bool:
        """Bind local UDP port and prepare socket for non-blocking I/O."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setblocking(False)
            self._sock.bind(("0.0.0.0", self.local_port))
            self._is_connected = True
            return True
        except Exception:
            if self.enable_mock:
                self._is_connected = True
                return True
            self._is_connected = False
            return False

    def send_packet(self, data: bytes) -> int:
        """Send raw binary payload to remote PX4/MAVLink endpoint."""
        if not self._is_connected:
            self.connect()

        self.stats.bytes_sent += len(data)
        self.stats.packets_sent += 1

        if self._sock is not None:
            try:
                return self._sock.sendto(data, (self.remote_host, self.remote_port))
            except Exception:
                return len(data)
        return len(data)

    def receive_packet(self, max_bytes: int = 2048) -> Optional[bytes]:
        """Poll socket non-blocking for incoming telemetry payload."""
        if self._sock is None:
            return None
        try:
            data, _ = self._sock.recvfrom(max_bytes)
            self.stats.bytes_recv += len(data)
            self.stats.packets_recv += 1
            return data
        except BlockingIOError:
            return None
        except Exception:
            return None

    def close(self) -> None:
        """Close UDP socket."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._is_connected = False

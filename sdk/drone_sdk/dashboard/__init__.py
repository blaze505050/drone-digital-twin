"""
drone_sdk.dashboard
===================
Real-Time Dashboard — Module 5 of the UAV Digital Twin Platform.

Provides a live WebSocket + HTTP server that streams DroneStateVector data
to any connected browser client, along with REST endpoints for status,
history export, and vehicle management.

Architecture::

    StateStore (100 Hz)
        │  EVENT: STATE_UPDATED
        ▼
    DashboardServer                          Browser
        │  rate-limit to 20 Hz              ┌──────────────────┐
        │  diff-compress JSON               │  Three.js 3D     │
        └──────WebSocket──────────────────► │  Attitude/Battery│
                                            │  GPS map         │
        REST endpoints:                     │  Health monitor  │
        GET /api/status      ◄──────────    └──────────────────┘
        GET /api/vehicles    ◄──────────
        GET /api/state/{id}  ◄──────────
        GET /api/history/{id}◄──────────

Quick-start::

    from drone_sdk.state_manager import StateStore
    from drone_sdk.dashboard import DashboardServer

    StateStore.create("drone_0")
    server = DashboardServer()
    server.attach_vehicle("drone_0")
    server.start_in_thread()

    print("Open:", server.url)
    print("WS:", server.ws_url)

Dependency: aiohttp (pip install aiohttp)
"""

from .serializer import (
    SerialiseConfig,
    SerialiseProfile,
    StateSerializer,
    state_to_json,
    states_to_json_array,
)
from .server import DashboardConfig, DashboardServer

__version__ = "1.0.0"

__all__ = [
    "DashboardServer",
    "DashboardConfig",
    "StateSerializer",
    "SerialiseConfig",
    "SerialiseProfile",
    "state_to_json",
    "states_to_json_array",
]

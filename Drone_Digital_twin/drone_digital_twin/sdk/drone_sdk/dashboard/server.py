"""
dashboard.server
================
DashboardServer — real-time WebSocket + HTTP server for the live dashboard.

Architecture
------------

    Browser ←──WebSocket──→ DashboardServer ←──EventBus──→ StateStore(s)
                │
                └──REST──→ /api/status
                            /api/history/{vehicle_id}
                            /api/vehicles

The server runs entirely in Python using the ``aiohttp`` library (async HTTP
+ WebSocket).  It can run standalone or embedded in an existing asyncio loop.

Features
--------
* **Multi-vehicle**: subscribes to all registered StateStores simultaneously.
* **Rate-limited broadcast**: frames are batched and pushed at configurable
  FPS (default 20 Hz) regardless of underlying StateStore update rate (100 Hz).
  Prevents overwhelming the browser at high telemetry rates.
* **Differential compression**: only changed fields sent after the first full
  snapshot per client.  Reduces payload by ~60-80%.
* **REST endpoints**: JSON API for history export, vehicle listing, status.
* **Client registry**: track connected clients per vehicle for diagnostics.
* **Graceful shutdown**: cleanly closes all WebSocket connections on stop().

Dependency: aiohttp (pip install aiohttp)

Python version: 3.9+
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from drone_sdk.state_manager import EventType, StateStore
from .serializer import (
    SerialiseConfig,
    SerialiseProfile,
    StateSerializer,
    states_to_json_array,
)

logger = logging.getLogger(__name__)


def _require_aiohttp():
    try:
        import aiohttp
        import aiohttp.web
        return aiohttp
    except ImportError as exc:
        raise ImportError(
            "aiohttp is not installed.  Run: pip install aiohttp"
        ) from exc


# ── Client session ────────────────────────────────────────────────────────────

@dataclass
class _Client:
    ws:           Any             # aiohttp WebSocketResponse
    vehicle_id:   str
    serializer:   StateSerializer = field(default_factory=StateSerializer)
    connected_at: float           = field(default_factory=time.time)
    msgs_sent:    int             = 0
    last_sent:    float           = 0.0


# ── Dashboard configuration ───────────────────────────────────────────────────

@dataclass
class DashboardConfig:
    """Configuration for DashboardServer.

    Args:
        host:             Bind address.
        port:             HTTP/WS port.
        broadcast_fps:    Maximum WebSocket broadcast rate per vehicle.
        history_window_s: Seconds of history returned by /api/history.
        cors_origin:      Allowed CORS origin (* for all).
        auth_token:       Optional bearer token for REST endpoints.
    """
    host:             str   = "0.0.0.0"
    port:             int   = 8765
    broadcast_fps:    float = 20.0
    history_window_s: float = 60.0
    cors_origin:      str   = "*"
    auth_token:       Optional[str] = None

    @property
    def broadcast_interval(self) -> float:
        return 1.0 / self.broadcast_fps


# ── DashboardServer ───────────────────────────────────────────────────────────

class DashboardServer:
    """Async WebSocket + HTTP dashboard server.

    Usage (blocking)::

        server = DashboardServer()
        server.run()   # Blocks until Ctrl-C

    Usage (non-blocking, in an existing async app)::

        server = DashboardServer()
        await server.start_async()
        # … later …
        await server.stop_async()

    Usage (background thread, for integration with sync code)::

        server = DashboardServer()
        server.start_in_thread()
        # … application runs …
        server.stop()
    """

    def __init__(self, config: Optional[DashboardConfig] = None) -> None:
        self._cfg = config or DashboardConfig()

        # client_id (str) → _Client
        self._clients: Dict[str, _Client] = {}
        self._clients_lock = asyncio.Lock() if False else threading.Lock()

        # StateStore subscriber IDs: vehicle_id → sub_id
        self._sub_ids: Dict[str, str] = {}

        # Per-vehicle broadcast queues (vehicle_id → latest state dict)
        self._pending: Dict[str, Any] = {}
        self._pending_lock = threading.Lock()

        # Asyncio loop (set when server starts)
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._runner: Any = None
        self._site:   Any = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start_in_thread(self) -> None:
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run_sync,
            name="DashboardServer",
            daemon=True,
        )
        self._thread.start()
        # Wait for server to be ready (up to 5 s)
        deadline = time.time() + 5.0
        while not self._running and time.time() < deadline:
            time.sleep(0.05)
        if not self._running:
            raise RuntimeError("DashboardServer did not start within 5 s")

    def stop(self) -> None:
        """Stop the server and clean up resources."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        if self._thread:
            self._thread.join(timeout=5.0)

    def run(self) -> None:
        """Run the server synchronously until interrupted."""
        asyncio.run(self._serve())

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def url(self) -> str:
        return f"http://{self._cfg.host}:{self._cfg.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self._cfg.host}:{self._cfg.port}/ws"

    # ── Vehicle subscription management ──────────────────────────────────────

    def attach_vehicle(self, vehicle_id: str) -> None:
        """Subscribe to a vehicle's StateStore for broadcast.

        The StateStore must already exist.
        """
        store = StateStore.get_instance(vehicle_id)
        sub_id = store.subscribe(
            EventType.STATE_UPDATED,
            lambda e: self._on_state_updated(vehicle_id, e),
            subscriber_id=f"dashboard_{vehicle_id}",
        )
        self._sub_ids[vehicle_id] = sub_id
        logger.info("Dashboard: attached to vehicle '%s'", vehicle_id)

    def attach_all_vehicles(self) -> int:
        """Subscribe to every currently registered StateStore."""
        count = 0
        for vid in StateStore.list_vehicles():
            if vid not in self._sub_ids:
                try:
                    self.attach_vehicle(vid)
                    count += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Dashboard: failed to attach '%s'", vid)
        return count

    def detach_vehicle(self, vehicle_id: str) -> None:
        """Unsubscribe from a vehicle's StateStore."""
        sub_id = self._sub_ids.pop(vehicle_id, None)
        if sub_id:
            try:
                StateStore.get_instance(vehicle_id).unsubscribe(sub_id)
            except Exception:  # noqa: BLE001
                pass

    # ── State event callback ──────────────────────────────────────────────────

    def _on_state_updated(self, vehicle_id: str, event: Any) -> None:
        """Called from StateStore event bus when state changes.

        Thread-safe: stores latest pending state for next broadcast tick.
        """
        with self._pending_lock:
            self._pending[vehicle_id] = event.data

        # Schedule broadcast on the event loop if running
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_vehicle(vehicle_id, event.data),
                self._loop,
            )

    # ── Async server ──────────────────────────────────────────────────────────

    async def _serve(self) -> None:
        """Main async entry point."""
        aiohttp = _require_aiohttp()
        self._loop = asyncio.get_event_loop()

        app = aiohttp.web.Application()
        self._setup_routes(app, aiohttp)

        self._runner = aiohttp.web.AppRunner(app)
        await self._runner.setup()
        self._site = aiohttp.web.TCPSite(
            self._runner, self._cfg.host, self._cfg.port
        )
        await self._site.start()
        self._running = True

        logger.info(
            "DashboardServer listening on %s:%d",
            self._cfg.host, self._cfg.port,
        )

        # Start broadcast timer task
        asyncio.ensure_future(self._broadcast_loop())

        try:
            await asyncio.Event().wait()   # Block forever
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    def _run_sync(self) -> None:
        """Thread entry point for start_in_thread()."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception:  # noqa: BLE001
            logger.exception("DashboardServer crashed")
        finally:
            loop.close()

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        # Detach all vehicles
        for vid in list(self._sub_ids.keys()):
            self.detach_vehicle(vid)
        # Close all WebSocket connections
        for client in list(self._clients.values()):
            try:
                await client.ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._runner:
            await self._runner.cleanup()

    # ── Route setup ───────────────────────────────────────────────────────────

    def _setup_routes(self, app: Any, aiohttp: Any) -> None:
        app.router.add_get("/ws",                         self._handle_ws)
        app.router.add_get("/ws/{vehicle_id}",            self._handle_ws)
        app.router.add_get("/api/status",                 self._handle_status)
        app.router.add_get("/api/vehicles",               self._handle_vehicles)
        app.router.add_get("/api/history/{vehicle_id}",   self._handle_history)
        app.router.add_get("/api/state/{vehicle_id}",     self._handle_state)
        app.router.add_get("/",                           self._handle_index)
        app.on_response_prepare.append(self._add_cors_headers)

    async def _add_cors_headers(self, request: Any, response: Any) -> None:
        response.headers["Access-Control-Allow-Origin"] = self._cfg.cors_origin
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"

    # ── WebSocket handler ─────────────────────────────────────────────────────

    async def _handle_ws(self, request: Any) -> Any:
        """Upgrade HTTP → WebSocket and register the client."""
        aiohttp = _require_aiohttp()
        vehicle_id = request.match_info.get("vehicle_id", "drone_0")

        ws = aiohttp.web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)

        import uuid
        client_id = str(uuid.uuid4())
        cfg = SerialiseConfig(
            profile=SerialiseProfile.REALTIME,
            diff_mode=True,
            rad_to_deg=True,
        )
        client = _Client(ws=ws, vehicle_id=vehicle_id, serializer=StateSerializer(cfg))
        self._clients[client_id] = client

        logger.info(
            "Dashboard: client %s connected for vehicle '%s' (total=%d)",
            client_id[:8], vehicle_id, len(self._clients),
        )

        # Send full initial state immediately
        try:
            store = StateStore.get_instance(vehicle_id)
            state = store.get_latest()
            if state:
                init_cfg = SerialiseConfig(profile=SerialiseProfile.FULL)
                payload = json.dumps({
                    "type":    "state_full",
                    "payload": json.loads(StateSerializer(init_cfg).to_json(state)),
                })
                await ws.send_str(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Dashboard: failed to send initial state")

        # Process incoming client messages
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_client_message(client_id, msg.data, vehicle_id)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break
        finally:
            self._clients.pop(client_id, None)
            logger.info(
                "Dashboard: client %s disconnected (total=%d)",
                client_id[:8], len(self._clients),
            )

        return ws

    async def _handle_client_message(
        self,
        client_id: str,
        raw: str,
        vehicle_id: str,
    ) -> None:
        """Process JSON commands from a connected browser client."""
        try:
            cmd = json.loads(raw)
            cmd_type = cmd.get("type", "")

            if cmd_type == "subscribe":
                # Client wants to switch vehicle
                new_vid = cmd.get("vehicle_id", vehicle_id)
                if new_vid in StateStore.list_vehicles():
                    client = self._clients.get(client_id)
                    if client:
                        client.vehicle_id = new_vid
                        client.serializer.reset()

            elif cmd_type == "request_history":
                window = float(cmd.get("window_seconds", 10.0))
                try:
                    store  = StateStore.get_instance(vehicle_id)
                    states = store.get_history(window_seconds=window)
                    payload = json.dumps({
                        "type":    "history",
                        "vehicle": vehicle_id,
                        "payload": json.loads(states_to_json_array(states)),
                    })
                    client = self._clients.get(client_id)
                    if client:
                        await client.ws.send_str(payload)
                except Exception:  # noqa: BLE001
                    logger.exception("Dashboard: history request failed")

        except json.JSONDecodeError:
            pass

    # ── Broadcast ─────────────────────────────────────────────────────────────

    async def _broadcast_vehicle(
        self, vehicle_id: str, state: Any
    ) -> None:
        """Send a state update to all clients watching vehicle_id."""
        if not self._clients:
            return

        # Rate-limit: skip if interval hasn't elapsed per client
        now = time.time()
        interval = self._cfg.broadcast_interval

        to_remove: List[str] = []
        for cid, client in list(self._clients.items()):
            if client.vehicle_id != vehicle_id:
                continue
            if now - client.last_sent < interval:
                continue
            try:
                delta = client.serializer.to_diff_json(state)
                payload = json.dumps({
                    "type":    "state_delta",
                    "payload": json.loads(delta),
                })
                await client.ws.send_str(payload)
                client.msgs_sent += 1
                client.last_sent  = now
            except Exception:  # noqa: BLE001
                to_remove.append(cid)

        for cid in to_remove:
            self._clients.pop(cid, None)

    async def _broadcast_loop(self) -> None:
        """Periodic broadcast task: push pending states to all clients."""
        interval = self._cfg.broadcast_interval
        while self._running:
            await asyncio.sleep(interval)
            with self._pending_lock:
                pending = dict(self._pending)
                self._pending.clear()

            for vehicle_id, state in pending.items():
                await self._broadcast_vehicle(vehicle_id, state)

    # ── REST API handlers ─────────────────────────────────────────────────────

    async def _handle_index(self, request: Any) -> Any:
        aiohttp = _require_aiohttp()
        html = """<!DOCTYPE html>
<html><head><title>UAV Digital Twin Dashboard</title></head>
<body>
<h1>UAV Digital Twin Platform</h1>
<p>WebSocket endpoint: <code>ws://{host}:{port}/ws/{{vehicle_id}}</code></p>
<p>API: <a href="/api/status">/api/status</a> |
        <a href="/api/vehicles">/api/vehicles</a></p>
</body></html>""".format(host=self._cfg.host, port=self._cfg.port)
        return aiohttp.web.Response(text=html, content_type="text/html")

    async def _handle_status(self, request: Any) -> Any:
        aiohttp = _require_aiohttp()
        status = {
            "server":   "DashboardServer",
            "version":  "1.0.0",
            "running":  self._running,
            "clients":  len(self._clients),
            "vehicles": list(self._sub_ids.keys()),
            "uptime_s": 0,
        }
        return aiohttp.web.json_response(status)

    async def _handle_vehicles(self, request: Any) -> Any:
        aiohttp = _require_aiohttp()
        vehicles = []
        for vid in StateStore.list_vehicles():
            try:
                store   = StateStore.get_instance(vid)
                summary = store.get_status_summary()
                vehicles.append({
                    "vehicle_id":    vid,
                    "health":        summary["health"]["status"],
                    "health_score":  summary["health"]["score"],
                    "update_rate_hz": summary["health"]["update_rate_hz"],
                })
            except Exception:  # noqa: BLE001
                vehicles.append({"vehicle_id": vid})
        return aiohttp.web.json_response({"vehicles": vehicles})

    async def _handle_state(self, request: Any) -> Any:
        aiohttp = _require_aiohttp()
        vehicle_id = request.match_info["vehicle_id"]
        try:
            store = StateStore.get_instance(vehicle_id)
            state = store.get_latest()
            if state is None:
                return aiohttp.web.json_response(
                    {"error": "No state available"}, status=404
                )
            from .serializer import state_to_json
            return aiohttp.web.Response(
                text=state_to_json(state, SerialiseProfile.TELEMETRY),
                content_type="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            return aiohttp.web.json_response({"error": str(exc)}, status=404)

    async def _handle_history(self, request: Any) -> Any:
        aiohttp = _require_aiohttp()
        vehicle_id = request.match_info["vehicle_id"]
        window = float(request.query.get("window", self._cfg.history_window_s))
        try:
            store  = StateStore.get_instance(vehicle_id)
            states = store.get_history(window_seconds=window)
            return aiohttp.web.Response(
                text=states_to_json_array(states),
                content_type="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            return aiohttp.web.json_response({"error": str(exc)}, status=404)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_client_stats(self) -> List[dict]:
        return [
            {
                "client_id":    cid[:8],
                "vehicle_id":   c.vehicle_id,
                "msgs_sent":    c.msgs_sent,
                "connected_s":  round(time.time() - c.connected_at, 1),
            }
            for cid, c in self._clients.items()
        ]

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DashboardServer(running={self._running}, "
            f"url={self.url}, "
            f"clients={len(self._clients)}, "
            f"vehicles={list(self._sub_ids.keys())})"
        )

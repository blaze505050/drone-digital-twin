"""
run_simulator.py
================
Unified launcher for the UAV 3D Flight Simulator & Digital Twin.

Starts:
  1. WebSocket telemetry server on ws://127.0.0.1:8765 (6-DOF physics @ 200Hz)
  2. Local HTTP server on http://localhost:8080 (Three.js 3D Visualiser)

Usage:
  python run_simulator.py
"""
import http.server
import socketserver
import threading
import webbrowser
import os
import sys
import time

from drone_sdk.flight_simulator import FlightSimulator, SimulatorConfig, UAVType

HTTP_PORT = 8080
SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulator")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SIM_DIR, **kwargs)

    def log_message(self, format, *args):
        # Quiet HTTP logging
        pass


def run_http_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", HTTP_PORT), Handler) as httpd:
        print(f"\n[HTTP Server] 3D Visual Simulator live at: http://localhost:{HTTP_PORT}")
        print(f"[HTTP Server] Press Ctrl+C to shut down.")
        httpd.serve_forever()


def main():
    print("=" * 65)
    print("  UAV Digital Twin — 3D Flight Simulator & Telemetry Engine")
    print("=" * 65)

    # 1. Start HTTP server in daemon thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # 2. Wait a moment for server to bind
    time.sleep(0.5)

    print(f"\n>>> LIVE SIMULATOR URL: http://localhost:{HTTP_PORT} <<<")
    print("Opening browser automatically...")
    try:
        webbrowser.open(f"http://localhost:{HTTP_PORT}")
    except Exception:
        pass

    # 3. Start Python Flight Simulator WebSocket backend (runs on main thread)
    print("\n[WebSocket] Starting 6-DOF Flight Dynamics backend on ws://127.0.0.1:8765...")
    sim = FlightSimulator(SimulatorConfig(
        uav_type=UAVType.QUADROTOR,
        physics_rate_hz=200.0,
        telemetry_rate_hz=50.0,
        enable_controller=True,
        enable_wind=True,
    ))

    try:
        sim.run_websocket_server()
    except KeyboardInterrupt:
        print("\nShutting down simulator server. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()

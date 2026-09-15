/**
 * telemetry.js — WebSocket Telemetry Client
 * Connects to Python FlightSimulator backend for real-time state streaming.
 */

export class TelemetryClient {
    constructor(url = 'ws://127.0.0.1:8765') {
        this.url = url;
        this.ws = null;
        this.connected = false;
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 16000;
        this._currentDelay = this.reconnectDelay;
        this._reconnectTimer = null;

        // Callbacks
        this.onState = null;          // (state: object) => void
        this.onConnectionChange = null; // (connected: boolean) => void
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                this.connected = true;
                this._currentDelay = this.reconnectDelay;
                console.log('[Telemetry] Connected to', this.url);
                if (this.onConnectionChange) this.onConnectionChange(true);
            };

            this.ws.onmessage = (event) => {
                try {
                    const state = JSON.parse(event.data);
                    if (this.onState) this.onState(state);
                } catch (e) {
                    // Ignore malformed messages
                }
            };

            this.ws.onclose = () => {
                this.connected = false;
                if (this.onConnectionChange) this.onConnectionChange(false);
                this._scheduleReconnect();
            };

            this.ws.onerror = (err) => {
                // Suppress error logging for connection refused (expected when backend not running)
                this.ws.close();
            };
        } catch (e) {
            this._scheduleReconnect();
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    sendInput(commands) {
        this.send(commands);
    }

    disconnect() {
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.connected = false;
    }

    _scheduleReconnect() {
        if (this._reconnectTimer) return;
        this._reconnectTimer = setTimeout(() => {
            this._reconnectTimer = null;
            this._currentDelay = Math.min(this._currentDelay * 1.5, this.maxReconnectDelay);
            this.connect();
        }, this._currentDelay);
    }
}

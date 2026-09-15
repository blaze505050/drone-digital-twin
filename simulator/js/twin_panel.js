/**
 * twin_panel.js — Digital Twin State & Residual Dashboard Overlay
 * Renders state comparison (estimated vs twin), covariance ellipsoid status,
 * and recalibration events.
 */

export class DigitalTwinPanel {
    constructor(containerId = 'hud') {
        this.container = document.getElementById(containerId);
        this.history = [];
        this.maxHistory = 50;
    }

    updateTwinData(telemetryData) {
        if (!telemetryData) return;

        // Extract digital twin telemetry metrics
        const health = telemetryData.twin_health !== undefined ? telemetryData.twin_health : 1.0;
        const phase = telemetryData.phase || 'HOVER';
        const wind = telemetryData.wind || [0, 0, 0];
        const rho = telemetryData.rho || 1.225;

        // Update DOM elements if present
        const healthEl = document.getElementById('twin-health');
        const healthBar = document.getElementById('twin-health-bar');
        const phaseEl = document.getElementById('twin-phase');
        const windEl = document.getElementById('twin-wind');

        if (healthEl) {
            healthEl.textContent = `${Math.round(health * 100)}%`;
        }
        if (healthBar) {
            healthBar.style.width = `${Math.round(health * 100)}%`;
            if (health > 0.8) {
                healthBar.style.background = '#00ff88';
            } else if (health > 0.5) {
                healthBar.style.background = '#ffd000';
            } else {
                healthBar.style.background = '#ff3366';
            }
        }
        if (phaseEl) {
            phaseEl.textContent = phase.toUpperCase();
        }
        if (windEl) {
            const windSpeed = Math.sqrt(wind[0]*wind[0] + wind[1]*wind[1] + wind[2]*wind[2]);
            windEl.textContent = `${windSpeed.toFixed(1)} m/s`;
        }

        this.history.push({
            t: telemetryData.t || 0,
            health,
            pos: telemetryData.pos || [0, 0, 0],
        });

        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
    }

    getAverageHealth() {
        if (this.history.length === 0) return 1.0;
        const sum = this.history.reduce((acc, h) => acc + h.health, 0);
        return sum / this.history.length;
    }
}

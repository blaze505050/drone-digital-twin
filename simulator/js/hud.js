/**
 * hud.js — Heads-Up Display Controller
 * Draws attitude indicator, compass, and updates all HUD data fields.
 */

export class HUDController {
    constructor() {
        this.attCanvas = document.getElementById('attitude-canvas');
        this.attCtx = this.attCanvas?.getContext('2d');
        this.compassCanvas = document.getElementById('compass-canvas');
        this.compassCtx = this.compassCanvas?.getContext('2d');
        this._lastUpdate = 0;
    }

    update(state) {
        if (!state) return;
        const now = Date.now();
        if (now - this._lastUpdate < 33) return; // Cap at ~30 FPS
        this._lastUpdate = now;

        // Time
        const t = state.t || 0;
        const mins = Math.floor(t / 60).toString().padStart(2, '0');
        const secs = (t % 60).toFixed(1).padStart(4, '0');
        this._setText('hud-time', `${mins}:${secs}`);

        // UAV type
        this._setText('hud-uav-type', (state.type || 'QUADROTOR').toUpperCase().replace('_', ' '));

        // Attitude
        const roll = state.euler?.[0] || 0;
        const pitch = state.euler?.[1] || 0;
        const yaw = state.euler?.[2] || 0;
        this._setText('att-roll', `R: ${roll.toFixed(1)}°`);
        this._setText('att-pitch', `P: ${pitch.toFixed(1)}°`);
        this._setText('att-yaw', `Y: ${yaw.toFixed(1)}°`);
        this.drawAttitudeIndicator(roll, pitch);

        // Altitude
        const alt = state.alt || 0;
        this._setText('hud-alt', alt.toFixed(1));

        // Airspeed
        this._setText('hud-speed', (state.airspeed || 0).toFixed(1));

        // Position: Real Drone
        const pos = state.pos || [0, 0, 0];
        this._setText('pos-n', pos[0].toFixed(1));
        this._setText('pos-e', pos[1].toFixed(1));
        this._setText('pos-d', pos[2].toFixed(1));

        // Position: Digital Twin
        const realPos = state.real_pos || pos;
        this._setText('twin-pos-n', realPos[0].toFixed(1));
        this._setText('twin-pos-e', realPos[1].toFixed(1));
        this._setText('twin-pos-d', realPos[2].toFixed(1));

        // Digital Twin Synchronization Status
        const syncStatus = state.sync_status || 'LOCKED';
        const syncErr = state.sync_error !== undefined ? state.sync_error : 0.02;
        const syncBadge = document.getElementById('hud-sync-badge');
        const syncStatusTag = document.getElementById('twin-sync-status');
        const health = state.twin_health || 1;

        if (syncBadge) {
            syncBadge.textContent = `${syncStatus} (${(health * 100).toFixed(0)}%)`;
            syncBadge.className = `hud-value ${syncStatus === 'LOCKED' ? 'sync-locked' : (syncStatus === 'DRIFTING' ? 'sync-drift' : 'sync-desync')}`;
        }
        if (syncStatusTag) {
            syncStatusTag.textContent = syncStatus;
            syncStatusTag.className = `twin-value sync-tag ${syncStatus === 'LOCKED' ? 'sync-locked' : (syncStatus === 'DRIFTING' ? 'sync-drift' : 'sync-desync')}`;
        }
        this._setText('twin-error', `ΔPos: ${syncErr.toFixed(2)}m`);
        this._setText('real-source-label', state.real_source || 'HIL / MAVLink');

        // Tactical FPV OSD updates
        this._setText('osd-spd', (state.airspeed || 0).toFixed(1));
        this._setText('osd-alt', alt.toFixed(1));
        this._setText('osd-batt', `${(14.8 + (batt * 2.0)).toFixed(1)}V ${(batt * 100).toFixed(0)}%`);
        const osdHorizon = document.getElementById('osd-horizon');
        if (osdHorizon) {
            osdHorizon.style.transform = `translateY(${Math.max(-40, Math.min(40, pitch * 1.5))}px) rotate(${-roll}deg)`;
        }

        // Motors
        if (state.motors) {
            for (let i = 0; i < state.motors.length && i < 8; i++) {
                const bar = document.getElementById(`motor-${i}`);
                if (bar) {
                    const pct = Math.min(100, Math.max(0, state.motors[i] * 100));
                    bar.style.setProperty('--motor-pct', `${pct}%`);
                    bar.style.cssText = `--motor-pct: ${pct}%`;
                    if (bar.querySelector || bar.style) {
                        bar.setAttribute('style', `--motor-h: ${pct}%`);
                    }
                }
            }
            this._updateMotorGrid(state.motors.length);
        }

        // Battery
        const batt = state.batt || 1;
        this._setText('battery-soc', `${(batt * 100).toFixed(0)}%`);
        const battFill = document.getElementById('battery-fill');
        if (battFill) {
            battFill.style.width = `${batt * 100}%`;
            if (batt < 0.2) battFill.style.background = 'linear-gradient(90deg, #ff4757, #ff6b6b)';
            else if (batt < 0.5) battFill.style.background = 'linear-gradient(90deg, #ff8a3d, #ffcc00)';
            else battFill.style.background = 'linear-gradient(90deg, #22d97f, #00d4ff)';
        }

        // Twin health
        this._setText('twin-health', `${(health * 100).toFixed(0)}%`);
        const healthBar = document.getElementById('twin-health-bar');
        if (healthBar) {
            healthBar.style.width = `${health * 100}%`;
            if (health < 0.5) healthBar.style.background = 'linear-gradient(90deg, #ff4757, #ff6b6b)';
            else if (health < 0.8) healthBar.style.background = 'linear-gradient(90deg, #ff8a3d, #ffcc00)';
        }

        // Phase
        this._setText('twin-phase', (state.phase || 'HOVER').toUpperCase());

        // Wind
        const wind = state.wind || [0, 0, 0];
        const windSpeed = Math.sqrt(wind[0] ** 2 + wind[1] ** 2 + wind[2] ** 2);
        this._setText('twin-wind', `${windSpeed.toFixed(1)} m/s`);

        // Compass
        this.drawCompass(yaw);
    }

    drawAttitudeIndicator(rollDeg, pitchDeg) {
        const ctx = this.attCtx;
        if (!ctx) return;
        const w = 180, h = 180;
        const cx = w / 2, cy = h / 2;
        const r = 75;

        ctx.clearRect(0, 0, w, h);

        // Clip to circle
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.clip();

        // Sky/ground split
        const rollRad = -rollDeg * Math.PI / 180;
        const pitchOffset = pitchDeg * 1.5;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rollRad);

        // Sky
        ctx.fillStyle = '#1a3a6a';
        ctx.fillRect(-r * 2, -r * 2 - pitchOffset, r * 4, r * 2);

        // Ground
        ctx.fillStyle = '#4a3a1a';
        ctx.fillRect(-r * 2, -pitchOffset, r * 4, r * 2);

        // Horizon line
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(-r, -pitchOffset);
        ctx.lineTo(r, -pitchOffset);
        ctx.stroke();

        // Pitch ladder (every 10°)
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 0.8;
        ctx.font = '8px JetBrains Mono, monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.textAlign = 'center';
        for (let p = -30; p <= 30; p += 10) {
            if (p === 0) continue;
            const py = -pitchOffset - p * 1.5;
            const lw = p % 20 === 0 ? 25 : 15;
            ctx.beginPath();
            ctx.moveTo(-lw, py);
            ctx.lineTo(lw, py);
            ctx.stroke();
            ctx.fillText(`${Math.abs(p)}`, lw + 12, py + 3);
        }

        ctx.restore();

        // Fixed aircraft reference (yellow wings)
        ctx.strokeStyle = '#ffcc00';
        ctx.lineWidth = 2.5;
        // Left wing
        ctx.beginPath();
        ctx.moveTo(cx - 35, cy);
        ctx.lineTo(cx - 15, cy);
        ctx.lineTo(cx - 15, cy + 6);
        ctx.stroke();
        // Right wing
        ctx.beginPath();
        ctx.moveTo(cx + 35, cy);
        ctx.lineTo(cx + 15, cy);
        ctx.lineTo(cx + 15, cy + 6);
        ctx.stroke();
        // Center dot
        ctx.fillStyle = '#ffcc00';
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();

        // Outer ring
        ctx.strokeStyle = 'rgba(100,180,255,0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Roll indicator triangle
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rollRad);
        ctx.fillStyle = '#00d4ff';
        ctx.beginPath();
        ctx.moveTo(0, -r + 2);
        ctx.lineTo(-5, -r + 10);
        ctx.lineTo(5, -r + 10);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    drawCompass(yawDeg) {
        const ctx = this.compassCtx;
        if (!ctx) return;
        const w = 300, h = 40;

        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = 'rgba(10,14,23,0.5)';
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2;
        const headingNorm = ((yawDeg % 360) + 360) % 360;

        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';

        const cardinals = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };

        for (let deg = headingNorm - 90; deg <= headingNorm + 90; deg += 5) {
            const normDeg = ((deg % 360) + 360) % 360;
            const px = cx + (deg - headingNorm) * 1.5;

            if (px < 10 || px > w - 10) continue;

            if (normDeg % 10 === 0) {
                ctx.strokeStyle = 'rgba(255,255,255,0.25)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(px, h - 8);
                ctx.lineTo(px, h - 2);
                ctx.stroke();
            }

            if (cardinals[normDeg]) {
                ctx.fillStyle = normDeg === 0 ? '#ff4444' : '#00d4ff';
                ctx.fillText(cardinals[normDeg], px, 14);
            } else if (normDeg % 30 === 0) {
                ctx.fillStyle = 'rgba(255,255,255,0.4)';
                ctx.fillText(`${normDeg}`, px, 14);
            }
        }

        // Center marker
        ctx.fillStyle = '#00d4ff';
        ctx.beginPath();
        ctx.moveTo(cx, h - 2);
        ctx.lineTo(cx - 4, h - 8);
        ctx.lineTo(cx + 4, h - 8);
        ctx.closePath();
        ctx.fill();

        // Heading readout
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.fillText(`${headingNorm.toFixed(0)}°`, cx, h - 12);
    }

    setConnectionStatus(connected) {
        const el = document.getElementById('hud-connection');
        if (el) {
            el.textContent = connected ? 'CONNECTED' : 'OFFLINE';
            el.className = 'hud-value hud-status-indicator' + (connected ? ' connected' : '');
        }
    }

    _updateMotorGrid(count) {
        const grid = document.getElementById('motor-grid');
        if (!grid) return;
        // Add more motor bars if needed
        while (grid.children.length < count) {
            const i = grid.children.length;
            const wrap = document.createElement('div');
            wrap.className = 'motor-bar-wrap';
            wrap.innerHTML = `<div class="motor-bar" id="motor-${i}"></div><span>M${i + 1}</span>`;
            grid.appendChild(wrap);
        }
    }

    _setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }
}

/**
 * hud.js — Heads-Up Display Controller & Tactical Minimap Radar
 * Draws attitude indicator, compass ribbon, circular tactical radar,
 * gamification challenge stats, and telemetry fields.
 */

export class HUDController {
    constructor() {
        this.attCanvas = document.getElementById('attitude-canvas');
        this.attCtx = this.attCanvas?.getContext('2d');
        this.compassCanvas = document.getElementById('compass-canvas');
        this.compassCtx = this.compassCanvas?.getContext('2d');
        this.radarCanvas = document.getElementById('radar-canvas');
        this.radarCtx = this.radarCanvas?.getContext('2d');
        this._lastUpdate = 0;
    }

    update(state, isRealConnected = false, gameState = null) {
        if (!state) return;
        const now = Date.now();
        if (now - this._lastUpdate < 30) return; // ~33 FPS
        this._lastUpdate = now;

        // Extract battery safely at the top
        const batt = state.batt !== undefined ? state.batt : 1.0;

        // Flight Time
        const t = state.t || 0;
        const mins = Math.floor(t / 60).toString().padStart(2, '0');
        const secs = (t % 60).toFixed(1).padStart(4, '0');
        this._setText('hud-time', `${mins}:${secs}`);

        // UAV vehicle type
        this._setText('hud-uav-type', (state.type || 'QUADROTOR').toUpperCase().replace('_', ' '));

        // Attitude
        const roll = state.euler?.[0] || 0;
        const pitch = state.euler?.[1] || 0;
        const yaw = state.euler?.[2] || 0;
        this._setText('att-roll', `R: ${roll.toFixed(1)}°`);
        this._setText('att-pitch', `P: ${pitch.toFixed(1)}°`);
        this._setText('att-yaw', `Y: ${yaw.toFixed(1)}°`);
        this.drawAttitudeIndicator(roll, pitch);

        // Altitude & Speed
        const alt = state.alt || 0;
        this._setText('hud-alt', alt.toFixed(1));
        this._setText('hud-speed', (state.airspeed || 0).toFixed(1));

        // Position: Virtual Digital Twin
        const twinPos = state.pos || [0, 0, 0];
        this._setText('twin-pos-n', twinPos[0].toFixed(1));
        this._setText('twin-pos-e', twinPos[1].toFixed(1));
        this._setText('twin-pos-d', twinPos[2].toFixed(1));

        // Position & Status: Physical Real Drone
        if (isRealConnected) {
            const realPos = state.real_pos || twinPos;
            this._setText('pos-n', realPos[0].toFixed(1));
            this._setText('pos-e', realPos[1].toFixed(1));
            this._setText('pos-d', realPos[2].toFixed(1));

            const syncStatus = state.sync_status || 'LOCKED';
            const syncErr = state.sync_error !== undefined ? state.sync_error : 0.02;
            const health = state.twin_health || 1.0;

            const syncBadge = document.getElementById('hud-sync-badge');
            if (syncBadge) {
                syncBadge.textContent = `${syncStatus} (${(health * 100).toFixed(0)}%)`;
                syncBadge.className = `hud-value ${syncStatus === 'LOCKED' ? 'sync-locked' : (syncStatus === 'DRIFTING' ? 'sync-drift' : 'sync-desync')}`;
            }

            const syncStatusTag = document.getElementById('twin-sync-status');
            if (syncStatusTag) {
                syncStatusTag.textContent = syncStatus;
                syncStatusTag.className = `twin-value sync-tag ${syncStatus === 'LOCKED' ? 'sync-locked' : (syncStatus === 'DRIFTING' ? 'sync-drift' : 'sync-desync')}`;
            }

            this._setText('twin-error', `ΔPos: ${syncErr.toFixed(2)}m`);
            this._setText('real-source-label', state.real_source || 'HIL / MAVLink Linked');
        } else {
            // Real drone offline: standalone virtual flight
            this._setText('pos-n', 'OFFLINE');
            this._setText('pos-e', '--');
            this._setText('pos-d', '--');

            const syncBadge = document.getElementById('hud-sync-badge');
            if (syncBadge) {
                syncBadge.textContent = 'VIRTUAL ONLY (100%)';
                syncBadge.className = 'hud-value sync-locked';
            }

            const syncStatusTag = document.getElementById('twin-sync-status');
            if (syncStatusTag) {
                syncStatusTag.textContent = 'VIRTUAL MODE';
                syncStatusTag.className = 'twin-value sync-tag sync-locked';
            }

            this._setText('twin-error', 'ΔPos: 0.00m');
            this._setText('real-source-label', 'Standalone Twin');
        }

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
                    bar.style.height = `${pct}%`;
                    bar.style.setProperty('--motor-pct', `${pct}%`);
                }
            }
            this._updateMotorGrid(state.motors.length);
        }

        // Battery Bar
        this._setText('battery-soc', `${(batt * 100).toFixed(0)}%`);
        const battFill = document.getElementById('battery-fill');
        if (battFill) {
            battFill.style.width = `${batt * 100}%`;
            if (batt < 0.2) battFill.style.background = 'linear-gradient(90deg, #ff4757, #ff6b6b)';
            else if (batt < 0.5) battFill.style.background = 'linear-gradient(90deg, #ff8a3d, #ffcc00)';
            else battFill.style.background = 'linear-gradient(90deg, #22d97f, #00d4ff)';
        }

        // Twin Health
        const health = state.twin_health || 1.0;
        this._setText('twin-health', `${(health * 100).toFixed(0)}%`);
        const healthBar = document.getElementById('twin-health-bar');
        if (healthBar) {
            healthBar.style.width = `${health * 100}%`;
            if (health < 0.5) healthBar.style.background = 'linear-gradient(90deg, #ff4757, #ff6b6b)';
            else if (health < 0.8) healthBar.style.background = 'linear-gradient(90deg, #ff8a3d, #ffcc00)';
        }

        // Flight Phase
        this._setText('twin-phase', (state.phase || 'HOVER').toUpperCase());

        // Compass Ribbon
        this.drawCompass(yaw);

        // Tactical Minimap Radar
        this.drawRadar(twinPos, yaw, gameState);

        // Update Gamification Mission Banner
        if (gameState) {
            this.updateGameHUD(gameState);
        }
    }

    updateGameHUD(gameState) {
        if (!gameState) return;
        this._setText('game-gate', `${gameState.activeGate} / ${gameState.totalGates}`);
        this._setText('game-lap', `${gameState.lapTime.toFixed(1)}s`);
        this._setText('game-score', `${gameState.score}`);
        if (gameState.bestLap !== null) {
            this._setText('game-best', `${gameState.bestLap.toFixed(1)}s`);
        }
    }

    drawRadar(dronePos, yawDeg, gameState) {
        const ctx = this.radarCtx;
        if (!ctx) return;
        const w = 160, h = 160;
        const cx = w / 2, cy = h / 2;
        const r = 74;
        const scale = 0.55; // 1 meter = 0.55 radar pixels

        ctx.clearRect(0, 0, w, h);

        // Radar background circle clip
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.clip();

        // Dark radar background
        ctx.fillStyle = '#060d18';
        ctx.fillRect(0, 0, w, h);

        // Concentric distance range rings (30m, 60m, 90m)
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.18)';
        ctx.lineWidth = 1;
        [30, 60, 90].forEach(rangeM => {
            ctx.beginPath();
            ctx.arc(cx, cy, rangeM * scale, 0, Math.PI * 2);
            ctx.stroke();
        });

        // Crosshairs
        ctx.beginPath();
        ctx.moveTo(cx, cy - r);
        ctx.lineTo(cx, cy + r);
        ctx.moveTo(cx - r, cy);
        ctx.lineTo(cx + r, cy);
        ctx.stroke();

        // Airfield Runway Representation (center)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.fillRect(-14 * scale, -100 * scale, 28 * scale, 200 * scale);
        ctx.restore();

        // Checkpoint Gates on Radar
        if (gameState && gameState.gatesList) {
            gameState.gatesList.forEach((gate, idx) => {
                const gx = cx + gate.pos[0] * scale;
                const gz = cy + gate.pos[2] * scale;

                if (idx === gameState.activeIdx) {
                    // Active Gate: Bright pulsing lime
                    ctx.fillStyle = '#00ff88';
                    ctx.shadowColor = '#00ff88';
                    ctx.shadowBlur = 8;
                    ctx.beginPath();
                    ctx.arc(gx, gz, 4, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                } else if (idx > gameState.activeIdx) {
                    // Upcoming Gate: Small cyan dot
                    ctx.fillStyle = 'rgba(0, 212, 255, 0.7)';
                    ctx.beginPath();
                    ctx.arc(gx, gz, 2.5, 0, Math.PI * 2);
                    ctx.fill();
                } else {
                    // Passed Gate: Dim purple
                    ctx.fillStyle = 'rgba(120, 100, 160, 0.35)';
                    ctx.beginPath();
                    ctx.arc(gx, gz, 2, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }

        // Drone position and Heading icon
        // Three.js: x=East, z=North
        const dx = cx + (dronePos[1] || 0) * scale;
        const dz = cy + (dronePos[0] || 0) * scale;
        const headingRad = -(yawDeg || 0) * Math.PI / 180;

        ctx.save();
        ctx.translate(dx, dz);
        ctx.rotate(headingRad);

        // Triangle heading arrow
        ctx.fillStyle = '#00f3ff';
        ctx.shadowColor = '#00f3ff';
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.moveTo(0, -6);
        ctx.lineTo(4, 5);
        ctx.lineTo(0, 3);
        ctx.lineTo(-4, 5);
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.restore();
        ctx.restore();

        // Radar Outer Glowing Border
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
    }

    drawAttitudeIndicator(rollDeg, pitchDeg) {
        const ctx = this.attCtx;
        if (!ctx) return;
        const w = 180, h = 180;
        const cx = w / 2, cy = h / 2;
        const r = 75;

        ctx.clearRect(0, 0, w, h);

        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.clip();

        const rollRad = -rollDeg * Math.PI / 180;
        const pitchOffset = pitchDeg * 1.5;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rollRad);

        // Sky
        ctx.fillStyle = '#16335a';
        ctx.fillRect(-r * 2, -r * 2 - pitchOffset, r * 4, r * 2);

        // Ground
        ctx.fillStyle = '#423318';
        ctx.fillRect(-r * 2, -pitchOffset, r * 4, r * 2);

        // Horizon
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(-r, -pitchOffset);
        ctx.lineTo(r, -pitchOffset);
        ctx.stroke();

        ctx.restore();

        // Fixed aircraft reference wings
        ctx.strokeStyle = '#ffcc00';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx - 35, cy);
        ctx.lineTo(cx - 15, cy);
        ctx.lineTo(cx - 15, cy + 6);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(cx + 35, cy);
        ctx.lineTo(cx + 15, cy);
        ctx.lineTo(cx + 15, cy + 6);
        ctx.stroke();

        ctx.fillStyle = '#ffcc00';
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();

        // Outer Ring
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
    }

    drawCompass(yawDeg) {
        const ctx = this.compassCtx;
        if (!ctx) return;
        const w = 320, h = 40;
        const cx = w / 2;
        const headingNorm = ((yawDeg % 360) + 360) % 360;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(10, 14, 23, 0.6)';
        ctx.fillRect(0, 0, w, h);

        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';

        const cardinals = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };

        for (let deg = headingNorm - 90; deg <= headingNorm + 90; deg += 5) {
            const normDeg = ((deg % 360) + 360) % 360;
            const px = cx + (deg - headingNorm) * 1.5;

            if (px < 10 || px > w - 10) continue;

            if (normDeg % 10 === 0) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
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
                ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
                ctx.fillText(`${normDeg}`, px, 14);
            }
        }

        // Center cursor
        ctx.fillStyle = '#00d4ff';
        ctx.beginPath();
        ctx.moveTo(cx, h - 2);
        ctx.lineTo(cx - 4, h - 8);
        ctx.lineTo(cx + 4, h - 8);
        ctx.closePath();
        ctx.fill();

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

/**
 * controls.js — GCS Laptop Flight Controller & Smooth Input Engine
 * Features:
 *   - Exponential stick response curves (Expo) for fine center control
 *   - Exponential Moving Average (EMA) low-pass smoothing for keyboard
 *   - Gamepad API support with configurable deadzones
 *   - Arming / Disarming safety state
 *   - Flight Mode selection (PosHold, AltHold, Stabilize, RTL)
 *   - Dual-dispatch control (Laptop controls both Real Drone & Virtual Twin)
 */

export class InputController {
    constructor() {
        this.keys = {};
        this.gamepadIndex = null;

        // Smoothed command state
        this.roll = 0.0;
        this.pitch = 0.0;
        this.yawRate = 0.0;
        this.throttle = 0.50;

        // Target raw commands (before smoothing)
        this.targetRoll = 0.0;
        this.targetPitch = 0.0;
        this.targetYaw = 0.0;
        this.targetThrottle = 0.50;

        // Flight controller state
        this.isArmed = false;
        this.flightMode = 'pos_hold'; // 'pos_hold', 'alt_hold', 'stabilize', 'rtl'
        this.dualSync = true; // Control both real and virtual twin

        // Expo & smoothing tuning
        this.expo = 0.35;         // 35% exponential rate
        this.smoothAlpha = 0.16;   // Low-pass filter factor (0.1 = very smooth, 1.0 = instant)

        // Event listeners
        window.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();
            this.keys[key] = true;

            // Shortcut toggles
            if (e.code === 'Space' && !e.repeat) {
                this.toggleArm();
            }
        });

        window.addEventListener('keyup', (e) => {
            this.keys[e.key.toLowerCase()] = false;
        });

        // Gamepad API
        window.addEventListener('gamepadconnected', (e) => {
            this.gamepadIndex = e.gamepad.index;
            console.log(`[GCS Controller] Gamepad connected: ${e.gamepad.id}`);
        });

        window.addEventListener('gamepaddisconnected', () => {
            this.gamepadIndex = null;
            console.log('[GCS Controller] Gamepad disconnected');
        });
    }

    toggleArm() {
        this.isArmed = !this.isArmed;
        const armBtn = document.getElementById('btn-arm');
        if (armBtn) {
            if (this.isArmed) {
                armBtn.textContent = 'DISARM MOTORS';
                armBtn.classList.add('armed');
            } else {
                armBtn.textContent = 'ARM MOTORS';
                armBtn.classList.remove('armed');
                this.throttle = 0.0;
                this.targetThrottle = 0.0;
            }
        }
    }

    setFlightMode(mode) {
        this.flightMode = mode;
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
    }

    applyExpo(value) {
        // Expo curve: y = x * (1 - expo) + x^3 * expo
        return value * (1.0 - this.expo) + Math.pow(value, 3) * this.expo;
    }

    getCommands() {
        if (this.gamepadIndex !== null) {
            this._pollGamepad();
        } else {
            this._pollKeyboard();
        }

        // Apply low-pass exponential smoothing (EMA) for buttery-smooth flight
        this.roll += (this.targetRoll - this.roll) * this.smoothAlpha;
        this.pitch += (this.targetPitch - this.pitch) * this.smoothAlpha;
        this.yawRate += (this.targetYaw - this.yawRate) * this.smoothAlpha;
        this.throttle += (this.targetThrottle - this.throttle) * (this.smoothAlpha * 0.8);

        // If not armed, throttle is killed
        const outputThrottle = this.isArmed ? this.throttle : 0.0;

        return {
            roll: this.applyExpo(this.roll),
            pitch: this.applyExpo(this.pitch),
            yaw_rate: this.applyExpo(this.yawRate),
            throttle: Math.max(0.0, Math.min(1.0, outputThrottle)),
            armed: this.isArmed,
            mode: this.flightMode,
            dual_sync: this.dualSync,
        };
    }

    _pollKeyboard() {
        const k = this.keys;
        const sensitivity = 0.70;
        const yawSensitivity = 1.20;

        // Roll (A / D)
        let r = 0;
        if (k['a']) r -= sensitivity;
        if (k['d']) r += sensitivity;
        this.targetRoll = r;

        // Pitch (W / S) - Forward / Backward
        let p = 0;
        if (k['w']) p += sensitivity;
        if (k['s']) p -= sensitivity;
        this.targetPitch = p;

        // Yaw (Left / Right Arrow or Q / E)
        let y = 0;
        if (k['q'] || k['arrowleft']) y -= yawSensitivity;
        if (k['e'] || k['arrowright']) y += yawSensitivity;
        this.targetYaw = y;

        // Throttle (Up / Down Arrow or Shift / Control)
        if (k['arrowup'] || k['shift']) {
            this.targetThrottle = Math.min(1.0, this.targetThrottle + 0.015);
        }
        if (k['arrowdown'] || k['control']) {
            this.targetThrottle = Math.max(0.0, this.targetThrottle - 0.015);
        }
    }

    _pollGamepad() {
        const gamepads = navigator.getGamepads();
        const gp = gamepads[this.gamepadIndex];
        if (!gp) {
            this._pollKeyboard();
            return;
        }

        const deadzone = 0.06;
        const filter = (v) => Math.abs(v) < deadzone ? 0 : v;

        // Stick mappings (Mode 2 standard: Left=Throttle/Yaw, Right=Pitch/Roll)
        const leftX = filter(gp.axes[0] || 0);
        const leftY = filter(gp.axes[1] || 0);
        const rightX = filter(gp.axes[2] || 0);
        const rightY = filter(gp.axes[3] || 0);

        this.targetRoll = rightX * 0.85;
        this.targetPitch = -rightY * 0.85; // Invert Y
        this.targetYaw = leftX * 1.5;

        // Throttle from Left Stick Y or Triggers
        const rt = gp.buttons[7]?.value || 0;
        const lt = gp.buttons[6]?.value || 0;
        if (rt > 0.05 || lt > 0.05) {
            this.targetThrottle = Math.max(0.0, Math.min(1.0, 0.5 + (rt - lt) * 0.5));
        } else {
            this.targetThrottle = Math.max(0.0, Math.min(1.0, (-(leftY) + 1.0) * 0.5));
        }

        // Arm toggle on button A (index 0) or Options (index 9)
        if (gp.buttons[0]?.pressed && !this._lastGpBtn0) {
            this.toggleArm();
        }
        this._lastGpBtn0 = gp.buttons[0]?.pressed;
    }
}

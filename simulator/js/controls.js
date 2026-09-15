/**
 * controls.js — Input Controller
 * Handles keyboard, gamepad, and touch input for drone flight control.
 */

export class InputController {
    constructor() {
        this.keys = {};
        this.gamepadIndex = null;

        // Continuous command state
        this.roll = 0;
        this.pitch = 0;
        this.yawRate = 0;
        this.throttle = 0.5;

        // Keyboard listeners
        window.addEventListener('keydown', (e) => { this.keys[e.key.toLowerCase()] = true; });
        window.addEventListener('keyup', (e) => { this.keys[e.key.toLowerCase()] = false; });

        // Gamepad connect/disconnect
        window.addEventListener('gamepadconnected', (e) => {
            this.gamepadIndex = e.gamepad.index;
            console.log(`[Controls] Gamepad connected: ${e.gamepad.id}`);
        });
        window.addEventListener('gamepaddisconnected', () => {
            this.gamepadIndex = null;
            console.log('[Controls] Gamepad disconnected');
        });
    }

    getCommands() {
        // Priority: gamepad > keyboard
        if (this.gamepadIndex !== null) {
            return this._readGamepad();
        }
        return this._readKeyboard();
    }

    _readKeyboard() {
        const k = this.keys;
        const sensitivity = 0.6;
        const throttleSens = 0.8;
        const yawSens = 1.0;

        // Roll (A/D or left/right)
        let roll = 0;
        if (k['a'] || k['arrowleft']) roll -= sensitivity;
        if (k['d'] || k['arrowright']) roll += sensitivity;

        // Pitch (W/S)
        let pitch = 0;
        if (k['w']) pitch += sensitivity;
        if (k['s']) pitch -= sensitivity;

        // Yaw (Q/E)
        let yawRate = 0;
        if (k['q']) yawRate -= yawSens;
        if (k['e']) yawRate += yawSens;

        // Throttle (up/down arrows or space/shift)
        if (k['arrowup'] || k[' ']) this.throttle = Math.min(1, this.throttle + 0.01);
        if (k['arrowdown'] || k['shift']) this.throttle = Math.max(0, this.throttle - 0.01);

        return {
            roll: roll,
            pitch: pitch,
            yaw_rate: yawRate,
            throttle: this.throttle,
            mode: 'manual',
        };
    }

    _readGamepad() {
        const gamepads = navigator.getGamepads();
        const gp = gamepads[this.gamepadIndex];
        if (!gp) return this._readKeyboard();

        // Standard mapping: left stick = roll/pitch, right stick = yaw, triggers = throttle
        const deadzone = 0.08;
        const applyDeadzone = (v) => Math.abs(v) < deadzone ? 0 : v;

        const roll = applyDeadzone(gp.axes[0] || 0);      // Left stick X
        const pitch = -applyDeadzone(gp.axes[1] || 0);     // Left stick Y (inverted)
        const yawRate = applyDeadzone(gp.axes[2] || 0);    // Right stick X

        // Throttle from triggers: RT (axis 3 or button 7) - LT
        let throttle = 0.5;
        if (gp.axes.length > 3) {
            throttle = (gp.axes[3] + 1) / 2;
        } else {
            const rt = gp.buttons[7]?.value || 0;
            const lt = gp.buttons[6]?.value || 0;
            throttle = 0.5 + (rt - lt) * 0.5;
        }

        return {
            roll: roll * 0.8,
            pitch: pitch * 0.8,
            yaw_rate: yawRate * 1.5,
            throttle: throttle,
            mode: 'manual',
        };
    }
}

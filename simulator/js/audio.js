/**
 * audio.js — Procedural Sound Synthesizer via Web Audio API
 * Generates realistic real-time multi-harmonic drone motor whine,
 * aerodynamic wind rush, checkpoint pass chimes, and arming tones.
 * Zero external audio files required.
 */

export class DroneAudioEngine {
    constructor() {
        this.ctx = null;
        this.isMuted = true; // Start muted by default to comply with browser autoplay policies
        this.isInitialized = false;

        // Motor sound nodes
        this.motorMasterGain = null;
        this.oscillators = [];
        this.oscGains = [];
        this.motorFilter = null;

        // Wind noise nodes
        this.windGain = null;
        this.windFilter = null;
        this.windNoiseNode = null;
    }

    init() {
        if (this.isInitialized) return;

        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContext();

            // Master Motor Gain
            this.motorMasterGain = this.ctx.createGain();
            this.motorMasterGain.gain.setValueAtTime(0.0, this.ctx.currentTime);

            // Low-pass filter for motor body resonance
            this.motorFilter = this.ctx.createBiquadFilter();
            this.motorFilter.type = 'lowpass';
            this.motorFilter.frequency.setValueAtTime(1400, this.ctx.currentTime);
            this.motorFilter.Q.setValueAtTime(2.5, this.ctx.currentTime);

            this.motorMasterGain.connect(this.motorFilter);
            this.motorFilter.connect(this.ctx.destination);

            // 4 Harmonically tuned oscillators for quadcopter motor hum (BLDC PWM whine)
            const harmonics = [1.0, 1.98, 3.02, 4.05];
            const harmonicGains = [0.45, 0.28, 0.15, 0.08];

            harmonics.forEach((h, idx) => {
                const osc = this.ctx.createOscillator();
                osc.type = idx % 2 === 0 ? 'sawtooth' : 'triangle';
                osc.frequency.setValueAtTime(120 * h, this.ctx.currentTime);

                const g = this.ctx.createGain();
                g.gain.setValueAtTime(harmonicGains[idx], this.ctx.currentTime);

                osc.connect(g);
                g.connect(this.motorMasterGain);
                osc.start();

                this.oscillators.push({ osc, baseHarmonic: h });
                this.oscGains.push(g);
            });

            // Wind Noise Generator (White noise through bandpass filter)
            this._setupWindNoise();

            this.isInitialized = true;
            this.isMuted = false;
        } catch (e) {
            console.warn('[AudioEngine] Web Audio not available:', e);
        }
    }

    _setupWindNoise() {
        if (!this.ctx) return;
        const bufferSize = this.ctx.sampleRate * 2;
        const noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }

        const whiteNoise = this.ctx.createBufferSource();
        whiteNoise.buffer = noiseBuffer;
        whiteNoise.loop = true;

        this.windFilter = this.ctx.createBiquadFilter();
        this.windFilter.type = 'bandpass';
        this.windFilter.frequency.setValueAtTime(450, this.ctx.currentTime);
        this.windFilter.Q.setValueAtTime(1.2, this.ctx.currentTime);

        this.windGain = this.ctx.createGain();
        this.windGain.gain.setValueAtTime(0.0, this.ctx.currentTime);

        whiteNoise.connect(this.windFilter);
        this.windFilter.connect(this.windGain);
        this.windGain.connect(this.ctx.destination);

        whiteNoise.start();
        this.windNoiseNode = whiteNoise;
    }

    toggleMute() {
        if (!this.isInitialized) {
            this.init();
            return !this.isMuted;
        }

        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }

        this.isMuted = !this.isMuted;
        if (this.isMuted) {
            if (this.motorMasterGain) this.motorMasterGain.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.05);
            if (this.windGain) this.windGain.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.05);
        }
        return !this.isMuted;
    }

    update(throttle, airspeed, isArmed) {
        if (!this.isInitialized || this.isMuted || !this.ctx) return;
        if (this.ctx.state === 'suspended') return;

        const t = this.ctx.currentTime;
        const nowThrottle = Math.max(0.0, Math.min(1.0, throttle || 0.0));
        const nowAirspeed = Math.max(0.0, airspeed || 0.0);

        if (!isArmed) {
            // Idle pitch or silent
            if (this.motorMasterGain) this.motorMasterGain.gain.setTargetAtTime(0.0, t, 0.1);
            if (this.windGain) this.windGain.gain.setTargetAtTime(0.0, t, 0.1);
            return;
        }

        // Base RPM frequency: 100 Hz idle up to 480 Hz full throttle
        const baseFreq = 95.0 + nowThrottle * 380.0;
        this.oscillators.forEach(({ osc, baseHarmonic }) => {
            osc.frequency.setTargetAtTime(baseFreq * baseHarmonic, t, 0.04);
        });

        // Motor volume responds to throttle
        const targetMotorVol = 0.06 + nowThrottle * 0.18;
        this.motorMasterGain.gain.setTargetAtTime(targetMotorVol, t, 0.04);

        // Filter cutoff opens up with high throttle
        const cutoff = 800 + nowThrottle * 2800;
        this.motorFilter.frequency.setTargetAtTime(cutoff, t, 0.05);

        // Wind noise volume and frequency increase with airspeed
        const windVol = Math.min(0.22, (nowAirspeed / 28.0) * 0.22);
        this.windGain.gain.setTargetAtTime(windVol, t, 0.08);
        this.windFilter.frequency.setTargetAtTime(300 + nowAirspeed * 40, t, 0.08);
    }

    playCheckpointChime() {
        if (!this.isInitialized || this.isMuted || !this.ctx) return;
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const t = this.ctx.currentTime;
        // Two-tone rising harmonic arpeggio (C6 -> G6)
        const notes = [1046.5, 1567.98];
        notes.forEach((freq, i) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();

            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, t + i * 0.08);

            gain.gain.setValueAtTime(0.0, t + i * 0.08);
            gain.gain.linearRampToValueAtTime(0.25, t + i * 0.08 + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.08 + 0.35);

            osc.connect(gain);
            gain.connect(this.ctx.destination);

            osc.start(t + i * 0.08);
            osc.stop(t + i * 0.08 + 0.4);
        });
    }

    playVictoryJingle() {
        if (!this.isInitialized || this.isMuted || !this.ctx) return;
        const t = this.ctx.currentTime;
        const chord = [523.25, 659.25, 783.99, 1046.5]; // C Major triumph
        chord.forEach((freq, idx) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();

            osc.type = 'triangle';
            osc.frequency.setValueAtTime(freq, t + idx * 0.1);

            gain.gain.setValueAtTime(0.0, t + idx * 0.1);
            gain.gain.linearRampToValueAtTime(0.2, t + idx * 0.1 + 0.03);
            gain.gain.exponentialRampToValueAtTime(0.001, t + idx * 0.1 + 0.8);

            osc.connect(gain);
            gain.connect(this.ctx.destination);

            osc.start(t + idx * 0.1);
            osc.stop(t + idx * 0.1 + 0.9);
        });
    }

    playAlertBeep() {
        if (!this.isInitialized || this.isMuted || !this.ctx) return;
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = 'square';
        osc.frequency.setValueAtTime(880, t);

        gain.gain.setValueAtTime(0.12, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start(t);
        osc.stop(t + 0.13);
    }
}

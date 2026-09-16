/**
 * main.js — 3D Visual Flight Dynamics Simulator & Digital Twin Command Station
 *
 * Features:
 *   - Standalone Virtual Twin flight by default (Real Drone & Sync Beam hidden)
 *   - Physical Real Drone telemetry link mode (Key: L / Button toggle)
 *   - Live Onboard Camera POV with tactical FPV OSD & webcam video input
 *   - Tactical Minimap Airfield Radar with ring gates and drone heading
 *   - Interactive 8-gate aerobatic ring racing course with particle bursts
 *   - Web Audio API procedural drone throttle whine & wind airflow sound engine
 *   - 6-DOF physics interpolation with smooth exponential stick controls
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';

import { createScene, updateScene, updateGroundEffect } from './scene.js';
import { DroneBuilder, updateDronePropellers, updateSyncBeam } from './drone.js';
import { HUDController } from './hud.js';
import { InputController } from './controls.js';
import { TelemetryClient } from './telemetry.js';
import { DroneAudioEngine } from './audio.js';
import { RingCourseManager } from './game.js';

// ── Global References ────────────────────────────────────────────────────

let renderer, scene, camera, composer;
let fpvRenderer, fpvCamera;
let realDrone, twinDrone, syncBeam, droneBuilder;
let hud, input, telemetry, audioEngine, gameManager;
let clock;
let currentUAVType = 'quadrotor';

// Camera Modes
const CAMERA_MODES = ['chase', 'fpv', 'orbit', 'top', 'cinematic'];
let cameraMode = 0;
let orbitAngle = 0;

// Viewport Modes
let isSwapped = false;
let isSplit = false;
let isWebcamActive = false;
let webcamStream = null;

// Hardware Connection State: Default is FALSE (Virtual Drone Only)
let isRealDroneLinked = false;

// Pose Smoothing Targets
let targetRealPos = new THREE.Vector3(0, 2, 0);
let targetRealQuat = new THREE.Quaternion();
let targetTwinPos = new THREE.Vector3(0, 2, 0);
let targetTwinQuat = new THREE.Quaternion();
let latestTelemetryState = null;

// ── Initialisation ───────────────────────────────────────────────────────

async function init() {
    const loadProgress = document.getElementById('load-progress');
    const loadStatus = document.getElementById('load-status');

    // 1. Audio Engine
    audioEngine = new DroneAudioEngine();

    // 2. Primary WebGL Renderer (Main 3D Viewport)
    loadStatus.textContent = 'Initialising high-performance WebGL renderer...';
    loadProgress.style.width = '12%';

    const canvas = document.getElementById('simulator-canvas');
    renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // 3. 3D Scene Environment
    loadStatus.textContent = 'Generating realistic terrain, airfield & sky dome...';
    loadProgress.style.width = '28%';

    scene = new THREE.Scene();
    createScene(scene);

    // 4. Main Camera & Post-Processing Pipeline
    loadStatus.textContent = 'Configuring camera & bloom passes...';
    loadProgress.style.width = '42%';

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.4, 5000);
    camera.position.set(-8, 6, 8);
    camera.lookAt(0, 3, 0);

    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        0.32,  // Bloom strength
        0.48,  // Radius
        0.84   // Threshold
    );
    composer.addPass(bloomPass);

    const smaaPass = new SMAAPass(window.innerWidth, window.innerHeight);
    composer.addPass(smaaPass);

    // 5. Secondary FPV Onboard Camera & Renderer
    loadStatus.textContent = 'Mounting tactical drone onboard camera...';
    loadProgress.style.width = '55%';

    const fpvCanvas = document.getElementById('fpv-canvas');
    fpvRenderer = new THREE.WebGLRenderer({
        canvas: fpvCanvas,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
    });
    fpvRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    fpvRenderer.setSize(320, 185);
    fpvRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    fpvRenderer.toneMappingExposure = 1.3;

    // Wide 90° FOV like real FPV drone cameras
    fpvCamera = new THREE.PerspectiveCamera(90, 320 / 185, 0.1, 3500);

    // 6. Spawn Drones & Sync Vector Beam
    loadStatus.textContent = 'Spawning Virtual Digital Twin & Physical Drone...';
    loadProgress.style.width = '68%';

    droneBuilder = new DroneBuilder(scene);
    buildDrones(currentUAVType);

    // 7. Gamification Course Manager (8-Gate Aerobatic Ring Challenge)
    loadStatus.textContent = 'Deploying aerial ring racing challenge course...';
    loadProgress.style.width = '78%';

    gameManager = new RingCourseManager(scene, audioEngine);

    // 8. HUD & Laptop GCS Flight Controller
    loadStatus.textContent = 'Configuring HUD instruments & flight controls...';
    loadProgress.style.width = '88%';

    hud = new HUDController();
    input = new InputController();

    // 9. Telemetry WebSocket Bridge
    loadStatus.textContent = 'Connecting telemetry bridge...';
    loadProgress.style.width = '94%';

    telemetry = new TelemetryClient('ws://127.0.0.1:8765');

    telemetry.onState = (state) => {
        latestTelemetryState = state;

        // If backend explicitly flags physical hardware connected
        if (state.real_connected !== undefined && state.real_connected !== isRealDroneLinked) {
            setRealDroneLinked(state.real_connected);
        }

        // Virtual twin pose
        const twinP = state.pos || [0, 0, -2];
        const twinE = state.euler || [0, 0, 0];
        targetTwinPos.set(twinP[1], -twinP[2], twinP[0]);
        targetTwinQuat.setFromEuler(new THREE.Euler(
            (twinE[1] || 0) * Math.PI / 180,
            -(twinE[2] || 0) * Math.PI / 180,
            -(twinE[0] || 0) * Math.PI / 180,
            'YXZ'
        ));

        // Real physical drone pose
        const realP = state.real_pos || twinP;
        const realE = state.real_euler || twinE;
        targetRealPos.set(realP[1], -realP[2], realP[0]);
        targetRealQuat.setFromEuler(new THREE.Euler(
            (realE[1] || 0) * Math.PI / 180,
            -(realE[2] || 0) * Math.PI / 180,
            -(realE[0] || 0) * Math.PI / 180,
            'YXZ'
        ));
    };

    telemetry.onConnectionChange = (connected) => {
        hud.setConnectionStatus(connected);
    };

    telemetry.connect();

    // 10. Reveal Simulation Viewports
    loadStatus.textContent = 'Simulation ready.';
    loadProgress.style.width = '100%';

    clock = new THREE.Clock();

    setTimeout(() => {
        document.getElementById('loading-screen').classList.add('hidden');
        document.getElementById('hud').style.display = 'block';
        document.getElementById('fpv-pip-container').style.display = 'flex';
        document.getElementById('type-selector').style.display = 'flex';
        document.getElementById('camera-mode').style.display = 'block';
        updateRealDroneVisibility();
    }, 450);

    // Event listeners
    window.addEventListener('resize', onResize);
    setupUIControls();

    // Start 60 FPS animation loop
    animate();
}

function buildDrones(type) {
    if (realDrone) scene.remove(realDrone);
    if (twinDrone) scene.remove(twinDrone);
    if (syncBeam) scene.remove(syncBeam);

    // 1. Real Drone (matte carbon composite)
    realDrone = droneBuilder.build(type, false);
    // 2. Virtual Digital Twin (holographic cyber-cyan)
    twinDrone = droneBuilder.build(type, true);
    // 3. Dynamic Sync Vector Beam
    syncBeam = droneBuilder.createSyncBeam();

    updateRealDroneVisibility();
}

// ── Physical Drone vs Virtual Twin Visibility ────────────────────────────

function setRealDroneLinked(linked) {
    isRealDroneLinked = linked;
    updateRealDroneVisibility();
    if (telemetry && telemetry.connected) {
        telemetry.send({ real_connected: isRealDroneLinked });
    }
}

function toggleRealDroneLink() {
    setRealDroneLinked(!isRealDroneLinked);
    if (audioEngine) {
        if (isRealDroneLinked) audioEngine.playCheckpointChime();
        else audioEngine.playAlertBeep();
    }
}

function updateRealDroneVisibility() {
    // 1. In 3D Scene: Only show Real Drone & Sync Beam if linked!
    if (realDrone) realDrone.visible = isRealDroneLinked;
    if (syncBeam) syncBeam.visible = isRealDroneLinked;
    if (twinDrone) twinDrone.visible = true; // Virtual twin always visible

    // 2. Button in Top GCS Bar
    const linkBtn = document.getElementById('btn-link-real');
    const linkText = document.getElementById('link-btn-text');
    if (linkBtn && linkText) {
        linkBtn.className = 'hud-link-btn' + (isRealDroneLinked ? ' linked' : '');
        linkText.textContent = isRealDroneLinked ? 'REAL DRONE LINKED' : 'LINK REAL DRONE';
    }

    // 3. FPV Picture-in-Picture Panel & Standby Overlay
    const standbyOverlay = document.getElementById('fpv-standby');
    const fpvOsd = document.getElementById('fpv-osd');
    const fpvTitleText = document.getElementById('fpv-title-text');
    const fpvRecDot = document.getElementById('fpv-rec-dot');

    if (standbyOverlay && fpvOsd) {
        if (isRealDroneLinked) {
            // Live Real Drone POV Active
            standbyOverlay.style.display = 'none';
            fpvOsd.style.display = 'block';
            if (fpvTitleText) fpvTitleText.textContent = 'REAL DRONE CAM POV [LIVE]';
            if (fpvRecDot) fpvRecDot.style.background = '#00ff88';
        } else {
            // Real Drone Offline: Show Standby Radar Screen
            standbyOverlay.style.display = 'flex';
            fpvOsd.style.display = 'none';
            if (fpvTitleText) fpvTitleText.textContent = 'REAL DRONE CAM POV [OFFLINE]';
            if (fpvRecDot) fpvRecDot.style.background = '#ff4757';
        }
    }
}

// ── Render Loop ──────────────────────────────────────────────────────────

function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();
    const elapsed = clock.getElapsedTime();

    // Flight control input from laptop keyboard / gamepad
    const cmds = input.getCommands();

    // Update scene animations (sun, water, wind turbines, ATC radar dish, windsock)
    updateScene(scene, elapsed, delta, [2.0, 1.0, 0.0]);

    // Send control inputs over telemetry WebSocket
    if (telemetry && telemetry.connected) {
        telemetry.sendInput(cmds);
    } else {
        runOfflineFlight(elapsed, delta);
    }

    // Determine active drone position for ground effect and gamification
    const activeDrone = isRealDroneLinked ? realDrone : twinDrone;
    const activeDronePos = activeDrone ? activeDrone.position : new THREE.Vector3(0, 2, 0);

    // Smooth position & rotation interpolation (LERP & SLERP)
    if (twinDrone) {
        twinDrone.position.lerp(targetTwinPos, Math.min(1.0, 18.0 * delta));
        twinDrone.quaternion.slerp(targetTwinQuat, Math.min(1.0, 18.0 * delta));
        updateDronePropellers(twinDrone, [cmds.throttle, cmds.throttle, cmds.throttle, cmds.throttle], delta);
    }

    if (realDrone) {
        realDrone.position.lerp(targetRealPos, Math.min(1.0, 18.0 * delta));
        realDrone.quaternion.slerp(targetRealQuat, Math.min(1.0, 18.0 * delta));
        updateDronePropellers(realDrone, [cmds.throttle, cmds.throttle, cmds.throttle, cmds.throttle], delta);
    }

    // Dynamic 3D Sync Vector Beam between Real Drone & Virtual Twin
    if (syncBeam && isRealDroneLinked && realDrone && twinDrone) {
        updateSyncBeam(syncBeam, realDrone, twinDrone);
    }

    // Ground effect dust particle ring
    if (activeDrone) {
        updateGroundEffect(activeDrone.position, cmds.throttle, delta);
    }

    // Update Gamification Course (Active Gate checking, particles, lap timer)
    if (gameManager && activeDrone) {
        gameManager.update(activeDrone.position, delta, elapsed);
    }

    // Update Web Audio engine (throttle pitch shifting & airspeed wind noise)
    const currentSpeed = latestTelemetryState?.airspeed || 5.0;
    if (audioEngine) {
        audioEngine.update(cmds.throttle, currentSpeed, input.isArmed);
    }

    // Update HUD & Tactical Minimap Radar
    if (hud && latestTelemetryState) {
        hud.update(latestTelemetryState, isRealDroneLinked, gameManager ? gameManager.getGameState() : null);
    }

    // Update Cameras
    updateMainCamera(delta, elapsed, activeDrone);
    updateFPVCamera();

    // ── Primary Viewport Render ──
    if (isSwapped) {
        // Swap: Main viewport renders FPV camera
        renderer.render(scene, fpvCamera);
    } else {
        // Normal: Main viewport renders chase/orbit/top camera with bloom
        composer.render();
    }

    // ── Secondary FPV PiP Viewport Render ──
    if (isRealDroneLinked && !isWebcamActive && fpvRenderer) {
        if (isSwapped) {
            fpvRenderer.render(scene, camera);
        } else {
            fpvRenderer.render(scene, fpvCamera);
        }
    }
}

// ── Camera Control ───────────────────────────────────────────────────────

function updateMainCamera(delta, elapsed, targetDrone) {
    if (!targetDrone) return;

    const dronePos = targetDrone.position.clone();
    const mode = CAMERA_MODES[cameraMode];

    switch (mode) {
        case 'chase': {
            const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(targetDrone.quaternion);
            const offset = fwd.clone().multiplyScalar(-7.5).add(new THREE.Vector3(0, 3.4, 0));
            const targetCamPos = dronePos.clone().add(offset);
            camera.position.lerp(targetCamPos, 6.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 0.5, dronePos.z);
            break;
        }
        case 'fpv': {
            camera.position.copy(dronePos).add(new THREE.Vector3(0, 0.18, 0));
            const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(targetDrone.quaternion);
            camera.lookAt(dronePos.clone().add(fwd.multiplyScalar(25)));
            break;
        }
        case 'orbit': {
            orbitAngle += 0.35 * delta;
            const r = 13;
            const ox = dronePos.x + r * Math.cos(orbitAngle);
            const oz = dronePos.z + r * Math.sin(orbitAngle);
            camera.position.lerp(new THREE.Vector3(ox, dronePos.y + 4.2, oz), 4.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 0.5, dronePos.z);
            break;
        }
        case 'top': {
            const topPos = new THREE.Vector3(dronePos.x, dronePos.y + 26, dronePos.z + 0.01);
            camera.position.lerp(topPos, 4.0 * delta);
            camera.lookAt(dronePos);
            break;
        }
        case 'cinematic': {
            const cinAngle = elapsed * 0.18;
            const cx = dronePos.x + 20 * Math.cos(cinAngle);
            const cz = dronePos.z + 20 * Math.sin(cinAngle);
            const cy = dronePos.y + 3.8 + Math.sin(elapsed * 0.4) * 2.0;
            camera.position.lerp(new THREE.Vector3(cx, cy, cz), 2.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 0.8, dronePos.z);
            break;
        }
    }
}

function updateFPVCamera() {
    const droneToMount = isRealDroneLinked && realDrone ? realDrone : twinDrone;
    if (!droneToMount || !fpvCamera) return;

    // Mount camera right on front nose
    const mountPos = droneToMount.position.clone().add(
        new THREE.Vector3(0, 0.12, 0).applyQuaternion(droneToMount.quaternion)
    );
    fpvCamera.position.copy(mountPos);

    const forwardVec = new THREE.Vector3(0, 0, -1).applyQuaternion(droneToMount.quaternion);
    fpvCamera.lookAt(mountPos.clone().add(forwardVec.multiplyScalar(30)));
}

// ── Offline Flight Simulation Mode ───────────────────────────────────────

function runOfflineFlight(t, delta) {
    // Smooth autonomous figure-8 trajectory for virtual twin
    const r = 16.0;
    const speed = 0.35;
    const xTwin = r * Math.sin(speed * t);
    const zTwin = r * Math.sin(speed * t * 2) * 0.5;
    const yTwin = 6.5 + Math.sin(t * 0.6) * 1.8;

    targetTwinPos.set(xTwin, yTwin, zTwin);
    targetTwinQuat.setFromEuler(new THREE.Euler(
        Math.sin(t * 1.4) * 0.14,
        -speed * t + Math.PI / 2,
        Math.cos(t * 0.9) * 0.12,
        'YXZ'
    ));

    // Real drone follows with micro latency & vibration if linked
    const dtLag = 0.14;
    const xReal = r * Math.sin(speed * (t - dtLag));
    const zReal = r * Math.sin(speed * (t - dtLag) * 2) * 0.5;
    const yReal = 6.5 + Math.sin((t - dtLag) * 0.6) * 1.8;

    targetRealPos.set(xReal, yReal, zReal);
    targetRealQuat.setFromEuler(new THREE.Euler(
        Math.sin((t - dtLag) * 1.4) * 0.14,
        -speed * (t - dtLag) + Math.PI / 2,
        Math.cos((t - dtLag) * 0.9) * 0.12,
        'YXZ'
    ));

    const syncError = targetRealPos.distanceTo(targetTwinPos);
    const syncStatus = syncError < 0.22 ? 'LOCKED' : (syncError < 0.55 ? 'DRIFTING' : 'DESYNC');

    latestTelemetryState = {
        t: t,
        pos: [zTwin, xTwin, -yTwin],
        real_pos: [zReal, xReal, -yReal],
        euler: [Math.sin(t * 1.4) * 8.0, Math.cos(t * 0.9) * 6.0, (-speed * t * 57.3) % 360],
        real_euler: [Math.sin(t * 1.4) * 8.0, Math.cos(t * 0.9) * 6.0, (-speed * t * 57.3) % 360],
        motors: [0.55, 0.55, 0.55, 0.55],
        alt: yTwin,
        airspeed: r * speed,
        batt: Math.max(0.4, 1.0 - t / 750),
        type: currentUAVType,
        phase: 'FLIGHT',
        wind: [2.0, 1.0, 0.0],
        twin_health: Math.max(0.0, 1.0 - (syncError / 1.5)),
        sync_error: syncError,
        sync_status: syncStatus,
        real_source: 'Hardware HIL Stream (MAVLink Active)',
        real_connected: isRealDroneLinked,
    };
}

// ── UI Controls & Listeners ──────────────────────────────────────────────

function setupUIControls() {
    // 1. Real Drone Link Buttons (Top bar & FPV Standby screen)
    const linkBtn = document.getElementById('btn-link-real');
    if (linkBtn) linkBtn.addEventListener('click', toggleRealDroneLink);

    const standbyLinkBtn = document.getElementById('btn-standby-link');
    if (standbyLinkBtn) standbyLinkBtn.addEventListener('click', toggleRealDroneLink);

    // 2. Audio Mute / Unmute Button
    const audioBtn = document.getElementById('btn-audio');
    if (audioBtn) {
        audioBtn.addEventListener('click', () => {
            if (!audioEngine) return;
            const isUnmuted = audioEngine.toggleMute();
            audioBtn.classList.toggle('unmuted', isUnmuted);
            const icon = document.getElementById('audio-btn-icon');
            if (icon) icon.textContent = isUnmuted ? '🔊 AUDIO ON' : '🔇 MUTE';
        });
    }

    // 3. Reset Aerial Ring Race Button
    const resetRaceBtn = document.getElementById('btn-reset-race');
    if (resetRaceBtn) {
        resetRaceBtn.addEventListener('click', () => {
            if (gameManager) gameManager.resetCourse();
        });
    }

    // 4. UAV Type Selector Buttons
    document.querySelectorAll('.type-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.dataset.type;
            if (type === currentUAVType) return;

            document.querySelector('.type-btn.active')?.classList.remove('active');
            btn.classList.add('active');
            currentUAVType = type;

            buildDrones(type);

            if (telemetry && telemetry.connected) {
                telemetry.send({ uav_type: type });
            }
        });
    });

    // 5. Laptop GCS Arm / Disarm Button
    const armBtn = document.getElementById('btn-arm');
    if (armBtn) {
        armBtn.addEventListener('click', () => {
            input.toggleArm();
            if (audioEngine) audioEngine.playAlertBeep();
        });
    }

    // 6. Laptop GCS Flight Mode Buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            input.setFlightMode(mode);
            if (telemetry && telemetry.connected) {
                telemetry.send({ mode: mode });
            }
        });
    });

    // 7. Dual Sync Toggle Checkbox
    const dualChk = document.getElementById('chk-sync-dual');
    if (dualChk) {
        dualChk.addEventListener('change', (e) => {
            input.dualSync = e.target.checked;
        });
    }

    // 8. Swap & Split View Buttons
    const swapBtn = document.getElementById('btn-swap-cam');
    if (swapBtn) swapBtn.addEventListener('click', toggleSwapView);

    const splitBtn = document.getElementById('btn-split-cam');
    if (splitBtn) splitBtn.addEventListener('click', toggleSplitView);

    // 9. USB Webcam / Video-In Toggle
    const webcamBtn = document.getElementById('btn-webcam');
    if (webcamBtn) webcamBtn.addEventListener('click', toggleWebcam);

    // 10. Global Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
        const key = e.key.toLowerCase();

        // Cycle camera mode with 'c'
        if (key === 'c') {
            cameraMode = (cameraMode + 1) % CAMERA_MODES.length;
            const label = CAMERA_MODES[cameraMode].toUpperCase().replace('_', ' ') + ' CAM (3D TWIN)';
            document.getElementById('cam-label').textContent = label;
        }

        // Swap FPV and Main View with 'v'
        if (key === 'v') {
            toggleSwapView();
        }

        // Toggle Real Drone Link with 'l'
        if (key === 'l') {
            toggleRealDroneLink();
        }

        // UAV Type shortcuts (1-5)
        const typeKeys = { '1': 'quadrotor', '2': 'hexarotor', '3': 'octorotor', '4': 'fixed_wing', '5': 'vtol' };
        if (typeKeys[key]) {
            const btn = document.querySelector(`.type-btn[data-type="${typeKeys[key]}"]`);
            if (btn) btn.click();
        }
    });
}

function toggleSwapView() {
    isSwapped = !isSwapped;
    const pipPanel = document.getElementById('fpv-pip-container');
    if (pipPanel) pipPanel.classList.toggle('swapped', isSwapped);
}

function toggleSplitView() {
    isSplit = !isSplit;
    document.body.classList.toggle('split-active', isSplit);
    onResize();
}

async function toggleWebcam() {
    const video = document.getElementById('webcam-video');
    const fpvCanvas = document.getElementById('fpv-canvas');
    const btn = document.getElementById('btn-webcam');

    if (isWebcamActive) {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        video.style.display = 'none';
        fpvCanvas.style.display = 'block';
        btn.textContent = 'CAM IN';
        btn.style.background = '';
        isWebcamActive = false;
    } else {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            });
            video.srcObject = webcamStream;
            video.style.display = 'block';
            fpvCanvas.style.display = 'none';
            btn.textContent = '3D CAM';
            btn.style.background = '#00ff88';
            isWebcamActive = true;
        } catch (err) {
            console.warn('[Camera] Could not open video device:', err);
            alert('No external video capture device found. Using 3D onboard gimbal camera.');
        }
    }
}

function onResize() {
    const w = isSplit ? window.innerWidth * 0.5 : window.innerWidth;
    const h = window.innerHeight;

    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    renderer.setSize(w, h);
    composer.setSize(w, h);

    const fpvContainer = document.getElementById('fpv-viewport');
    if (fpvContainer && fpvRenderer) {
        const fpvW = isSplit ? window.innerWidth * 0.5 : 320;
        const fpvH = isSplit ? window.innerHeight : 185;
        fpvCamera.aspect = fpvW / fpvH;
        fpvCamera.updateProjectionMatrix();
        fpvRenderer.setSize(fpvW, fpvH);
    }
}

// ── Boot ─────────────────────────────────────────────────────────────────

init().catch(err => {
    console.error('Failed to initialize simulator:', err);
    const status = document.getElementById('load-status');
    if (status) status.textContent = `Error: ${err.message}`;
});

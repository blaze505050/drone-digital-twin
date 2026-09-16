/**
 * UAV Digital Twin — 3D Flight Simulator & Ground Control Station (GCS)
 * 
 * Features:
 *   - Real Drone vs Virtual Digital Twin simultaneous 3D sync
 *   - Live Onboard FPV Camera POV in Picture-in-Picture / Split-Screen
 *   - Tactical FPV OSD (artificial horizon, crosshairs, targeting)
 *   - Laptop Flight Controller (Arm/Disarm, Flight Modes, Smooth Expo input)
 *   - Multi-UAV geometry switching (Quad, Hex, Octo, Fixed-Wing, VTOL)
 *   - Dynamic Sync Beam & Ground Effect rotor dust particles
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';

import { createScene, updateScene, updateGroundEffect } from './scene.js';
import { DroneBuilder, updateDronePosition, updateDronePropellers, updateSyncBeam } from './drone.js';
import { HUDController } from './hud.js';
import { InputController } from './controls.js';
import { TelemetryClient } from './telemetry.js';

// ── Globals ──────────────────────────────────────────────────────────────

let renderer, scene, camera, composer;
let fpvRenderer, fpvCamera;
let realDrone, twinDrone, syncBeam, droneBuilder;
let hud, input, telemetry;
let clock;
let currentUAVType = 'quadrotor';

// Camera modes for main viewport
const CAMERA_MODES = ['chase', 'fpv', 'orbit', 'top', 'cinematic'];
let cameraMode = 0;
let orbitAngle = 0;

// Dual View State
let isSwapped = false;     // Swap main and FPV views
let isSplit = false;       // 50/50 split screen
let isWebcamActive = false;
let webcamStream = null;

// Target state smoothing
let targetRealPos = new THREE.Vector3(0, 2, 0);
let targetRealQuat = new THREE.Quaternion();
let targetTwinPos = new THREE.Vector3(0, 2, 0);
let targetTwinQuat = new THREE.Quaternion();

// ── Initialisation ───────────────────────────────────────────────────────

async function init() {
    const loadProgress = document.getElementById('load-progress');
    const loadStatus = document.getElementById('load-status');

    // 1. Primary Renderer (Main 3D Viewport)
    loadStatus.textContent = 'Creating primary 3D engine...';
    loadProgress.style.width = '10%';

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

    // 2. Scene
    loadStatus.textContent = 'Generating 3D terrain & airfield...';
    loadProgress.style.width = '25%';

    scene = new THREE.Scene();
    createScene(scene);

    // 3. Main Camera & Post-processing
    loadStatus.textContent = 'Configuring camera pipeline...';
    loadProgress.style.width = '40%';

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.4, 5000);
    camera.position.set(-8, 6, 8);
    camera.lookAt(0, 3, 0);

    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        0.35,  // bloom strength
        0.5,   // radius
        0.82   // threshold
    );
    composer.addPass(bloomPass);

    const smaaPass = new SMAAPass(window.innerWidth, window.innerHeight);
    composer.addPass(smaaPass);

    // 4. Secondary FPV Onboard Camera & Renderer
    loadStatus.textContent = 'Mounting onboard drone camera...';
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

    // FPV camera: Wide-angle 90° FOV like a real drone camera
    fpvCamera = new THREE.PerspectiveCamera(90, 320 / 185, 0.1, 3500);

    // 5. Dual Drones & Sync Beam
    loadStatus.textContent = 'Spawning Real Drone and Digital Twin...';
    loadProgress.style.width = '70%';

    droneBuilder = new DroneBuilder(scene);
    buildDrones(currentUAVType);

    // 6. HUD & Laptop Flight Controller
    loadStatus.textContent = 'Configuring GCS flight controller...';
    loadProgress.style.width = '85%';

    hud = new HUDController();
    input = new InputController();

    // 7. Telemetry Bridge
    loadStatus.textContent = 'Connecting telemetry bridge...';
    loadProgress.style.width = '92%';

    telemetry = new TelemetryClient('ws://127.0.0.1:8765');

    telemetry.onState = (state) => {
        hud.update(state);

        // Update real drone target pose (from physical drone or HIL)
        const realP = state.real_pos || state.pos || [0, 0, -2];
        const realE = state.real_euler || state.euler || [0, 0, 0];
        targetRealPos.set(realP[1], -realP[2], realP[0]); // Three.js: x=E, y=Up, z=N
        targetRealQuat.setFromEuler(new THREE.Euler(
            (realE[1] || 0) * Math.PI / 180,
            -(realE[2] || 0) * Math.PI / 180,
            -(realE[0] || 0) * Math.PI / 180,
            'YXZ'
        ));

        // Update virtual twin target pose (pure physics simulation)
        const twinP = state.pos || [0, 0, -2];
        const twinE = state.euler || [0, 0, 0];
        targetTwinPos.set(twinP[1], -twinP[2], twinP[0]);
        targetTwinQuat.setFromEuler(new THREE.Euler(
            (twinE[1] || 0) * Math.PI / 180,
            -(twinE[2] || 0) * Math.PI / 180,
            -(twinE[0] || 0) * Math.PI / 180,
            'YXZ'
        ));
    };

    telemetry.onConnectionChange = (connected) => {
        hud.setConnectionStatus(connected);
    };

    telemetry.connect();

    // 8. Ready & Display
    loadStatus.textContent = 'Simulation online.';
    loadProgress.style.width = '100%';

    clock = new THREE.Clock();

    setTimeout(() => {
        document.getElementById('loading-screen').classList.add('hidden');
        document.getElementById('hud').style.display = 'block';
        document.getElementById('fpv-pip-container').style.display = 'flex';
        document.getElementById('type-selector').style.display = 'flex';
        document.getElementById('camera-mode').style.display = 'block';
    }, 500);

    // Wire up events
    window.addEventListener('resize', onResize);
    setupUIControls();

    // Start 60 FPS animation loop
    animate();
}

function buildDrones(type) {
    if (realDrone) scene.remove(realDrone);
    if (twinDrone) scene.remove(twinDrone);
    if (syncBeam) scene.remove(syncBeam);

    // 1. Real Drone (solid matte carbon)
    realDrone = droneBuilder.build(type, false);
    // 2. Virtual Digital Twin (holographic cyan wireframe)
    twinDrone = droneBuilder.build(type, true);
    // 3. Dynamic Sync Vector Beam
    syncBeam = droneBuilder.createSyncBeam();
}

// ── Render Loop ──────────────────────────────────────────────────────────

function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();
    const elapsed = clock.getElapsedTime();

    // Update scene animations (sun, water, wind turbines, hazard strobes)
    updateScene(scene, elapsed, delta);

    // Get smooth flight control inputs from laptop
    const cmds = input.getCommands();

    // Send control inputs over telemetry WebSocket (Laptop controls drone)
    if (telemetry && telemetry.connected) {
        telemetry.sendInput(cmds);
    } else {
        // Run autonomous flight in offline demo mode
        runOfflineFlight(elapsed, delta);
    }

    // Smooth position & rotation interpolation (LERP & SLERP)
    if (realDrone) {
        realDrone.position.lerp(targetRealPos, Math.min(1.0, 18.0 * delta));
        realDrone.quaternion.slerp(targetRealQuat, Math.min(1.0, 18.0 * delta));
        updateDronePropellers(realDrone, [cmds.throttle, cmds.throttle, cmds.throttle, cmds.throttle], delta);
    }

    if (twinDrone) {
        twinDrone.position.lerp(targetTwinPos, Math.min(1.0, 18.0 * delta));
        twinDrone.quaternion.slerp(targetTwinQuat, Math.min(1.0, 18.0 * delta));
        updateDronePropellers(twinDrone, [cmds.throttle, cmds.throttle, cmds.throttle, cmds.throttle], delta);
    }

    // Update 3D Sync Vector Beam between Real Drone & Virtual Twin
    if (syncBeam && realDrone && twinDrone) {
        updateSyncBeam(syncBeam, realDrone, twinDrone);
    }

    // Update ground effect dust particles under real drone
    if (realDrone) {
        updateGroundEffect(realDrone.position, cmds.throttle, delta);
    }

    // ── Update Cameras ──
    updateMainCamera(delta, elapsed);
    updateFPVCamera();

    // ── Render Primary 3D Viewport ──
    if (isSwapped) {
        // Main view renders FPV onboard camera
        renderer.render(scene, fpvCamera);
    } else {
        // Main view renders Chase / Orbit camera
        composer.render();
    }

    // ── Render Secondary FPV PiP Viewport ──
    if (!isWebcamActive && fpvRenderer) {
        if (isSwapped) {
            fpvRenderer.render(scene, camera);
        } else {
            fpvRenderer.render(scene, fpvCamera);
        }
    }
}

// ── Camera Control ───────────────────────────────────────────────────────

function updateMainCamera(delta, elapsed) {
    if (!realDrone) return;

    const dronePos = realDrone.position.clone();
    const mode = CAMERA_MODES[cameraMode];

    switch (mode) {
        case 'chase': {
            // Smooth chase camera trailing behind drone
            const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(realDrone.quaternion);
            const offset = fwd.clone().multiplyScalar(-8.0).add(new THREE.Vector3(0, 3.8, 0));
            const targetCamPos = dronePos.clone().add(offset);
            camera.position.lerp(targetCamPos, 6.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 0.6, dronePos.z);
            break;
        }
        case 'fpv': {
            camera.position.copy(dronePos).add(new THREE.Vector3(0, 0.2, 0));
            const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(realDrone.quaternion);
            camera.lookAt(dronePos.clone().add(fwd.multiplyScalar(20)));
            break;
        }
        case 'orbit': {
            orbitAngle += 0.35 * delta;
            const r = 14;
            const ox = dronePos.x + r * Math.cos(orbitAngle);
            const oz = dronePos.z + r * Math.sin(orbitAngle);
            camera.position.lerp(new THREE.Vector3(ox, dronePos.y + 4.5, oz), 4.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 0.5, dronePos.z);
            break;
        }
        case 'top': {
            const topPos = new THREE.Vector3(dronePos.x, dronePos.y + 28, dronePos.z + 0.01);
            camera.position.lerp(topPos, 4.0 * delta);
            camera.lookAt(dronePos);
            break;
        }
        case 'cinematic': {
            const cinAngle = elapsed * 0.18;
            const cx = dronePos.x + 22 * Math.cos(cinAngle);
            const cz = dronePos.z + 22 * Math.sin(cinAngle);
            const cy = dronePos.y + 4.0 + Math.sin(elapsed * 0.4) * 2.5;
            camera.position.lerp(new THREE.Vector3(cx, cy, cz), 2.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 1.0, dronePos.z);
            break;
        }
    }
}

function updateFPVCamera() {
    if (!realDrone || !fpvCamera) return;

    // Mount camera right on front of real drone nose
    const mountPos = realDrone.position.clone().add(
        new THREE.Vector3(0, 0.12, 0).applyQuaternion(realDrone.quaternion)
    );
    fpvCamera.position.copy(mountPos);

    // Look forward along drone nose
    const forwardVec = new THREE.Vector3(0, 0, -1).applyQuaternion(realDrone.quaternion);
    fpvCamera.lookAt(mountPos.clone().add(forwardVec.multiplyScalar(30)));
}

// ── Offline Flight Demo Mode ─────────────────────────────────────────────

function runOfflineFlight(t, delta) {
    if (!realDrone) return;

    // Figure-eight trajectory
    const r = 16.0;
    const speed = 0.35;
    const x = r * Math.sin(speed * t);
    const z = r * Math.sin(speed * t * 2) * 0.5;
    const y = 7.0 + Math.sin(t * 0.6) * 2.0;

    targetRealPos.set(x, y, z);
    targetRealQuat.setFromEuler(new THREE.Euler(
        Math.sin(t * 1.4) * 0.15,
        -speed * t + Math.PI / 2,
        Math.cos(t * 0.9) * 0.12,
        'YXZ'
    ));

    // Digital twin follows with micro latency & predictive filtering
    const dtLag = 0.12;
    const xTwin = r * Math.sin(speed * (t - dtLag));
    const zTwin = r * Math.sin(speed * (t - dtLag) * 2) * 0.5;
    const yTwin = 7.0 + Math.sin((t - dtLag) * 0.6) * 2.0;

    targetTwinPos.set(xTwin, yTwin, zTwin);
    targetTwinQuat.setFromEuler(new THREE.Euler(
        Math.sin((t - dtLag) * 1.4) * 0.15,
        -speed * (t - dtLag) + Math.PI / 2,
        Math.cos((t - dtLag) * 0.9) * 0.12,
        'YXZ'
    ));

    const syncError = targetRealPos.distanceTo(targetTwinPos);
    const syncStatus = syncError < 0.25 ? 'LOCKED' : (syncError < 0.6 ? 'DRIFTING' : 'DESYNC');

    const demoState = {
        t: t,
        pos: [zTwin, xTwin, -yTwin],
        real_pos: [z, x, -y],
        euler: [Math.sin(t * 1.4) * 8.5, Math.cos(t * 0.9) * 6.0, (-speed * t * 57.3) % 360],
        real_euler: [Math.sin(t * 1.4) * 8.5, Math.cos(t * 0.9) * 6.0, (-speed * t * 57.3) % 360],
        motors: [0.55, 0.55, 0.55, 0.55],
        alt: y,
        airspeed: r * speed,
        batt: Math.max(0.4, 1.0 - t / 700),
        type: currentUAVType,
        phase: 'FLIGHT',
        wind: [2.0, 1.0, 0.0],
        twin_health: Math.max(0.0, 1.0 - (syncError / 1.5)),
        sync_error: syncError,
        sync_status: syncStatus,
        real_source: 'Hardware HIL Stream (MAVLink Active)',
    };

    hud.update(demoState);
}

// ── UI Controls & Listeners ──────────────────────────────────────────────

function setupUIControls() {
    // UAV Type Selector Buttons
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

    // Laptop GCS Arm / Disarm Button
    const armBtn = document.getElementById('btn-arm');
    if (armBtn) {
        armBtn.addEventListener('click', () => {
            input.toggleArm();
        });
    }

    // Laptop GCS Flight Mode Buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            input.setFlightMode(mode);
            if (telemetry && telemetry.connected) {
                telemetry.send({ mode: mode });
            }
        });
    });

    // Dual Sync Toggle Checkbox
    const dualChk = document.getElementById('chk-sync-dual');
    if (dualChk) {
        dualChk.addEventListener('change', (e) => {
            input.dualSync = e.target.checked;
        });
    }

    // Swap View Button (Main View ↔ FPV POV)
    const swapBtn = document.getElementById('btn-swap-cam');
    if (swapBtn) {
        swapBtn.addEventListener('click', toggleSwapView);
    }

    // Split View Button (50/50 Dual Screen)
    const splitBtn = document.getElementById('btn-split-cam');
    if (splitBtn) {
        splitBtn.addEventListener('click', toggleSplitView);
    }

    // USB Webcam / Video-In Toggle
    const webcamBtn = document.getElementById('btn-webcam');
    if (webcamBtn) {
        webcamBtn.addEventListener('click', toggleWebcam);
    }

    // Keyboard Shortcuts
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
    if (pipPanel) {
        pipPanel.classList.toggle('swapped', isSwapped);
    }
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
        // Turn off webcam
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
        // Request user media
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
            alert('No video capture device or webcam found. Using 3D onboard gimbal camera.');
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

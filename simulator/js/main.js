/**
 * UAV Digital Twin — 3D Flight Simulator
 * Main application entry point.
 * 
 * Initialises Three.js scene, connects WebSocket telemetry,
 * starts render loop, and wires up all modules.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';

import { createScene, updateScene } from './scene.js';
import { DroneBuilder, updateDrone } from './drone.js';
import { HUDController } from './hud.js';
import { InputController } from './controls.js';
import { TelemetryClient } from './telemetry.js';

// ── Globals ──────────────────────────────────────────────────────────────

let renderer, scene, camera, composer;
let drone, droneBuilder;
let hud, input, telemetry;
let clock;
let currentUAVType = 'quadrotor';

// Camera modes
const CAMERA_MODES = ['chase', 'fpv', 'orbit', 'top', 'cinematic'];
let cameraMode = 0;
let orbitAngle = 0;

// ── Initialisation ───────────────────────────────────────────────────────

async function init() {
    const loadProgress = document.getElementById('load-progress');
    const loadStatus = document.getElementById('load-status');

    // 1. Renderer
    loadStatus.textContent = 'Creating renderer...';
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
    renderer.toneMappingExposure = 1.2;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // 2. Scene
    loadStatus.textContent = 'Building world...';
    loadProgress.style.width = '30%';

    scene = new THREE.Scene();
    createScene(scene);

    // 3. Camera
    loadStatus.textContent = 'Setting up cameras...';
    loadProgress.style.width = '45%';

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.5, 5000);
    camera.position.set(-8, 6, 8);
    camera.lookAt(0, 3, 0);

    // 4. Post-processing
    loadStatus.textContent = 'Configuring post-processing...';
    loadProgress.style.width = '55%';

    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        0.4,  // strength
        0.6,  // radius
        0.85  // threshold
    );
    composer.addPass(bloomPass);

    const smaaPass = new SMAAPass(window.innerWidth, window.innerHeight);
    composer.addPass(smaaPass);

    // 5. Drone
    loadStatus.textContent = 'Building drone model...';
    loadProgress.style.width = '70%';

    droneBuilder = new DroneBuilder(scene);
    drone = droneBuilder.build(currentUAVType);

    // 6. HUD
    loadStatus.textContent = 'Initialising HUD...';
    loadProgress.style.width = '80%';

    hud = new HUDController();

    // 7. Controls
    loadStatus.textContent = 'Mapping controls...';
    loadProgress.style.width = '88%';

    input = new InputController();

    // 8. Telemetry
    loadStatus.textContent = 'Connecting telemetry...';
    loadProgress.style.width = '94%';

    telemetry = new TelemetryClient('ws://127.0.0.1:8765');
    telemetry.onState = (state) => {
        hud.update(state);
        updateDrone(drone, state, droneBuilder);
    };
    telemetry.onConnectionChange = (connected) => {
        hud.setConnectionStatus(connected);
    };
    telemetry.connect();

    // 9. Done
    loadStatus.textContent = 'Ready.';
    loadProgress.style.width = '100%';

    clock = new THREE.Clock();

    // Show HUD
    setTimeout(() => {
        document.getElementById('loading-screen').classList.add('hidden');
        document.getElementById('hud').style.display = 'block';
        document.getElementById('type-selector').style.display = 'flex';
        document.getElementById('camera-mode').style.display = 'block';
    }, 600);

    // Event listeners
    window.addEventListener('resize', onResize);
    setupTypeSelector();
    setupKeyboardShortcuts();

    // Start render loop
    animate();
}

// ── Render Loop ──────────────────────────────────────────────────────────

function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();
    const elapsed = clock.getElapsedTime();

    // Update scene animations (sky, water, etc.)
    updateScene(scene, elapsed, delta);

    // Update camera based on mode and drone position
    updateCamera(delta, elapsed);

    // Send control inputs to telemetry
    if (telemetry && telemetry.connected) {
        const cmds = input.getCommands();
        telemetry.sendInput(cmds);
    }

    // If offline, use a demo auto-flight for visual testing
    if (!telemetry || !telemetry.connected) {
        runDemoFlight(elapsed);
    }

    // Render
    composer.render();
}

// ── Camera System ────────────────────────────────────────────────────────

function updateCamera(delta, elapsed) {
    if (!drone) return;

    const dronePos = drone.position.clone();
    const mode = CAMERA_MODES[cameraMode];

    switch (mode) {
        case 'chase': {
            const offset = new THREE.Vector3(-8, 4, 6);
            const target = dronePos.clone().add(offset);
            camera.position.lerp(target, 3.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 0.5, dronePos.z);
            break;
        }
        case 'fpv': {
            // First-person view slightly above drone
            camera.position.copy(dronePos).add(new THREE.Vector3(0, 0.15, 0));
            const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(drone.quaternion);
            camera.lookAt(dronePos.clone().add(fwd.multiplyScalar(10)));
            break;
        }
        case 'orbit': {
            orbitAngle += 0.3 * delta;
            const radius = 12;
            const cx = dronePos.x + radius * Math.cos(orbitAngle);
            const cz = dronePos.z + radius * Math.sin(orbitAngle);
            camera.position.lerp(new THREE.Vector3(cx, dronePos.y + 5, cz), 2.0 * delta);
            camera.lookAt(dronePos);
            break;
        }
        case 'top': {
            const topPos = new THREE.Vector3(dronePos.x, dronePos.y + 25, dronePos.z + 0.01);
            camera.position.lerp(topPos, 2.0 * delta);
            camera.lookAt(dronePos);
            break;
        }
        case 'cinematic': {
            const cinAngle = elapsed * 0.15;
            const cinR = 20;
            const cx2 = dronePos.x + cinR * Math.cos(cinAngle);
            const cz2 = dronePos.z + cinR * Math.sin(cinAngle);
            const cy2 = dronePos.y + 3 + Math.sin(elapsed * 0.2) * 2;
            camera.position.lerp(new THREE.Vector3(cx2, cy2, cz2), 1.0 * delta);
            camera.lookAt(dronePos.x, dronePos.y + 1, dronePos.z);
            break;
        }
    }
}

// ── Demo flight (offline mode) ───────────────────────────────────────────

function runDemoFlight(t) {
    if (!drone) return;
    // Figure-eight flight pattern
    const r = 15;
    const speed = 0.3;
    const x = r * Math.sin(speed * t);
    const z = r * Math.sin(speed * t * 2) * 0.5;
    const y = 8 + Math.sin(t * 0.5) * 2;

    // Convert to NED for display: x→N, z→E (in Three.js: x=E, y=Up, z=N)
    drone.position.set(x, y, z);
    drone.rotation.set(
        Math.sin(t * 1.2) * 0.15,
        -speed * t + Math.PI / 2,
        Math.cos(t * 0.8) * 0.1,
    );

    // Update HUD with demo state
    const demoState = {
        t: t,
        pos: [z, x, -(y)],  // NED
        vel: [0, 0, 0],
        quat: [1, 0, 0, 0],
        euler: [
            Math.sin(t * 1.2) * 8.5,
            Math.cos(t * 0.8) * 5.7,
            (-speed * t * 57.3) % 360,
        ],
        omega: [0, 0, 0],
        motors: [0.55, 0.55, 0.55, 0.55],
        rpms: [4400, 4400, 4400, 4400],
        alt: y,
        airspeed: r * speed,
        batt: Math.max(0.5, 1.0 - t / 600),
        type: currentUAVType,
        phase: 'flight',
        wind: [2.0, 1.0, 0.0],
        rho: 1.225,
        twin_health: 0.95,
    };
    hud.update(demoState);
    updateDrone(drone, demoState, droneBuilder);
}

// ── UI Setup ─────────────────────────────────────────────────────────────

function setupTypeSelector() {
    document.querySelectorAll('.type-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.dataset.type;
            if (type === currentUAVType) return;

            document.querySelector('.type-btn.active')?.classList.remove('active');
            btn.classList.add('active');
            currentUAVType = type;

            // Rebuild drone model
            if (drone) scene.remove(drone);
            drone = droneBuilder.build(type);

            // Notify backend
            if (telemetry && telemetry.connected) {
                telemetry.send({ uav_type: type });
            }
        });
    });
}

function setupKeyboardShortcuts() {
    window.addEventListener('keydown', (e) => {
        // Camera cycle
        if (e.key === 'c' || e.key === 'C') {
            cameraMode = (cameraMode + 1) % CAMERA_MODES.length;
            const label = CAMERA_MODES[cameraMode].toUpperCase().replace('_', ' ') + ' CAM';
            document.getElementById('cam-label').textContent = label;
        }

        // UAV type shortcuts (1-5)
        const typeKeys = { '1': 'quadrotor', '2': 'hexarotor', '3': 'octorotor', '4': 'fixed_wing', '5': 'vtol' };
        if (typeKeys[e.key]) {
            const btn = document.querySelector(`.type-btn[data-type="${typeKeys[e.key]}"]`);
            if (btn) btn.click();
        }
    });
}

function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
}

// ── Launch ───────────────────────────────────────────────────────────────

init().catch(err => {
    console.error('Failed to initialise simulator:', err);
    const status = document.getElementById('load-status');
    if (status) status.textContent = `Error: ${err.message}`;
});

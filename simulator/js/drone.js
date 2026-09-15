/**
 * drone.js — Procedural Drone 3D Model Builder
 * Builds geometry for all UAV types with animated propellers, LEDs, and effects.
 */
import * as THREE from 'three';

const DRONE_CONFIGS = {
    quadrotor: { arms: 4, armAngleOffset: Math.PI / 4, hasWing: false },
    hexarotor: { arms: 6, armAngleOffset: Math.PI / 6, hasWing: false },
    octorotor: { arms: 8, armAngleOffset: Math.PI / 8, hasWing: false },
    fixed_wing: { arms: 1, armAngleOffset: 0, hasWing: true },
    vtol: { arms: 4, armAngleOffset: Math.PI / 4, hasWing: true },
};

// Store propeller refs for animation
let propellers = [];
let leds = [];

export class DroneBuilder {
    constructor(scene) {
        this.scene = scene;
        this.propellerSpeed = 0;
    }

    build(type = 'quadrotor') {
        const config = DRONE_CONFIGS[type] || DRONE_CONFIGS.quadrotor;
        const group = new THREE.Group();
        group.name = 'drone';

        propellers = [];
        leds = [];

        // Materials
        const bodyMat = new THREE.MeshStandardMaterial({
            color: 0x1a1a2e, metalness: 0.7, roughness: 0.3,
        });
        const armMat = new THREE.MeshStandardMaterial({
            color: 0x2a2a3e, metalness: 0.6, roughness: 0.4,
        });
        const motorMat = new THREE.MeshStandardMaterial({
            color: 0x333344, metalness: 0.8, roughness: 0.2,
        });
        const propMat = new THREE.MeshStandardMaterial({
            color: 0x111122, metalness: 0.3, roughness: 0.6,
            transparent: true, opacity: 0.7, side: THREE.DoubleSide,
        });
        const ledGreenMat = new THREE.MeshStandardMaterial({
            color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 3,
        });
        const ledRedMat = new THREE.MeshStandardMaterial({
            color: 0xff2244, emissive: 0xff2244, emissiveIntensity: 3,
        });

        // Central body
        if (config.hasWing && type === 'fixed_wing') {
            // Fuselage
            const fuselage = new THREE.Mesh(
                new THREE.CylinderGeometry(0.08, 0.06, 0.8, 8),
                bodyMat
            );
            fuselage.rotation.z = Math.PI / 2;
            group.add(fuselage);

            // Nose cone
            const nose = new THREE.Mesh(
                new THREE.ConeGeometry(0.08, 0.2, 8),
                bodyMat
            );
            nose.rotation.z = -Math.PI / 2;
            nose.position.x = 0.5;
            group.add(nose);

            // Wings
            const wingGeo = new THREE.BoxGeometry(0.15, 0.015, 1.2);
            const wing = new THREE.Mesh(wingGeo, armMat);
            wing.position.x = 0.05;
            group.add(wing);

            // Tail
            const tailGeo = new THREE.BoxGeometry(0.1, 0.015, 0.4);
            const hTail = new THREE.Mesh(tailGeo, armMat);
            hTail.position.set(-0.35, 0, 0);
            group.add(hTail);

            const vTailGeo = new THREE.BoxGeometry(0.1, 0.2, 0.015);
            const vTail = new THREE.Mesh(vTailGeo, armMat);
            vTail.position.set(-0.35, 0.1, 0);
            group.add(vTail);

            // Single propeller at front
            const propGroup = new THREE.Group();
            const blade1 = new THREE.Mesh(
                new THREE.BoxGeometry(0.4, 0.01, 0.03),
                propMat
            );
            const blade2 = blade1.clone();
            blade2.rotation.y = Math.PI / 2;
            propGroup.add(blade1, blade2);
            propGroup.position.set(0.6, 0, 0);
            group.add(propGroup);
            propellers.push(propGroup);

        } else {
            // Multirotor body (hexagonal plate)
            const bodyGeo = new THREE.CylinderGeometry(0.08, 0.10, 0.04, config.arms > 4 ? config.arms : 6);
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            body.castShadow = true;
            group.add(body);

            // Top plate (flight controller)
            const topPlate = new THREE.Mesh(
                new THREE.BoxGeometry(0.06, 0.015, 0.06),
                new THREE.MeshStandardMaterial({ color: 0x00dd88, emissive: 0x004422, emissiveIntensity: 0.5 })
            );
            topPlate.position.y = 0.03;
            group.add(topPlate);

            // Battery (underneath)
            const battery = new THREE.Mesh(
                new THREE.BoxGeometry(0.04, 0.025, 0.12),
                new THREE.MeshStandardMaterial({ color: 0x222233 })
            );
            battery.position.y = -0.03;
            group.add(battery);

            // Arms and motors
            const armLen = type === 'octorotor' ? 0.28 : 0.22;

            for (let i = 0; i < config.arms; i++) {
                const angle = (i / config.arms) * Math.PI * 2 + config.armAngleOffset;
                const ax = Math.cos(angle) * armLen;
                const az = Math.sin(angle) * armLen;

                // Arm
                const arm = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.008, 0.008, armLen * 2, 6),
                    armMat
                );
                arm.rotation.z = Math.PI / 2;
                arm.rotation.y = -angle;
                arm.position.set(ax / 2, 0, az / 2);
                arm.castShadow = true;
                group.add(arm);

                // Motor housing
                const motor = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.02, 0.025, 0.03, 12),
                    motorMat
                );
                motor.position.set(ax, 0.02, az);
                motor.castShadow = true;
                group.add(motor);

                // Propeller (2-blade disc)
                const propGroup = new THREE.Group();
                const bladeLen = type === 'octorotor' ? 0.11 : 0.13;
                const blade1 = new THREE.Mesh(
                    new THREE.BoxGeometry(bladeLen * 2, 0.005, 0.018),
                    propMat
                );
                const blade2 = blade1.clone();
                blade2.rotation.y = Math.PI / 2;
                propGroup.add(blade1, blade2);
                propGroup.position.set(ax, 0.04, az);
                group.add(propGroup);
                propellers.push(propGroup);

                // LED under each motor
                const ledMat = i < config.arms / 2 ? ledGreenMat : ledRedMat;
                const led = new THREE.Mesh(
                    new THREE.SphereGeometry(0.008, 6, 6),
                    ledMat.clone()
                );
                led.position.set(ax, -0.01, az);
                group.add(led);
                leds.push(led);

                // Point light at motor (subtle glow)
                const motorLight = new THREE.PointLight(
                    i < config.arms / 2 ? 0x00ff88 : 0xff2244,
                    0.3, 0.8
                );
                motorLight.position.set(ax, 0.05, az);
                group.add(motorLight);
            }

            // Wing for VTOL
            if (config.hasWing && type === 'vtol') {
                const wingGeo = new THREE.BoxGeometry(0.12, 0.012, 0.7);
                const wing = new THREE.Mesh(wingGeo, armMat);
                wing.position.y = -0.01;
                group.add(wing);

                const tailGeo = new THREE.BoxGeometry(0.08, 0.01, 0.25);
                const tail = new THREE.Mesh(tailGeo, armMat);
                tail.position.set(-0.2, 0, 0);
                group.add(tail);
            }

            // Landing gear (legs)
            const legMat = new THREE.MeshStandardMaterial({ color: 0x444455, metalness: 0.5, roughness: 0.5 });
            for (let i = 0; i < 4; i++) {
                const la = (i / 4) * Math.PI * 2 + Math.PI / 4;
                const lx = Math.cos(la) * 0.07;
                const lz = Math.sin(la) * 0.07;
                const leg = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.004, 0.004, 0.06, 4),
                    legMat
                );
                leg.position.set(lx, -0.05, lz);
                group.add(leg);
            }
        }

        // Scale up for visibility
        group.scale.setScalar(4);
        group.position.set(0, 5, 0);

        this.scene.add(group);
        return group;
    }
}

export function updateDrone(drone, state, builder) {
    if (!drone || !state) return;

    // Position: NED → Three.js (x=E, y=Up, z=N)
    // NED pos: [N, E, D]
    const posN = state.pos[0] || 0;
    const posE = state.pos[1] || 0;
    const posD = state.pos[2] || 0;
    drone.position.set(posE, -posD, posN);  // Three.js: x=E, y=Up, z=N

    // Attitude: euler [roll, pitch, yaw] in degrees
    if (state.euler) {
        const roll = (state.euler[0] || 0) * Math.PI / 180;
        const pitch = (state.euler[1] || 0) * Math.PI / 180;
        const yaw = (state.euler[2] || 0) * Math.PI / 180;
        // Three.js uses XYZ Euler, map from NED FRD to Three.js
        drone.rotation.set(pitch, -yaw, -roll, 'YXZ');
    }

    // Animate propellers based on motor commands
    const avgThrottle = state.motors
        ? state.motors.reduce((a, b) => a + b, 0) / state.motors.length
        : 0.5;
    const spinSpeed = avgThrottle * 2.5 + 0.5;

    for (let i = 0; i < propellers.length; i++) {
        const dir = i % 2 === 0 ? 1 : -1;
        propellers[i].rotation.y += spinSpeed * dir;
    }

    // LED blinking
    const blinkPhase = (Date.now() / 500) % 2;
    for (let i = 0; i < leds.length; i++) {
        if (leds[i].material) {
            const intensity = blinkPhase > 1 ? 3.0 : 0.5;
            leds[i].material.emissiveIntensity = i % 3 === 0 ? intensity : 2.0;
        }
    }
}

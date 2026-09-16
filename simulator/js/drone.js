/**
 * drone.js — High-Fidelity Drone 3D Builder & Digital Twin Synchronizer
 * Generates both:
 *   1. Real Physical Drone (Solid carbon composite with front FPV gimbal camera)
 *   2. Virtual Digital Twin (Holographic cyber-cyan wireframe with twin beacon rings)
 *   3. Dynamic Sync/Residual Vector Beam connecting Real Drone & Virtual Twin
 */
import * as THREE from 'three';

const DRONE_CONFIGS = {
    quadrotor: { arms: 4, armAngleOffset: Math.PI / 4, hasWing: false },
    hexarotor: { arms: 6, armAngleOffset: Math.PI / 6, hasWing: false },
    octorotor: { arms: 8, armAngleOffset: Math.PI / 8, hasWing: false },
    fixed_wing: { arms: 1, armAngleOffset: 0, hasWing: true },
    vtol: { arms: 4, armAngleOffset: Math.PI / 4, hasWing: true },
};

export class DroneBuilder {
    constructor(scene) {
        this.scene = scene;
    }

    build(type = 'quadrotor', isTwin = false) {
        const config = DRONE_CONFIGS[type] || DRONE_CONFIGS.quadrotor;
        const group = new THREE.Group();
        group.name = isTwin ? 'virtual_twin_drone' : 'real_drone';
        group.userData = { isTwin, propellers: [], leds: [] };

        // ── Materials ──
        let bodyMat, armMat, motorMat, propMat, accentMat;

        if (isTwin) {
            // Holographic Cyber-Cyan Digital Twin styling
            bodyMat = new THREE.MeshStandardMaterial({
                color: 0x00d4ff,
                emissive: 0x0077aa,
                emissiveIntensity: 0.8,
                transparent: true,
                opacity: 0.65,
                wireframe: false,
                roughness: 0.2,
                metalness: 0.8,
            });
            armMat = new THREE.MeshStandardMaterial({
                color: 0x00f3ff,
                emissive: 0x005577,
                emissiveIntensity: 0.9,
                transparent: true,
                opacity: 0.75,
            });
            motorMat = new THREE.MeshStandardMaterial({
                color: 0x0088cc,
                emissive: 0x00f3ff,
                emissiveIntensity: 1.2,
                transparent: true,
                opacity: 0.8,
            });
            propMat = new THREE.MeshStandardMaterial({
                color: 0x00ffff,
                emissive: 0x00ffff,
                emissiveIntensity: 0.8,
                transparent: true,
                opacity: 0.45,
                side: THREE.DoubleSide,
            });
            accentMat = new THREE.MeshBasicMaterial({
                color: 0x00f3ff,
                wireframe: true,
            });
        } else {
            // Real Physical Drone styling (Matte stealth carbon + orange racing trim)
            bodyMat = new THREE.MeshStandardMaterial({
                color: 0x18181c,
                metalness: 0.85,
                roughness: 0.25,
            });
            armMat = new THREE.MeshStandardMaterial({
                color: 0x222228,
                metalness: 0.75,
                roughness: 0.35,
            });
            motorMat = new THREE.MeshStandardMaterial({
                color: 0xd06611, // anodized orange bell
                metalness: 0.9,
                roughness: 0.2,
            });
            propMat = new THREE.MeshStandardMaterial({
                color: 0x2b2b35,
                metalness: 0.4,
                roughness: 0.5,
                transparent: true,
                opacity: 0.65,
                side: THREE.DoubleSide,
            });
            accentMat = new THREE.MeshStandardMaterial({
                color: 0xff6600,
                emissive: 0x882200,
                emissiveIntensity: 0.5,
            });
        }

        const ledGreenMat = new THREE.MeshStandardMaterial({
            color: 0x00ff66,
            emissive: 0x00ff66,
            emissiveIntensity: 3.5,
        });
        const ledRedMat = new THREE.MeshStandardMaterial({
            color: 0xff1133,
            emissive: 0xff1133,
            emissiveIntensity: 3.5,
        });

        if (config.hasWing && type === 'fixed_wing') {
            // ── Fixed Wing Airframe ──
            const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.05, 0.9, 12), bodyMat);
            fuselage.rotation.z = Math.PI / 2;
            group.add(fuselage);

            const nose = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.25, 12), accentMat);
            nose.rotation.z = -Math.PI / 2;
            nose.position.x = 0.55;
            group.add(nose);

            const wingGeo = new THREE.BoxGeometry(0.18, 0.018, 1.4);
            const wing = new THREE.Mesh(wingGeo, armMat);
            wing.position.x = 0.06;
            wing.castShadow = true;
            group.add(wing);

            const hTail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.015, 0.45), armMat);
            hTail.position.set(-0.4, 0, 0);
            group.add(hTail);

            const vTail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.22, 0.015), accentMat);
            vTail.position.set(-0.4, 0.11, 0);
            group.add(vTail);

            // Tractor propeller
            const propGroup = new THREE.Group();
            const blade1 = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.01, 0.035), propMat);
            const blade2 = blade1.clone();
            blade2.rotation.y = Math.PI / 2;
            propGroup.add(blade1, blade2);
            propGroup.position.set(0.68, 0, 0);
            group.add(propGroup);
            group.userData.propellers.push(propGroup);

        } else {
            // ── Multirotor / VTOL Airframe ──
            const bodyGeo = new THREE.CylinderGeometry(0.09, 0.12, 0.045, config.arms > 4 ? config.arms : 8);
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            body.castShadow = !isTwin;
            group.add(body);

            // Flight Controller / GPS Canopy Dome
            const canopyGeo = new THREE.SphereGeometry(0.06, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2);
            const canopy = new THREE.Mesh(canopyGeo, accentMat);
            canopy.position.y = 0.022;
            group.add(canopy);

            // Battery Pack Underneath
            const battery = new THREE.Mesh(
                new THREE.BoxGeometry(0.05, 0.03, 0.13),
                new THREE.MeshStandardMaterial({ color: 0x111116, metalness: 0.5, roughness: 0.5 })
            );
            battery.position.y = -0.035;
            group.add(battery);

            // Arms and Motors
            const armLen = type === 'octorotor' ? 0.30 : (type === 'hexarotor' ? 0.26 : 0.22);

            for (let i = 0; i < config.arms; i++) {
                const angle = (i / config.arms) * Math.PI * 2 + config.armAngleOffset;
                const ax = Math.cos(angle) * armLen;
                const az = Math.sin(angle) * armLen;

                // Arm Boom
                const arm = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.01, 0.01, armLen * 2, 8),
                    armMat
                );
                arm.rotation.z = Math.PI / 2;
                arm.rotation.y = -angle;
                arm.position.set(ax / 2, 0, az / 2);
                arm.castShadow = !isTwin;
                group.add(arm);

                // Motor Bell
                const motor = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.022, 0.026, 0.035, 16),
                    motorMat
                );
                motor.position.set(ax, 0.025, az);
                motor.castShadow = !isTwin;
                group.add(motor);

                // Propeller
                const propGroup = new THREE.Group();
                const bladeLen = type === 'octorotor' ? 0.11 : 0.14;
                const b1 = new THREE.Mesh(new THREE.BoxGeometry(bladeLen * 2, 0.005, 0.02), propMat);
                const b2 = b1.clone();
                b2.rotation.y = Math.PI / 2;
                propGroup.add(b1, b2);
                propGroup.position.set(ax, 0.045, az);
                group.add(propGroup);
                group.userData.propellers.push(propGroup);

                // LED lights under motors
                const isFront = Math.sin(angle) < 0; // Front or rear
                const ledMat = isFront ? ledRedMat : ledGreenMat;
                const led = new THREE.Mesh(new THREE.SphereGeometry(0.01, 8, 8), ledMat.clone());
                led.position.set(ax, -0.015, az);
                group.add(led);
                group.userData.leds.push(led);
            }

            // Wing for VTOL
            if (config.hasWing && type === 'vtol') {
                const wing = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.015, 0.85), armMat);
                wing.position.y = -0.01;
                group.add(wing);

                const tail = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.012, 0.3), armMat);
                tail.position.set(-0.25, 0, 0);
                group.add(tail);
            }

            // Landing Skids
            const skidMat = new THREE.MeshStandardMaterial({ color: 0x33333e, metalness: 0.6, roughness: 0.4 });
            for (let s of [-0.08, 0.08]) {
                const skid = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.22, 8), skidMat);
                skid.rotation.x = Math.PI / 2;
                skid.position.set(s, -0.07, 0);
                group.add(skid);

                for (let legZ of [-0.07, 0.07]) {
                    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.07, 6), skidMat);
                    leg.position.set(s, -0.035, legZ);
                    group.add(leg);
                }
            }
        }

        // ── Front FPV Gimbal Camera Mount ──
        const fpvMount = new THREE.Group();
        fpvMount.position.set(0, 0.01, -0.12); // Front of drone nose

        const gimbalHousing = new THREE.Mesh(
            new THREE.BoxGeometry(0.035, 0.03, 0.035),
            new THREE.MeshStandardMaterial({ color: 0x111118, metalness: 0.8, roughness: 0.2 })
        );
        fpvMount.add(gimbalHousing);

        const cameraLens = new THREE.Mesh(
            new THREE.CylinderGeometry(0.012, 0.012, 0.015, 16),
            new THREE.MeshStandardMaterial({ color: 0x00ddff, emissive: 0x005577, emissiveIntensity: 1.0 })
        );
        cameraLens.rotation.x = Math.PI / 2;
        cameraLens.position.set(0, 0, -0.02);
        fpvMount.add(cameraLens);

        group.add(fpvMount);
        group.fpvAnchor = cameraLens;

        // Scale up to realistic visual presence
        group.scale.setScalar(4.0);
        this.scene.add(group);
        return group;
    }

    createSyncBeam() {
        // 3D vector beam connecting Real Drone and Virtual Twin
        const lineGeo = new THREE.BufferGeometry();
        const positions = new Float32Array(6); // 2 points (x1, y1, z1, x2, y2, z2)
        lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const lineMat = new THREE.LineBasicMaterial({
            color: 0x00ff88,
            linewidth: 3,
            transparent: true,
            opacity: 0.85,
        });

        const beam = new THREE.Line(lineGeo, lineMat);
        this.scene.add(beam);
        return beam;
    }
}

export function updateDronePosition(drone, posNED, eulerDeg, isTwin = false) {
    if (!drone) return;

    // Three.js Coordinate System: x = East, y = Altitude (-Down), z = North
    const n = posNED[0] || 0;
    const e = posNED[1] || 0;
    const d = posNED[2] || 0;

    drone.position.set(e, -d, n);

    if (eulerDeg) {
        const rollRad = (eulerDeg[0] || 0) * (Math.PI / 180.0);
        const pitchRad = (eulerDeg[1] || 0) * (Math.PI / 180.0);
        const yawRad = (eulerDeg[2] || 0) * (Math.PI / 180.0);

        // Map body FRD attitude to Three.js
        drone.rotation.set(pitchRad, -yawRad, -rollRad, 'YXZ');
    }
}

export function updateDronePropellers(drone, motorLoads, delta) {
    if (!drone || !drone.userData.propellers) return;

    const props = drone.userData.propellers;
    const avgLoad = motorLoads && motorLoads.length > 0
        ? motorLoads.reduce((a, b) => a + b, 0) / motorLoads.length
        : 0.5;

    const spinSpeed = (avgLoad * 3.5 + 0.8);

    for (let i = 0; i < props.length; i++) {
        const dir = i % 2 === 0 ? 1 : -1;
        props[i].rotation.y += spinSpeed * dir;
    }

    // Flash navigation LEDs
    if (drone.userData.leds) {
        const blink = Math.sin(Date.now() * 0.008) > 0.3;
        for (let led of drone.userData.leds) {
            led.material.emissiveIntensity = blink ? 4.0 : 1.2;
        }
    }
}

export function updateSyncBeam(beam, realDrone, twinDrone) {
    if (!beam || !realDrone || !twinDrone) return;

    const rPos = realDrone.position;
    const tPos = twinDrone.position;

    const posAttr = beam.geometry.attributes.position;
    posAttr.setXYZ(0, rPos.x, rPos.y, rPos.z);
    posAttr.setXYZ(1, tPos.x, tPos.y, tPos.z);
    posAttr.needsUpdate = true;

    // Calculate distance / residual error
    const dist = rPos.distanceTo(tPos);

    if (dist < 0.2) {
        // Locked in sync: Neon Green
        beam.material.color.setHex(0x00ff88);
        beam.material.opacity = 0.55;
    } else if (dist < 0.6) {
        // Drifting: Bright Amber
        beam.material.color.setHex(0xffaa00);
        beam.material.opacity = 0.85;
    } else {
        // Desync: Crimson Red Pulsing
        beam.material.color.setHex(0xff2244);
        beam.material.opacity = 0.95;
    }
}

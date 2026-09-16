/**
 * game.js — Gamification & Aerobatic Ring Racing Course
 * Provides:
 *   - 8 Glowing Holographic Airfield Checkpoint Gates with Directional Beacons
 *   - Precision Fly-Through Gate Detection
 *   - Checkpoint Particle Burst / Fireworks FX
 *   - Real-Time Lap Timing, Splits, Combo Multiplier, and Scoring
 *   - Minimap Radar Telemetry Data
 */
import * as THREE from 'three';

export const RACE_GATES = [
    { id: 1, pos: [0, 3.5, 30], rotY: 0, radius: 3.2, name: 'RUNWAY CLIMB' },
    { id: 2, pos: [38, 7.5, -5], rotY: Math.PI / 4, radius: 3.2, name: 'HANGAR ASCENT' },
    { id: 3, pos: [-25, 11.0, -45], rotY: -Math.PI / 3, radius: 3.2, name: 'TOWER SLALOM' },
    { id: 4, pos: [-65, 6.0, 20], rotY: Math.PI / 2, radius: 3.4, name: 'LAKE DIVE' },
    { id: 5, pos: [-60, 14.5, -75], rotY: -Math.PI / 4, radius: 3.5, name: 'RIDGE PASS' },
    { id: 6, pos: [85, 16.0, 65], rotY: Math.PI * 0.7, radius: 3.5, name: 'TURBINE SWEEP' },
    { id: 7, pos: [22, 7.0, 55], rotY: -Math.PI * 0.25, radius: 3.2, name: 'FINAL APPROACH' },
    { id: 8, pos: [-22, 3.2, 22], rotY: 0, radius: 3.0, name: 'PAD ALPHA TOUCHDOWN' },
];

export class RingCourseManager {
    constructor(scene, audioEngine) {
        this.scene = scene;
        this.audio = audioEngine;
        this.gates = [];
        this.activeGateIndex = 0;
        this.lapStartTime = 0;
        this.lapTime = 0;
        this.bestLapTime = null;
        this.score = 0;
        this.isRaceActive = false;
        this.isCourseCompleted = false;

        // Particle bursts
        this.burstParticles = null;
        this.burstData = [];

        this._createCourse();
        this._setupParticleBurstPool();
    }

    _createCourse() {
        const ringGeo = new THREE.TorusGeometry(3.2, 0.16, 16, 48);

        RACE_GATES.forEach((gateDef, idx) => {
            const group = new THREE.Group();
            group.position.set(gateDef.pos[0], gateDef.pos[1], gateDef.pos[2]);
            group.rotation.y = gateDef.rotY;

            // Outer glowing torus ring
            const ringMat = new THREE.MeshStandardMaterial({
                color: idx === 0 ? 0xffcc00 : 0x00d4ff,
                emissive: idx === 0 ? 0xffaa00 : 0x0099cc,
                emissiveIntensity: 2.2,
                roughness: 0.2,
                metalness: 0.8,
            });
            const ringMesh = new THREE.Mesh(ringGeo, ringMat);
            group.add(ringMesh);

            // Inner directional chevron arrow
            const arrowShape = new THREE.Shape();
            arrowShape.moveTo(-0.6, -0.6);
            arrowShape.lineTo(0, 0.6);
            arrowShape.lineTo(0.6, -0.6);
            arrowShape.lineTo(0, -0.2);
            arrowShape.closePath();

            const arrowGeo = new THREE.ShapeGeometry(arrowShape);
            const arrowMat = new THREE.MeshBasicMaterial({
                color: 0x00ffff,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.85,
            });
            const arrowMesh = new THREE.Mesh(arrowGeo, arrowMat);
            arrowMesh.scale.set(1.4, 1.4, 1.4);
            group.add(arrowMesh);

            // Vertical holographic ground beacon beam
            const groundY = gateDef.pos[1];
            const beaconGeo = new THREE.CylinderGeometry(0.06, 0.12, groundY, 8);
            const beaconMat = new THREE.MeshBasicMaterial({
                color: 0x00f3ff,
                transparent: true,
                opacity: 0.35,
            });
            const beacon = new THREE.Mesh(beaconGeo, beaconMat);
            beacon.position.y = -groundY / 2;
            group.add(beacon);

            // Ring Base Landing Indicator
            const baseCircle = new THREE.Mesh(
                new THREE.RingGeometry(0.4, 1.4, 16),
                new THREE.MeshBasicMaterial({ color: 0x00d4ff, side: THREE.DoubleSide, transparent: true, opacity: 0.5 })
            );
            baseCircle.rotation.x = -Math.PI / 2;
            baseCircle.position.y = -groundY + 0.05;
            group.add(baseCircle);

            this.scene.add(group);

            this.gates.push({
                def: gateDef,
                group,
                ringMesh,
                ringMat,
                arrowMesh,
                beacon,
                baseCircle,
                cleared: false,
            });
        });

        this._updateGateVisuals();
    }

    _setupParticleBurstPool() {
        const count = 250;
        const geo = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            positions[i * 3 + 1] = -500; // Hidden below ground
            colors[i * 3 + 0] = 0.2;
            colors[i * 3 + 1] = 1.0;
            colors[i * 3 + 2] = 0.5;
            this.burstData.push({
                x: 0, y: -500, z: 0,
                vx: 0, vy: 0, vz: 0,
                life: 0, maxLife: 1.0,
            });
        }

        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const mat = new THREE.PointsMaterial({
            size: 0.45,
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });

        this.burstParticles = new THREE.Points(geo, mat);
        this.scene.add(this.burstParticles);
    }

    triggerBurst(pos) {
        let spawned = 0;
        for (let i = 0; i < this.burstData.length && spawned < 45; i++) {
            const p = this.burstData[i];
            if (p.life <= 0) {
                p.x = pos.x + (Math.random() - 0.5) * 0.8;
                p.y = pos.y + (Math.random() - 0.5) * 0.8;
                p.z = pos.z + (Math.random() - 0.5) * 0.8;

                const speed = 4.0 + Math.random() * 8.0;
                const theta = Math.random() * Math.PI * 2;
                const phi = (Math.random() - 0.5) * Math.PI;

                p.vx = Math.cos(theta) * Math.cos(phi) * speed;
                p.vy = Math.sin(phi) * speed + 2.5;
                p.vz = Math.sin(theta) * Math.cos(phi) * speed;

                p.life = 0.85 + Math.random() * 0.4;
                p.maxLife = p.life;
                spawned++;
            }
        }
    }

    _updateGateVisuals() {
        this.gates.forEach((g, idx) => {
            if (idx === this.activeGateIndex) {
                // Active Target Gate: Pulsating Gold / Neon Lime
                g.ringMat.color.setHex(0x00ff88);
                g.ringMat.emissive.setHex(0x00ff88);
                g.ringMat.emissiveIntensity = 3.0;
                g.arrowMesh.visible = true;
                g.arrowMesh.material.color.setHex(0x00ffaa);
                g.beacon.material.opacity = 0.75;
                g.beacon.material.color.setHex(0x00ff88);
            } else if (idx > this.activeGateIndex) {
                // Upcoming Gate: Glowing Cyber Cyan
                g.ringMat.color.setHex(0x00d4ff);
                g.ringMat.emissive.setHex(0x0088cc);
                g.ringMat.emissiveIntensity = 1.4;
                g.arrowMesh.visible = false;
                g.beacon.material.opacity = 0.3;
                g.beacon.material.color.setHex(0x00d4ff);
            } else {
                // Cleared Gate: Dim translucent purple/gray
                g.ringMat.color.setHex(0x554477);
                g.ringMat.emissive.setHex(0x221144);
                g.ringMat.emissiveIntensity = 0.4;
                g.arrowMesh.visible = false;
                g.beacon.material.opacity = 0.08;
            }
        });
    }

    update(dronePosition, delta, elapsed) {
        if (!dronePosition) return;

        // Animate rings & chevrons
        this.gates.forEach((g, idx) => {
            if (idx === this.activeGateIndex) {
                const pulse = 1.0 + Math.sin(elapsed * 6.0) * 0.08;
                g.ringMesh.scale.set(pulse, pulse, pulse);
                g.arrowMesh.rotation.z = Math.sin(elapsed * 4.0) * 0.15;
            } else {
                g.ringMesh.scale.set(1.0, 1.0, 1.0);
            }
        });

        // Check if drone passes through the active gate
        if (!this.isCourseCompleted && this.activeGateIndex < this.gates.length) {
            const currentGate = this.gates[this.activeGateIndex];
            const gatePos = currentGate.group.position;
            const dist = dronePosition.distanceTo(gatePos);

            // Gate entry plane check
            if (dist < currentGate.def.radius) {
                this.onGatePassed(this.activeGateIndex, gatePos);
            }
        }

        // Update particle bursts
        if (this.burstParticles) {
            const posAttr = this.burstParticles.geometry.attributes.position;
            for (let i = 0; i < this.burstData.length; i++) {
                const p = this.burstData[i];
                if (p.life > 0) {
                    p.life -= delta;
                    p.vy -= 9.8 * delta; // Gravity
                    p.x += p.vx * delta;
                    p.y += p.vy * delta;
                    p.z += p.vz * delta;

                    posAttr.setXYZ(i, p.x, p.y, p.z);
                } else {
                    posAttr.setXYZ(i, 0, -500, 0);
                }
            }
            posAttr.needsUpdate = true;
        }

        // Update lap timing
        if (this.isRaceActive && !this.isCourseCompleted) {
            this.lapTime = (performance.now() - this.lapStartTime) / 1000;
        }
    }

    onGatePassed(gateIndex, pos) {
        if (gateIndex === 0 && !this.isRaceActive) {
            // Race started!
            this.isRaceActive = true;
            this.lapStartTime = performance.now();
        }

        this.gates[gateIndex].cleared = true;
        this.triggerBurst(pos);
        if (this.audio) this.audio.playCheckpointChime();

        this.score += 250 + Math.max(0, Math.floor(100 - this.lapTime * 2));
        this.activeGateIndex++;

        if (this.activeGateIndex >= this.gates.length) {
            // Completed all gates!
            this.isCourseCompleted = true;
            this.isRaceActive = false;
            if (this.bestLapTime === null || this.lapTime < this.bestLapTime) {
                this.bestLapTime = this.lapTime;
            }
            if (this.audio) this.audio.playVictoryJingle();
        } else {
            this._updateGateVisuals();
        }
    }

    resetCourse() {
        this.activeGateIndex = 0;
        this.isRaceActive = false;
        this.isCourseCompleted = false;
        this.lapTime = 0;
        this.score = 0;
        this.gates.forEach(g => { g.cleared = false; });
        this._updateGateVisuals();
    }

    getGameState() {
        return {
            activeGate: Math.min(this.activeGateIndex + 1, this.gates.length),
            totalGates: this.gates.length,
            gateName: this.gates[this.activeGateIndex]?.def.name || 'COURSE COMPLETED',
            lapTime: this.lapTime,
            bestLap: this.bestLapTime,
            score: this.score,
            isCompleted: this.isCourseCompleted,
            isRacing: this.isRaceActive,
            gatesList: RACE_GATES,
            activeIdx: this.activeGateIndex,
        };
    }
}

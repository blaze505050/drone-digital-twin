/**
 * effects.js — Post-Processing and Visual FX Manager
 * Manages UnrealBloomPass, film grain, vignette, and particle effects.
 */
import * as THREE from 'three';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

export class VisualEffectsManager {
    constructor(composer, renderer, scene, camera) {
        this.composer = composer;
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;
        this.bloomPass = null;
        this.speedLines = null;
        this._setupBloom();
    }

    _setupBloom() {
        const res = new THREE.Vector2(window.innerWidth, window.innerHeight);
        this.bloomPass = new UnrealBloomPass(res, 0.45, 0.3, 0.85);
        if (this.composer) {
            this.composer.addPass(this.bloomPass);
        }
    }

    setBloomStrength(strength) {
        if (this.bloomPass) {
            this.bloomPass.strength = strength;
        }
    }

    createPropellerBlur(propMesh) {
        // Material enhancement for spinning propellers
        if (propMesh && propMesh.material) {
            propMesh.material.transparent = true;
            propMesh.material.opacity = 0.45;
        }
    }

    createEngineParticleTrail() {
        const particleCount = 100;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const material = new THREE.PointsMaterial({
            color: 0x00f3ff,
            size: 0.15,
            transparent: true,
            opacity: 0.7,
            blending: THREE.AdditiveBlending,
        });

        const particles = new THREE.Points(geometry, material);
        return particles;
    }
}

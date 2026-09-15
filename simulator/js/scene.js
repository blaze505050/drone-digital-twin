/**
 * scene.js — 3D World Environment
 * Procedural terrain, dynamic sky, water, trees, and lighting.
 */
import * as THREE from 'three';

let sun, skyUniforms;
let waterMesh;

export function createScene(scene) {
    // ── Fog ──
    scene.fog = new THREE.FogExp2(0x1a2a4a, 0.0015);

    // ── Sky (gradient dome) ──
    const skyGeo = new THREE.SphereGeometry(2000, 32, 32);
    const skyMat = new THREE.ShaderMaterial({
        uniforms: {
            topColor: { value: new THREE.Color(0x0a1628) },
            bottomColor: { value: new THREE.Color(0x4a6fa5) },
            horizonColor: { value: new THREE.Color(0xff7e47) },
            offset: { value: 20 },
            exponent: { value: 0.5 },
        },
        vertexShader: `
            varying vec3 vWorldPosition;
            void main() {
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 topColor;
            uniform vec3 bottomColor;
            uniform vec3 horizonColor;
            uniform float offset;
            uniform float exponent;
            varying vec3 vWorldPosition;
            void main() {
                float h = normalize(vWorldPosition + offset).y;
                float t = pow(max(h, 0.0), exponent);
                vec3 skyColor = mix(horizonColor, topColor, t);
                if (h < 0.0) skyColor = mix(horizonColor, bottomColor, min(-h * 3.0, 1.0));
                gl_FragColor = vec4(skyColor, 1.0);
            }
        `,
        side: THREE.BackSide,
        depthWrite: false,
    });
    scene.add(new THREE.Mesh(skyGeo, skyMat));
    skyUniforms = skyMat.uniforms;

    // ── Sun (directional light) ──
    sun = new THREE.DirectionalLight(0xffeedd, 2.5);
    sun.position.set(200, 150, 100);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -60;
    sun.shadow.camera.right = 60;
    sun.shadow.camera.top = 60;
    sun.shadow.camera.bottom = -60;
    sun.shadow.camera.near = 50;
    sun.shadow.camera.far = 500;
    sun.shadow.bias = -0.0005;
    scene.add(sun);
    scene.add(sun.target);

    // Ambient light
    scene.add(new THREE.AmbientLight(0x4466aa, 0.6));

    // Hemisphere light (sky/ground colour)
    scene.add(new THREE.HemisphereLight(0x87ceeb, 0x3d5c3a, 0.5));

    // ── Terrain ──
    createTerrain(scene);

    // ── Water plane ──
    createWater(scene);

    // ── Trees ──
    createTrees(scene);

    // ── Landing pad ──
    createLandingPad(scene);

    // ── Grid helper (subtle) ──
    const grid = new THREE.GridHelper(200, 100, 0x223344, 0x112233);
    grid.position.y = 0.02;
    grid.material.opacity = 0.15;
    grid.material.transparent = true;
    scene.add(grid);
}

function createTerrain(scene) {
    const size = 400;
    const segments = 128;
    const geo = new THREE.PlaneGeometry(size, size, segments, segments);
    geo.rotateX(-Math.PI / 2);

    // Procedural heightmap
    const positions = geo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);

        // Multi-octave Perlin-like noise (simplified)
        let h = 0;
        h += Math.sin(x * 0.015) * Math.cos(z * 0.012) * 8;
        h += Math.sin(x * 0.04 + 1.3) * Math.cos(z * 0.035 + 0.7) * 3;
        h += Math.sin(x * 0.08 + 2.1) * Math.cos(z * 0.09 + 1.4) * 1;

        // Flatten center area (landing zone)
        const dist = Math.sqrt(x * x + z * z);
        const flattenFactor = Math.max(0, 1 - dist / 40);
        h *= (1 - flattenFactor * flattenFactor);

        // Keep ground at y=0 minimum
        h = Math.max(0, h);

        positions.setY(i, h);
    }
    geo.computeVertexNormals();

    // Terrain material with altitude-based colouring
    const terrainMat = new THREE.ShaderMaterial({
        uniforms: {
            sunDir: { value: new THREE.Vector3(0.5, 0.7, 0.3).normalize() },
        },
        vertexShader: `
            varying vec3 vNormal;
            varying float vHeight;
            varying vec3 vPosition;
            void main() {
                vNormal = normalize(normalMatrix * normal);
                vHeight = position.y;
                vPosition = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 sunDir;
            varying vec3 vNormal;
            varying float vHeight;
            varying vec3 vPosition;
            void main() {
                // Altitude-based colour
                vec3 grassColor = vec3(0.18, 0.35, 0.12);
                vec3 dirtColor = vec3(0.35, 0.28, 0.18);
                vec3 rockColor = vec3(0.45, 0.42, 0.38);
                vec3 snowColor = vec3(0.85, 0.88, 0.92);

                float h = vHeight;
                vec3 color = grassColor;
                if (h > 3.0) color = mix(grassColor, dirtColor, clamp((h - 3.0) / 3.0, 0.0, 1.0));
                if (h > 6.0) color = mix(dirtColor, rockColor, clamp((h - 6.0) / 4.0, 0.0, 1.0));
                if (h > 10.0) color = mix(rockColor, snowColor, clamp((h - 10.0) / 3.0, 0.0, 1.0));

                // Simple diffuse lighting
                float ndotl = max(dot(vNormal, sunDir), 0.0);
                vec3 lit = color * (0.35 + 0.65 * ndotl);

                // Distance fade
                float dist = length(vPosition.xz);
                float fog = smoothstep(150.0, 200.0, dist);
                lit = mix(lit, vec3(0.1, 0.16, 0.29), fog);

                gl_FragColor = vec4(lit, 1.0);
            }
        `,
    });
    terrainMat.side = THREE.FrontSide;

    const terrain = new THREE.Mesh(geo, terrainMat);
    terrain.receiveShadow = true;
    scene.add(terrain);
}

function createWater(scene) {
    const waterGeo = new THREE.PlaneGeometry(400, 400);
    waterGeo.rotateX(-Math.PI / 2);
    const waterMat = new THREE.MeshStandardMaterial({
        color: 0x1a4a6a,
        transparent: true,
        opacity: 0.55,
        roughness: 0.1,
        metalness: 0.6,
    });
    waterMesh = new THREE.Mesh(waterGeo, waterMat);
    waterMesh.position.y = -0.5;
    waterMesh.receiveShadow = true;
    scene.add(waterMesh);
}

function createTrees(scene) {
    const treeMat = new THREE.MeshLambertMaterial({ color: 0x1a5a2a });
    const trunkMat = new THREE.MeshLambertMaterial({ color: 0x5a3a1a });

    for (let i = 0; i < 80; i++) {
        const angle = Math.random() * Math.PI * 2;
        const dist = 30 + Math.random() * 120;
        const x = Math.cos(angle) * dist;
        const z = Math.sin(angle) * dist;

        // Skip if too close to center
        if (Math.sqrt(x * x + z * z) < 25) continue;

        const height = 3 + Math.random() * 5;
        const radius = 1.5 + Math.random() * 2;

        // Trunk
        const trunk = new THREE.Mesh(
            new THREE.CylinderGeometry(0.15, 0.25, height * 0.6, 6),
            trunkMat
        );
        trunk.position.set(x, height * 0.3, z);
        trunk.castShadow = true;
        scene.add(trunk);

        // Canopy (cone)
        const canopy = new THREE.Mesh(
            new THREE.ConeGeometry(radius, height * 0.7, 8),
            treeMat
        );
        canopy.position.set(x, height * 0.7, z);
        canopy.castShadow = true;
        scene.add(canopy);
    }
}

function createLandingPad(scene) {
    // Central landing pad (helipad style)
    const padGeo = new THREE.CylinderGeometry(4, 4, 0.1, 32);
    const padMat = new THREE.MeshStandardMaterial({
        color: 0x333333,
        roughness: 0.8,
        metalness: 0.2,
    });
    const pad = new THREE.Mesh(padGeo, padMat);
    pad.position.y = 0.05;
    pad.receiveShadow = true;
    scene.add(pad);

    // H marking
    const hShape = new THREE.Group();
    const barMat = new THREE.MeshStandardMaterial({ color: 0xffcc00, emissive: 0x664400, emissiveIntensity: 0.3 });

    // Left vertical
    const lv = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.05, 2.5), barMat);
    lv.position.set(-0.8, 0.12, 0);
    hShape.add(lv);
    // Right vertical
    const rv = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.05, 2.5), barMat);
    rv.position.set(0.8, 0.12, 0);
    hShape.add(rv);
    // Cross bar
    const cb = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.05, 0.3), barMat);
    cb.position.set(0, 0.12, 0);
    hShape.add(cb);

    scene.add(hShape);

    // Corner lights (blinking)
    const lightColors = [0x00ff00, 0xff0000, 0x00ff00, 0xff0000];
    for (let i = 0; i < 4; i++) {
        const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
        const lx = Math.cos(angle) * 3.5;
        const lz = Math.sin(angle) * 3.5;

        const light = new THREE.PointLight(lightColors[i], 1.5, 8);
        light.position.set(lx, 0.3, lz);
        scene.add(light);

        const bulb = new THREE.Mesh(
            new THREE.SphereGeometry(0.1, 8, 8),
            new THREE.MeshStandardMaterial({
                color: lightColors[i],
                emissive: lightColors[i],
                emissiveIntensity: 2,
            })
        );
        bulb.position.copy(light.position);
        scene.add(bulb);
    }
}

export function updateScene(scene, elapsed, delta) {
    // Animate sun position (slow day cycle)
    if (sun) {
        const sunAngle = elapsed * 0.02;
        sun.position.set(
            200 * Math.cos(sunAngle),
            100 + 80 * Math.sin(sunAngle * 0.5),
            150 * Math.sin(sunAngle),
        );
    }

    // Animate water
    if (waterMesh) {
        waterMesh.position.y = -0.5 + Math.sin(elapsed * 0.5) * 0.05;
    }
}

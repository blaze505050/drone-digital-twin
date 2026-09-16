/**
 * scene.js — High-Fidelity 3D World Environment
 * Multi-band procedural terrain, asphalt runway, dual helipads,
 * hangar facilities, communication towers with hazard beacons,
 * wind turbines, and rotor downwash ground effect particles.
 */
import * as THREE from 'three';

let sun, skyUniforms;
let waterMesh;
let hazardLights = [];
let windTurbineRotors = [];
let dustParticles;
let dustPositions, dustVelocities;
const DUST_COUNT = 150;

export function createScene(scene) {
    // ── Atmospheric Fog ──
    scene.fog = new THREE.FogExp2(0x0c1527, 0.0016);

    // ── Sky (Atmospheric gradient dome) ──
    const skyGeo = new THREE.SphereGeometry(2500, 32, 32);
    const skyMat = new THREE.ShaderMaterial({
        uniforms: {
            topColor: { value: new THREE.Color(0x07111e) },
            bottomColor: { value: new THREE.Color(0x1a3350) },
            horizonColor: { value: new THREE.Color(0xee8844) },
            sunColor: { value: new THREE.Color(0xfffaed) },
            offset: { value: 30 },
            exponent: { value: 0.55 },
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
                if (h < 0.0) skyColor = mix(horizonColor, bottomColor, min(-h * 4.0, 1.0));
                gl_FragColor = vec4(skyColor, 1.0);
            }
        `,
        side: THREE.BackSide,
        depthWrite: false,
    });
    scene.add(new THREE.Mesh(skyGeo, skyMat));
    skyUniforms = skyMat.uniforms;

    // ── Sun & Shadows ──
    sun = new THREE.DirectionalLight(0xfff3e0, 2.8);
    sun.position.set(220, 180, 140);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -90;
    sun.shadow.camera.right = 90;
    sun.shadow.camera.top = 90;
    sun.shadow.camera.bottom = -90;
    sun.shadow.camera.near = 40;
    sun.shadow.camera.far = 600;
    sun.shadow.bias = -0.0004;
    scene.add(sun);
    scene.add(sun.target);

    // Secondary soft fill light
    const fillLight = new THREE.DirectionalLight(0x4477aa, 0.7);
    fillLight.position.set(-150, 100, -100);
    scene.add(fillLight);

    // Ambient & Hemisphere lighting
    scene.add(new THREE.AmbientLight(0x223355, 0.7));
    scene.add(new THREE.HemisphereLight(0x77aacc, 0x223322, 0.6));

    // ── World Geometry ──
    createEnhancedTerrain(scene);
    createWater(scene);
    createAirfieldFacility(scene);
    createTrees(scene);
    createHazardTowers(scene);
    createWindTurbines(scene);
    createGroundEffectParticles(scene);
}

function createEnhancedTerrain(scene) {
    const size = 500;
    const segments = 160;
    const geo = new THREE.PlaneGeometry(size, size, segments, segments);
    geo.rotateX(-Math.PI / 2);

    const positions = geo.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);

        // Fractal elevation
        let h = 0;
        h += Math.sin(x * 0.012) * Math.cos(z * 0.011) * 12;
        h += Math.sin(x * 0.028 + 1.2) * Math.cos(z * 0.024 + 0.8) * 6;
        h += Math.sin(x * 0.07 + 2.5) * Math.cos(z * 0.065 + 1.6) * 2.5;

        // Mountain ridge on North-West quadrant
        if (x < -40 && z < -40) {
            h += Math.abs(x + 40) * 0.18 + Math.abs(z + 40) * 0.16;
        }

        // Airfield plateau flattening (center runway & pads)
        const distCenter = Math.sqrt(x * x + z * z);
        const runwayZone = Math.abs(z) < 18 && Math.abs(x) < 100;
        let flatten = Math.max(0, 1 - distCenter / 60);
        if (runwayZone) flatten = 1.0;

        h *= (1.0 - flatten);
        h = Math.max(0, h);
        positions.setY(i, h);

        // Multi-tier natural vertex coloring
        let r, g, b;
        if (h < 0.8) {
            // Sandy riverbank / lake edge
            r = 0.55; g = 0.50; b = 0.38;
        } else if (h < 5.0) {
            // Lush lowland grass
            r = 0.18; g = 0.38; b = 0.14;
        } else if (h < 10.0) {
            // Highland scrub / soil
            r = 0.32; g = 0.30; b = 0.22;
        } else if (h < 17.0) {
            // Slate rock cliff
            r = 0.42; g = 0.40; b = 0.38;
        } else {
            // Snow-capped peak
            r = 0.88; g = 0.90; b = 0.95;
        }

        colors[i * 3 + 0] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    const terrainMat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.88,
        metalness: 0.08,
        flatShading: true,
    });

    const terrain = new THREE.Mesh(geo, terrainMat);
    terrain.receiveShadow = true;
    scene.add(terrain);
}

function createWater(scene) {
    const waterGeo = new THREE.PlaneGeometry(500, 500);
    waterGeo.rotateX(-Math.PI / 2);
    const waterMat = new THREE.MeshStandardMaterial({
        color: 0x0f2c48,
        roughness: 0.15,
        metalness: 0.85,
        transparent: true,
        opacity: 0.70,
    });
    waterMesh = new THREE.Mesh(waterGeo, waterMat);
    waterMesh.position.y = -0.45;
    waterMesh.receiveShadow = true;
    scene.add(waterMesh);
}

function createAirfieldFacility(scene) {
    const facilityGroup = new THREE.Group();

    // ── 1. Main Asphalt Runway (180m x 14m) ──
    const runwayGeo = new THREE.PlaneGeometry(180, 14);
    runwayGeo.rotateX(-Math.PI / 2);
    const runwayMat = new THREE.MeshStandardMaterial({
        color: 0x181a1d,
        roughness: 0.92,
        metalness: 0.05,
    });
    const runway = new THREE.Mesh(runwayGeo, runwayMat);
    runway.position.set(0, 0.02, 0);
    runway.receiveShadow = true;
    facilityGroup.add(runway);

    // Runway Centerline Dashes
    const dashMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    for (let x = -80; x <= 80; x += 10) {
        const dash = new THREE.Mesh(new THREE.PlaneGeometry(5, 0.45), dashMat);
        dash.rotateX(-Math.PI / 2);
        dash.position.set(x, 0.03, 0);
        facilityGroup.add(dash);
    }

    // Runway Threshold White Stripes (Piano Keys)
    for (let side of [-86, 86]) {
        for (let z = -5; z <= 5; z += 1.8) {
            const stripe = new THREE.Mesh(new THREE.PlaneGeometry(5, 0.9), dashMat);
            stripe.rotateX(-Math.PI / 2);
            stripe.position.set(side, 0.03, z);
            facilityGroup.add(stripe);
        }
    }

    // ── 2. Heli-Pad ALPHA (Primary Launch Pad) ──
    createHelipad(facilityGroup, 0, 0, 'ALPHA', 0x00ff88, 0xffcc00);

    // ── 3. Heli-Pad BRAVO (Secondary Standby Pad) ──
    createHelipad(facilityGroup, 35, 18, 'BRAVO', 0x00d4ff, 0x00f3ff);

    // ── 4. Airfield Hangar Structure ──
    const hangarMat = new THREE.MeshStandardMaterial({ color: 0x2c3340, metalness: 0.4, roughness: 0.6 });
    const hangarRoofMat = new THREE.MeshStandardMaterial({ color: 0x1f2530, metalness: 0.5, roughness: 0.4 });

    const hangar = new THREE.Mesh(new THREE.BoxGeometry(22, 7, 16), hangarMat);
    hangar.position.set(-45, 3.5, 24);
    hangar.castShadow = true;
    hangar.receiveShadow = true;
    facilityGroup.add(hangar);

    const roof = new THREE.Mesh(new THREE.ConeGeometry(14, 3, 4), hangarRoofMat);
    roof.position.set(-45, 8.5, 24);
    roof.rotateY(Math.PI / 4);
    roof.castShadow = true;
    facilityGroup.add(roof);

    scene.add(facilityGroup);
}

function createHelipad(parent, x, z, label, beaconColor, markColor) {
    const padGroup = new THREE.Group();
    padGroup.position.set(x, 0.05, z);

    // Concrete disc
    const padGeo = new THREE.CylinderGeometry(5.2, 5.2, 0.12, 32);
    const padMat = new THREE.MeshStandardMaterial({ color: 0x25282e, roughness: 0.85, metalness: 0.15 });
    const pad = new THREE.Mesh(padGeo, padMat);
    pad.receiveShadow = true;
    padGroup.add(pad);

    // Outer Yellow Circle
    const ringGeo = new THREE.RingGeometry(4.4, 4.8, 32);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({ color: markColor, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.y = 0.07;
    padGroup.add(ring);

    // 'H' Marking
    const barMat = new THREE.MeshBasicMaterial({ color: markColor });
    const leftBar = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 3.2), barMat);
    leftBar.rotateX(-Math.PI / 2);
    leftBar.position.set(-1.1, 0.08, 0);
    padGroup.add(leftBar);

    const rightBar = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 3.2), barMat);
    rightBar.rotateX(-Math.PI / 2);
    rightBar.position.set(1.1, 0.08, 0);
    padGroup.add(rightBar);

    const crossBar = new THREE.Mesh(new THREE.PlaneGeometry(2.3, 0.5), barMat);
    crossBar.rotateX(-Math.PI / 2);
    crossBar.position.set(0, 0.08, 0);
    padGroup.add(crossBar);

    // Perimeter Pulsing LED Beacons
    for (let i = 0; i < 8; i++) {
        const theta = (i / 8) * Math.PI * 2;
        const lx = Math.cos(theta) * 4.9;
        const lz = Math.sin(theta) * 4.9;

        const bulb = new THREE.Mesh(
            new THREE.SphereGeometry(0.12, 8, 8),
            new THREE.MeshStandardMaterial({
                color: beaconColor,
                emissive: beaconColor,
                emissiveIntensity: 2.2,
            })
        );
        bulb.position.set(lx, 0.2, lz);
        padGroup.add(bulb);

        const pointLight = new THREE.PointLight(beaconColor, 1.2, 7);
        pointLight.position.set(lx, 0.35, lz);
        padGroup.add(pointLight);
    }

    parent.add(padGroup);
}

function createHazardTowers(scene) {
    const towerCoords = [
        [-80, 50],
        [75, -60],
        [-60, -75],
    ];

    const towerMat = new THREE.MeshStandardMaterial({ color: 0xcc2222, metalness: 0.6, roughness: 0.3 });

    for (let [tx, tz] of towerCoords) {
        const towerGroup = new THREE.Group();
        towerGroup.position.set(tx, 0, tz);

        // Lattice tower mast
        const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 1.4, 30, 4), towerMat);
        mast.position.y = 15;
        mast.castShadow = true;
        towerGroup.add(mast);

        // Top strobe beacon
        const beacon = new THREE.Mesh(
            new THREE.SphereGeometry(0.35, 8, 8),
            new THREE.MeshStandardMaterial({
                color: 0xff1111,
                emissive: 0xff0000,
                emissiveIntensity: 3.5,
            })
        );
        beacon.position.y = 30.5;
        towerGroup.add(beacon);

        const redLight = new THREE.PointLight(0xff0000, 2.5, 30);
        redLight.position.set(0, 31, 0);
        towerGroup.add(redLight);
        hazardLights.push({ mesh: beacon, light: redLight });

        scene.add(towerGroup);
    }
}

function createWindTurbines(scene) {
    const turbineCoords = [
        [110, 80],
        [140, 110],
    ];

    const whiteMat = new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.1, roughness: 0.4 });

    for (let [wx, wz] of turbineCoords) {
        const turbine = new THREE.Group();
        turbine.position.set(wx, 0, wz);

        // Tower
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 1.8, 38, 12), whiteMat);
        pole.position.y = 19;
        pole.castShadow = true;
        turbine.add(pole);

        // Nacelle
        const nacelle = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 5), whiteMat);
        nacelle.position.set(0, 38, 0);
        nacelle.castShadow = true;
        turbine.add(nacelle);

        // Rotor Hub + 3 Blades
        const rotor = new THREE.Group();
        rotor.position.set(0, 38, 2.6);

        const hub = new THREE.Mesh(new THREE.SphereGeometry(1.0, 12, 12), whiteMat);
        rotor.add(hub);

        for (let b = 0; b < 3; b++) {
            const blade = new THREE.Mesh(new THREE.BoxGeometry(0.5, 14, 0.15), whiteMat);
            blade.position.y = 7;
            blade.castShadow = true;

            const bladeHolder = new THREE.Group();
            bladeHolder.rotateZ((b * Math.PI * 2) / 3);
            bladeHolder.add(blade);
            rotor.add(bladeHolder);
        }

        turbine.add(rotor);
        windTurbineRotors.push(rotor);
        scene.add(turbine);
    }
}

function createTrees(scene) {
    const treeMat = new THREE.MeshLambertMaterial({ color: 0x164d26 });
    const trunkMat = new THREE.MeshLambertMaterial({ color: 0x4a2e16 });

    for (let i = 0; i < 90; i++) {
        const angle = Math.random() * Math.PI * 2;
        const dist = 38 + Math.random() * 140;
        const x = Math.cos(angle) * dist;
        const z = Math.sin(angle) * dist;

        // Keep clear of runway and pads
        if (Math.abs(z) < 22 && Math.abs(x) < 100) continue;

        const height = 4 + Math.random() * 5;
        const radius = 1.8 + Math.random() * 2.2;

        const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.35, height * 0.5, 6), trunkMat);
        trunk.position.set(x, height * 0.25, z);
        trunk.castShadow = true;
        scene.add(trunk);

        const canopy = new THREE.Mesh(new THREE.ConeGeometry(radius, height * 0.75, 7), treeMat);
        canopy.position.set(x, height * 0.75, z);
        canopy.castShadow = true;
        scene.add(canopy);
    }
}

function createGroundEffectParticles(scene) {
    const geo = new THREE.BufferGeometry();
    dustPositions = new Float32Array(DUST_COUNT * 3);
    dustVelocities = new Float32Array(DUST_COUNT * 3);

    for (let i = 0; i < DUST_COUNT; i++) {
        dustPositions[i * 3 + 0] = 0;
        dustPositions[i * 3 + 1] = -100; // start hidden
        dustPositions[i * 3 + 2] = 0;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(dustPositions, 3));

    const mat = new THREE.PointsMaterial({
        color: 0xddc8a0,
        size: 0.35,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });

    dustParticles = new THREE.Points(geo, mat);
    scene.add(dustParticles);
}

export function updateGroundEffect(dronePos, throttle, delta) {
    if (!dustParticles) return;

    const alt = dronePos.y;
    // Ground effect activates when drone is within 4 meters of surface
    if (alt < 4.0 && throttle > 0.15) {
        dustParticles.material.opacity = Math.min(0.65, (1.0 - alt / 4.0) * throttle * 1.5);
        const posAttr = dustParticles.geometry.attributes.position;

        for (let i = 0; i < DUST_COUNT; i++) {
            let px = posAttr.getX(i);
            let py = posAttr.getY(i);
            let pz = posAttr.getZ(i);

            // If particle died or is too far, respawn in ring under drone
            const distFromDrone = Math.sqrt((px - dronePos.x)**2 + (pz - dronePos.z)**2);
            if (distFromDrone > 4.5 || py < 0.01 || Math.random() < 0.04) {
                const angle = Math.random() * Math.PI * 2;
                const r = 0.4 + Math.random() * 1.2;
                px = dronePos.x + Math.cos(angle) * r;
                pz = dronePos.z + Math.sin(angle) * r;
                py = 0.05 + Math.random() * 0.15;
                dustVelocities[i * 3 + 0] = Math.cos(angle) * (2.5 + Math.random() * 3.0);
                dustVelocities[i * 3 + 1] = 0.2 + Math.random() * 0.5;
                dustVelocities[i * 3 + 2] = Math.sin(angle) * (2.5 + Math.random() * 3.0);
            } else {
                px += dustVelocities[i * 3 + 0] * delta;
                py += dustVelocities[i * 3 + 1] * delta;
                pz += dustVelocities[i * 3 + 2] * delta;
                dustVelocities[i * 3 + 1] -= 0.8 * delta; // gravity
            }

            posAttr.setXYZ(i, px, py, pz);
        }
        posAttr.needsUpdate = true;
    } else {
        dustParticles.material.opacity = Math.max(0.0, dustParticles.material.opacity - delta * 2.0);
    }
}

export function updateScene(scene, elapsed, delta) {
    // 1. Slow sun orbit
    if (sun) {
        const sunAngle = elapsed * 0.025;
        sun.position.set(
            220 * Math.cos(sunAngle),
            120 + 70 * Math.sin(sunAngle * 0.5),
            140 * Math.sin(sunAngle),
        );
    }

    // 2. Animated water ripples
    if (waterMesh) {
        waterMesh.position.y = -0.45 + Math.sin(elapsed * 0.8) * 0.04;
    }

    // 3. Spin wind turbines
    for (let rotor of windTurbineRotors) {
        rotor.rotation.z -= delta * 1.2;
    }

    // 4. Strobe tower hazard lights (1 Hz blink)
    const strobe = Math.sin(elapsed * 6.28) > 0.6;
    for (let item of hazardLights) {
        item.mesh.material.emissiveIntensity = strobe ? 4.0 : 0.2;
        item.light.intensity = strobe ? 3.0 : 0.1;
    }
}

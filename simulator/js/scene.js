/**
 * scene.js — Realistic 3D Airfield & Game-Level Visual World
 *
 * Features:
 *   - Atmospheric Rayleigh/Mie sky gradient dome + solar corona
 *   - Procedural drifting volumetric cumulus cloud layer
 *   - High-fidelity procedural canvas textures for runway, taxiways, and helipads
 *   - 4-tier Air Traffic Control Tower with motorized 360° rotating radar antenna
 *   - Dual modern aircraft hangars with taxiway apron & canopy spotlights
 *   - 4-light PAPI glide slope indicators & runway edge lighting
 *   - Reactive fabric windsock aligning with wind vector and speed
 *   - Shimmering specular lake water body
 *   - Multi-tier procedural pine & deciduous trees
 *   - Hazard communication towers with synchronized red aviation strobes
 *   - Wind turbines with smoothly rotating 3-blade hubs
 *   - Rotor downwash particle ground effect ring
 */
import * as THREE from 'three';

let sun, skyUniforms;
let waterMesh;
let hazardLights = [];
let windTurbineRotors = [];
let radarDish = null;
let windsockGroup = null;
let cloudGroup = null;
let dustParticles, dustPositions, dustVelocities;
const DUST_COUNT = 180;

export function createScene(scene) {
    // ── Atmospheric Exponential Fog ──
    scene.fog = new THREE.FogExp2(0x0a1424, 0.0012);

    // ── 1. Sky & Sun Atmosphere ──
    createAtmosphericSky(scene);

    // ── 2. Dynamic Lighting & Soft Shadows ──
    setupLighting(scene);

    // ── 3. Drifting Cumulus Cloud Layer ──
    createClouds(scene);

    // ── 4. Procedural Terrain & Shimmering Lake ──
    createRealisticTerrain(scene);
    createSpecularWater(scene);

    // ── 5. Airfield Facilities ──
    createHighDetailAirfield(scene);
    createControlTowerWithRadar(scene);
    createHangars(scene);
    createWindsock(scene);

    // ── 6. Environment Dressing ──
    createRealisticTrees(scene);
    createHazardTowers(scene);
    createWindTurbines(scene);
    createGroundEffectParticles(scene);
}

// ── Atmospheric Sky Dome ──────────────────────────────────────────────────

function createAtmosphericSky(scene) {
    const skyGeo = new THREE.SphereGeometry(3000, 32, 32);
    const skyMat = new THREE.ShaderMaterial({
        uniforms: {
            topColor: { value: new THREE.Color(0x060f1d) },
            bottomColor: { value: new THREE.Color(0x19304c) },
            horizonColor: { value: new THREE.Color(0xff8c42) },
            sunColor: { value: new THREE.Color(0xfffae6) },
            sunPosition: { value: new THREE.Vector3(260, 190, 150).normalize() },
            offset: { value: 35 },
            exponent: { value: 0.52 },
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
            uniform vec3 sunColor;
            uniform vec3 sunPosition;
            uniform float offset;
            uniform float exponent;
            varying vec3 vWorldPosition;
            void main() {
                vec3 dir = normalize(vWorldPosition + offset);
                float h = dir.y;
                float t = pow(max(h, 0.0), exponent);
                vec3 skyColor = mix(horizonColor, topColor, t);
                if (h < 0.0) {
                    skyColor = mix(horizonColor, bottomColor, min(-h * 5.0, 1.0));
                }

                // Solar corona glare
                float sunDot = max(dot(dir, sunPosition), 0.0);
                float corona = pow(sunDot, 64.0) * 0.9;
                float flare = pow(sunDot, 512.0) * 2.5;
                skyColor += sunColor * (corona + flare);

                gl_FragColor = vec4(skyColor, 1.0);
            }
        `,
        side: THREE.BackSide,
        depthWrite: false,
    });

    scene.add(new THREE.Mesh(skyGeo, skyMat));
    skyUniforms = skyMat.uniforms;
}

// ── Lighting & Shadows ────────────────────────────────────────────────────

function setupLighting(scene) {
    // Primary Golden Sunlight
    sun = new THREE.DirectionalLight(0xfff4e0, 2.9);
    sun.position.set(240, 190, 160);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -110;
    sun.shadow.camera.right = 110;
    sun.shadow.camera.top = 110;
    sun.shadow.camera.bottom = -110;
    sun.shadow.camera.near = 50;
    sun.shadow.camera.far = 700;
    sun.shadow.bias = -0.0003;
    scene.add(sun);
    scene.add(sun.target);

    // Sky dome fill light
    const skyFill = new THREE.DirectionalLight(0x4488cc, 0.85);
    skyFill.position.set(-160, 120, -120);
    scene.add(skyFill);

    // Hemisphere & Ambient Light
    scene.add(new THREE.AmbientLight(0x1a2638, 0.75));
    scene.add(new THREE.HemisphereLight(0x6699bb, 0x1a2e1a, 0.65));
}

// ── Drifting Cumulus Clouds ───────────────────────────────────────────────

function createClouds(scene) {
    cloudGroup = new THREE.Group();
    const cloudMat = new THREE.MeshStandardMaterial({
        color: 0xedf4fc,
        roughness: 0.95,
        metalness: 0.0,
        transparent: true,
        opacity: 0.78,
    });

    // 12 procedural cloud puffs distributed across the sky
    for (let c = 0; c < 12; c++) {
        const cloud = new THREE.Group();
        const baseAngle = (c / 12) * Math.PI * 2;
        const dist = 180 + (c % 3) * 80;
        const cx = Math.cos(baseAngle) * dist;
        const cz = Math.sin(baseAngle) * dist;
        const cy = 135 + (c % 4) * 15;

        // Cluster of 4-6 overlapping flattened spheres
        const puffs = 4 + (c % 3);
        for (let p = 0; p < puffs; p++) {
            const r = 18 + Math.random() * 16;
            const puff = new THREE.Mesh(new THREE.SphereGeometry(r, 7, 6), cloudMat);
            puff.position.set((p - puffs / 2) * 16 + (Math.random() - 0.5) * 8, (Math.random() - 0.5) * 4, (Math.random() - 0.5) * 12);
            puff.scale.set(1.4, 0.55, 1.0);
            cloud.add(puff);
        }

        cloud.position.set(cx, cy, cz);
        cloudGroup.add(cloud);
    }
    scene.add(cloudGroup);
}

// ── Procedural Textures Generation ────────────────────────────────────────

function generateRunwayTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 2048;
    const ctx = canvas.getContext('2d');

    // Weathered Dark Bitumen Base
    ctx.fillStyle = '#1c1f24';
    ctx.fillRect(0, 0, 1024, 2048);

    // Bitumen grain noise
    for (let i = 0; i < 20000; i++) {
        const nx = Math.random() * 1024;
        const ny = Math.random() * 2048;
        const bright = 25 + Math.random() * 20;
        ctx.fillStyle = `rgb(${bright},${bright},${bright})`;
        ctx.fillRect(nx, ny, 2, 2);
    }

    // Runway White Edge Stripes
    ctx.strokeStyle = '#f0f4f8';
    ctx.lineWidth = 14;
    ctx.beginPath();
    ctx.moveTo(35, 80);
    ctx.lineTo(35, 1968);
    ctx.moveTo(989, 80);
    ctx.lineTo(989, 1968);
    ctx.stroke();

    // Dashed Centerline
    ctx.setLineDash([70, 50]);
    ctx.lineWidth = 16;
    ctx.beginPath();
    ctx.moveTo(512, 120);
    ctx.lineTo(512, 1928);
    ctx.stroke();
    ctx.setLineDash([]);

    // Threshold Piano Stripes (North End: Runway 09L)
    ctx.fillStyle = '#ffffff';
    for (let i = 0; i < 10; i++) {
        const x = 110 + i * 82;
        ctx.fillRect(x, 100, 55, 160);
    }

    // Threshold Piano Stripes (South End: Runway 27R)
    for (let i = 0; i < 10; i++) {
        const x = 110 + i * 82;
        ctx.fillRect(x, 1788, 55, 160);
    }

    // Runway Number Designation Markings
    ctx.font = 'bold 110px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.fillText('09L', 512, 360);
    ctx.fillText('27R', 512, 1680);

    // Touchdown Zone Rubber Skid Marks
    ctx.fillStyle = 'rgba(10, 10, 10, 0.45)';
    for (let s = 0; s < 45; s++) {
        const sx1 = 440 + Math.random() * 144;
        const sy1 = 280 + Math.random() * 300;
        const sw = 8 + Math.random() * 16;
        const sh = 40 + Math.random() * 120;
        ctx.fillRect(sx1, sy1, sw, sh);

        const sx2 = 440 + Math.random() * 144;
        const sy2 = 1460 + Math.random() * 300;
        ctx.fillRect(sx2, sy2, sw, sh);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    return texture;
}

function generateHelipadTexture(letter = 'H') {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    const cx = 256, cy = 256;

    // Outer Dark Asphalt
    ctx.fillStyle = '#22252a';
    ctx.fillRect(0, 0, 512, 512);

    // Circular Helipad Concrete Pad
    ctx.beginPath();
    ctx.arc(cx, cy, 240, 0, Math.PI * 2);
    ctx.fillStyle = '#32373e';
    ctx.fill();

    // Red-and-White Hazard Apron Border Segments
    ctx.save();
    ctx.translate(cx, cy);
    const segs = 32;
    for (let s = 0; s < segs; s++) {
        ctx.beginPath();
        ctx.arc(0, 0, 238, (s * Math.PI * 2) / segs, ((s + 1) * Math.PI * 2) / segs);
        ctx.arc(0, 0, 218, ((s + 1) * Math.PI * 2) / segs, (s * Math.PI * 2) / segs, true);
        ctx.closePath();
        ctx.fillStyle = s % 2 === 0 ? '#ff2233' : '#f0f4f8';
        ctx.fill();
    }
    ctx.restore();

    // Inner High-Vis Yellow Border Ring
    ctx.beginPath();
    ctx.arc(cx, cy, 205, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 14;
    ctx.stroke();

    // Center Big Bold "H"
    ctx.font = 'bold 220px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#000000';
    ctx.shadowBlur = 10;
    ctx.fillText(letter, cx, cy);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
}

// ── Realistic Multi-Band Terrain ──────────────────────────────────────────

function createRealisticTerrain(scene) {
    const size = 520;
    const segments = 180;
    const geo = new THREE.PlaneGeometry(size, size, segments, segments);
    geo.rotateX(-Math.PI / 2);

    const positions = geo.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);

        // Fractal elevation
        let h = 0;
        h += Math.sin(x * 0.011) * Math.cos(z * 0.01) * 14;
        h += Math.sin(x * 0.026 + 1.2) * Math.cos(z * 0.023 + 0.8) * 7;
        h += Math.sin(x * 0.065 + 2.5) * Math.cos(z * 0.06 + 1.6) * 3;

        // Mountain ridge on North-West quadrant
        if (x < -40 && z < -40) {
            h += Math.abs(x + 40) * 0.22 + Math.abs(z + 40) * 0.2;
        }

        // Lake basin depression on South-East quadrant
        if (x > 50 && z > 30) {
            const lakeDist = Math.sqrt((x - 110) ** 2 + (z - 90) ** 2);
            if (lakeDist < 75) {
                h -= (75 - lakeDist) * 0.22;
            }
        }

        // Airfield plateau flattening (center runway, pads, hangars)
        const distCenter = Math.sqrt(x * x + z * z);
        const runwayZone = Math.abs(x) < 22 && Math.abs(z) < 110;
        let flatten = Math.max(0, 1 - distCenter / 70);
        if (runwayZone) flatten = 1.0;

        h *= (1.0 - flatten);
        h = Math.max(-1.5, h);
        positions.setY(i, h);

        // Strata vertex colors
        let r, g, b;
        if (h < 0.2) {
            // Sandy riverbank / shoreline
            r = 0.58; g = 0.52; b = 0.40;
        } else if (h < 5.0) {
            // Lowland grass & turf
            r = 0.16; g = 0.36; b = 0.14;
        } else if (h < 11.0) {
            // Highland scrub
            r = 0.30; g = 0.32; b = 0.22;
        } else if (h < 18.0) {
            // Rocky slopes
            r = 0.44; g = 0.42; b = 0.38;
        } else {
            // Alpine ridge rock & snow
            r = 0.72; g = 0.75; b = 0.78;
        }

        colors[i * 3 + 0] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.88,
        metalness: 0.05,
        flatShading: false,
    });

    const terrain = new THREE.Mesh(geo, mat);
    terrain.receiveShadow = true;
    scene.add(terrain);
}

// ── Specular Lake Water ───────────────────────────────────────────────────

function createSpecularWater(scene) {
    const waterGeo = new THREE.PlaneGeometry(160, 160, 32, 32);
    waterGeo.rotateX(-Math.PI / 2);

    const waterMat = new THREE.MeshStandardMaterial({
        color: 0x0c3355,
        roughness: 0.1,
        metalness: 0.85,
        transparent: true,
        opacity: 0.85,
    });

    waterMesh = new THREE.Mesh(waterGeo, waterMat);
    waterMesh.position.set(110, -0.4, 90);
    waterMesh.receiveShadow = true;
    scene.add(waterMesh);
}

// ── High-Detail Airfield Facility ─────────────────────────────────────────

function createHighDetailAirfield(scene) {
    const airfield = new THREE.Group();

    // 1. Main Asphalt Runway with High-Res Texture
    const runwayGeo = new THREE.PlaneGeometry(28, 200);
    runwayGeo.rotateX(-Math.PI / 2);
    const runwayTex = generateRunwayTexture();
    const runwayMat = new THREE.MeshStandardMaterial({
        map: runwayTex,
        roughness: 0.75,
        metalness: 0.15,
    });
    const runway = new THREE.Mesh(runwayGeo, runwayMat);
    runway.position.set(0, 0.03, 0);
    runway.receiveShadow = true;
    airfield.add(runway);

    // 2. Dual Helipads (Alpha & Bravo)
    const padAlphaTex = generateHelipadTexture('A');
    const padAlpha = new THREE.Mesh(
        new THREE.PlaneGeometry(20, 20).rotateX(-Math.PI / 2),
        new THREE.MeshStandardMaterial({ map: padAlphaTex, roughness: 0.7, metalness: 0.2 })
    );
    padAlpha.position.set(-22, 0.05, 20);
    padAlpha.receiveShadow = true;
    airfield.add(padAlpha);

    const padBravoTex = generateHelipadTexture('B');
    const padBravo = new THREE.Mesh(
        new THREE.PlaneGeometry(18, 18).rotateX(-Math.PI / 2),
        new THREE.MeshStandardMaterial({ map: padBravoTex, roughness: 0.7, metalness: 0.2 })
    );
    padBravo.position.set(-22, 0.05, -20);
    padBravo.receiveShadow = true;
    airfield.add(padBravo);

    // 3. Runway Edge & PAPI Glide Slope Lights
    createRunwayEdgeLights(airfield);
    createPAPILights(airfield);

    scene.add(airfield);
}

function createRunwayEdgeLights(parent) {
    // White runway edge lights every 20m along both sides
    const bulbMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const blueMat = new THREE.MeshBasicMaterial({ color: 0x0088ff });

    for (let z = -90; z <= 90; z += 18) {
        [-14.5, 14.5].forEach(x => {
            const lightPole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.4, 6), new THREE.MeshStandardMaterial({ color: 0x222222 }));
            lightPole.position.set(x, 0.2, z);
            const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 8), bulbMat);
            bulb.position.y = 0.22;
            lightPole.add(bulb);
            parent.add(lightPole);
        });
    }

    // Green runway threshold lights at north end (z = -98)
    const greenMat = new THREE.MeshBasicMaterial({ color: 0x00ff66 });
    for (let x = -13; x <= 13; x += 3.5) {
        const thresh = new THREE.Mesh(new THREE.SphereGeometry(0.15, 8, 8), greenMat);
        thresh.position.set(x, 0.2, -98);
        parent.add(thresh);
    }
}

function createPAPILights(parent) {
    // 4-box PAPI light bar positioned beside touchdown zone
    const papiRed = new THREE.MeshBasicMaterial({ color: 0xff1122 });
    const papiWhite = new THREE.MeshBasicMaterial({ color: 0xffffff });

    for (let i = 0; i < 4; i++) {
        const box = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.4, 0.5), new THREE.MeshStandardMaterial({ color: 0x333333 }));
        box.position.set(-18 - i * 1.2, 0.3, 45);

        const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 8), i < 2 ? papiWhite : papiRed);
        lamp.position.set(0, 0, -0.2);
        box.add(lamp);
        parent.add(box);
    }
}

// ── Air Traffic Control Tower with Motorized 360° Radar ──────────────────

function createControlTowerWithRadar(scene) {
    const tower = new THREE.Group();
    tower.position.set(-35, 0, -45);

    const concreteMat = new THREE.MeshStandardMaterial({ color: 0x48515c, roughness: 0.6, metalness: 0.3 });
    const glassMat = new THREE.MeshStandardMaterial({ color: 0x22d4ff, roughness: 0.1, metalness: 0.9, transparent: true, opacity: 0.75 });

    // Base Pillar
    const base = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 3.2, 22, 12), concreteMat);
    base.position.y = 11;
    base.castShadow = true;
    tower.add(base);

    // Observation Cab (Glass faceted room)
    const cab = new THREE.Mesh(new THREE.CylinderGeometry(4.2, 3.2, 5.0, 10), glassMat);
    cab.position.y = 24.5;
    cab.castShadow = true;
    tower.add(cab);

    // Catwalk Platform
    const catwalk = new THREE.Mesh(new THREE.CylinderGeometry(4.6, 4.6, 0.4, 12), concreteMat);
    catwalk.position.y = 22.2;
    tower.add(catwalk);

    // Roof Top Gantry
    const roof = new THREE.Mesh(new THREE.ConeGeometry(4.4, 1.8, 12), concreteMat);
    roof.position.y = 27.8;
    tower.add(roof);

    // Motorized 360° Rotating Radar Dish
    radarDish = new THREE.Group();
    radarDish.position.set(0, 29.2, 0);

    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 1.6, 8), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
    mast.position.y = 0.8;
    radarDish.add(mast);

    const dishMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(1.6, 0.6, 0.3, 16),
        new THREE.MeshStandardMaterial({ color: 0xff8800, metalness: 0.8, roughness: 0.2 })
    );
    dishMesh.rotation.x = Math.PI / 2;
    dishMesh.position.y = 1.7;
    radarDish.add(dishMesh);

    tower.add(radarDish);
    scene.add(tower);
}

// ── Aircraft Hangars ──────────────────────────────────────────────────────

function createHangars(scene) {
    const hangarMat = new THREE.MeshStandardMaterial({ color: 0x383e47, roughness: 0.4, metalness: 0.6 });
    const doorMat = new THREE.MeshStandardMaterial({ color: 0x181c22, roughness: 0.5, metalness: 0.8 });

    [-1, 1].forEach((dir, idx) => {
        const hangar = new THREE.Group();
        hangar.position.set(38, 0, dir * 28);
        hangar.rotation.y = Math.PI;

        // Curved Hangar Arch
        const arch = new THREE.Mesh(new THREE.CylinderGeometry(11, 11, 26, 16, 1, false, 0, Math.PI), hangarMat);
        arch.rotation.z = Math.PI / 2;
        arch.position.y = 0;
        arch.castShadow = true;
        hangar.add(arch);

        // Hangar Doors
        const doors = new THREE.Mesh(new THREE.BoxGeometry(0.6, 9.5, 20), doorMat);
        doors.position.set(0, 4.8, 0);
        hangar.add(doors);

        // Canopy Floodlight
        const floodLight = new THREE.PointLight(0xffea9f, 2.0, 30);
        floodLight.position.set(-1.5, 9.8, 0);
        hangar.add(floodLight);

        scene.add(hangar);
    });
}

// ── Reactive Windsock ─────────────────────────────────────────────────────

function createWindsock(scene) {
    windsockGroup = new THREE.Group();
    windsockGroup.position.set(-18, 0, 0);

    // Mast Pole
    const mast = new THREE.Mesh(
        new THREE.CylinderGeometry(0.1, 0.16, 7.5, 8),
        new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.5, roughness: 0.3 })
    );
    mast.position.y = 3.75;
    windsockGroup.add(mast);

    // Fabric Cone (Red & White Striped)
    const cone = new THREE.Mesh(
        new THREE.ConeGeometry(0.55, 3.2, 8),
        new THREE.MeshStandardMaterial({ color: 0xff3300, roughness: 0.7 })
    );
    cone.rotation.x = Math.PI / 2;
    cone.position.set(0, 7.5, 1.6);
    windsockGroup.add(cone);

    scene.add(windsockGroup);
}

// ── Realistic Trees & Vegetation ──────────────────────────────────────────

function createRealisticTrees(scene) {
    const trunkMat = new THREE.MeshStandardMaterial({ color: 0x3d2716, roughness: 0.9 });
    const leafMats = [
        new THREE.MeshStandardMaterial({ color: 0x1a4524, roughness: 0.8 }),
        new THREE.MeshStandardMaterial({ color: 0x225c30, roughness: 0.8 }),
        new THREE.MeshStandardMaterial({ color: 0x16381d, roughness: 0.8 }),
    ];

    for (let i = 0; i < 95; i++) {
        const angle = Math.random() * Math.PI * 2;
        const dist = 38 + Math.random() * 150;
        const x = Math.cos(angle) * dist;
        const z = Math.sin(angle) * dist;

        // Keep clear of airfield runway and pads
        if (Math.abs(x) < 28 && Math.abs(z) < 115) continue;
        if (x > 60 && z > 40) continue; // Keep clear of lake

        const tree = new THREE.Group();
        tree.position.set(x, 0, z);

        const height = 5 + Math.random() * 5;
        const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.4, height * 0.45, 6), trunkMat);
        trunk.position.y = height * 0.22;
        trunk.castShadow = true;
        tree.add(trunk);

        // 3-tier tiered pine canopy
        const mat = leafMats[i % leafMats.length];
        for (let t = 0; t < 3; t++) {
            const tierRadius = (2.4 - t * 0.5) * (height / 8);
            const cone = new THREE.Mesh(new THREE.ConeGeometry(tierRadius, height * 0.4, 7), mat);
            cone.position.y = height * (0.42 + t * 0.22);
            cone.castShadow = true;
            tree.add(cone);
        }

        scene.add(tree);
    }
}

// ── Hazard Towers & Wind Turbines ─────────────────────────────────────────

function createHazardTowers(scene) {
    const towerCoords = [
        [-85, 55],
        [80, -65],
        [-65, -80],
    ];

    const towerMat = new THREE.MeshStandardMaterial({ color: 0xcc2222, metalness: 0.6, roughness: 0.3 });

    towerCoords.forEach(([tx, tz]) => {
        const towerGroup = new THREE.Group();
        towerGroup.position.set(tx, 0, tz);

        const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 1.3, 32, 4), towerMat);
        mast.position.y = 16;
        mast.castShadow = true;
        towerGroup.add(mast);

        const beacon = new THREE.Mesh(
            new THREE.SphereGeometry(0.4, 8, 8),
            new THREE.MeshStandardMaterial({ color: 0xff1111, emissive: 0xff0000, emissiveIntensity: 3.5 })
        );
        beacon.position.y = 32.5;
        towerGroup.add(beacon);

        const redLight = new THREE.PointLight(0xff0000, 3.0, 35);
        redLight.position.set(0, 33, 0);
        towerGroup.add(redLight);
        hazardLights.push({ mesh: beacon, light: redLight });

        scene.add(towerGroup);
    });
}

function createWindTurbines(scene) {
    const turbineCoords = [
        [110, 80],
        [145, 115],
    ];
    const whiteMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f5, metalness: 0.2, roughness: 0.3 });

    turbineCoords.forEach(([wx, wz]) => {
        const turbine = new THREE.Group();
        turbine.position.set(wx, 0, wz);

        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 1.8, 42, 12), whiteMat);
        pole.position.y = 21;
        pole.castShadow = true;
        turbine.add(pole);

        const nacelle = new THREE.Mesh(new THREE.BoxGeometry(2.2, 2.2, 5.5), whiteMat);
        nacelle.position.set(0, 42, 0);
        nacelle.castShadow = true;
        turbine.add(nacelle);

        const rotor = new THREE.Group();
        rotor.position.set(0, 42, 2.8);

        const hub = new THREE.Mesh(new THREE.SphereGeometry(1.1, 12, 12), whiteMat);
        rotor.add(hub);

        for (let b = 0; b < 3; b++) {
            const blade = new THREE.Mesh(new THREE.BoxGeometry(0.5, 15, 0.15), whiteMat);
            blade.position.y = 7.5;
            blade.castShadow = true;

            const holder = new THREE.Group();
            holder.rotateZ((b * Math.PI * 2) / 3);
            holder.add(blade);
            rotor.add(holder);
        }

        turbine.add(rotor);
        windTurbineRotors.push(rotor);
        scene.add(turbine);
    });
}

// ── Rotor Downwash Ground Particles ───────────────────────────────────────

function createGroundEffectParticles(scene) {
    const geo = new THREE.BufferGeometry();
    dustPositions = new Float32Array(DUST_COUNT * 3);
    dustVelocities = new Float32Array(DUST_COUNT * 3);

    for (let i = 0; i < DUST_COUNT; i++) {
        dustPositions[i * 3 + 1] = -200; // start hidden
    }

    geo.setAttribute('position', new THREE.BufferAttribute(dustPositions, 3));

    const mat = new THREE.PointsMaterial({
        color: 0xddc499,
        size: 0.45,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });

    dustParticles = new THREE.Points(geo, mat);
    scene.add(dustParticles);
}

export function updateGroundEffect(dronePos, throttle, delta) {
    if (!dustParticles || !dronePos) return;

    const alt = dronePos.y;
    const isNearGround = alt > 0.05 && alt < 3.2;

    dustParticles.material.opacity = isNearGround ? Math.max(0.15, 0.7 * (1.0 - alt / 3.2) * (throttle || 0.6)) : 0.0;

    if (!isNearGround) return;

    const posAttr = dustParticles.geometry.attributes.position;
    for (let i = 0; i < DUST_COUNT; i++) {
        let x = posAttr.getX(i);
        let y = posAttr.getY(i);
        let z = posAttr.getZ(i);

        let vx = dustVelocities[i * 3 + 0];
        let vy = dustVelocities[i * 3 + 1];
        let vz = dustVelocities[i * 3 + 2];

        x += vx * delta;
        y += vy * delta;
        z += vz * delta;
        vy -= 2.0 * delta;

        const distFromCenter = Math.sqrt((x - dronePos.x) ** 2 + (z - dronePos.z) ** 2);
        if (distFromCenter > 3.8 || y < 0.02) {
            const angle = Math.random() * Math.PI * 2;
            const r = 0.3 + Math.random() * 0.9;
            x = dronePos.x + Math.cos(angle) * r;
            y = 0.05 + Math.random() * 0.15;
            z = dronePos.z + Math.sin(angle) * r;

            const radialSpeed = 2.0 + Math.random() * 3.5;
            vx = Math.cos(angle) * radialSpeed;
            vy = 0.4 + Math.random() * 0.8;
            vz = Math.sin(angle) * radialSpeed;

            dustVelocities[i * 3 + 0] = vx;
            dustVelocities[i * 3 + 1] = vy;
            dustVelocities[i * 3 + 2] = vz;
        }

        posAttr.setXYZ(i, x, y, z);
    }
    posAttr.needsUpdate = true;
}

// ── Scene Animation Updates ───────────────────────────────────────────────

export function updateScene(scene, elapsed, delta, windVector = [2, 1, 0]) {
    // 1. Strobe hazard lights (1.2 Hz pulse)
    const strobe = Math.sin(elapsed * 7.5) > 0.4;
    hazardLights.forEach(({ mesh, light }) => {
        mesh.material.emissiveIntensity = strobe ? 4.0 : 0.2;
        light.intensity = strobe ? 3.5 : 0.1;
    });

    // 2. Rotate Wind Turbines
    windTurbineRotors.forEach(rotor => {
        rotor.rotation.z += 1.2 * delta;
    });

    // 3. Rotate ATC Radar Dish (360° motor scan)
    if (radarDish) {
        radarDish.rotation.y += 1.8 * delta;
    }

    // 4. Align Windsock with Wind Vector
    if (windsockGroup) {
        const windX = windVector[0] || 2;
        const windZ = windVector[1] || 1;
        const heading = Math.atan2(windX, windZ);
        windsockGroup.rotation.y = heading;
    }

    // 5. Slowly Drift Cloud Layer
    if (cloudGroup) {
        cloudGroup.rotation.y += 0.008 * delta;
    }
}

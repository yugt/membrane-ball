// Cyber-Neon Membrane Breakout 3D Game Engine (Massless Circular Membrane - LOCKED FRAME VERIFICATION)

// --- Audio Synthesizer (Web Audio API) ---
class GameAudio {
    constructor() {
        this.ctx = null;
    }
    init() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }
    playBounce() {
        this.init();
        if (!this.ctx) return;
        let osc = this.ctx.createOscillator();
        let gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        
        osc.type = 'sine';
        osc.frequency.setValueAtTime(150, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(300, this.ctx.currentTime + 0.15);
        
        gain.gain.setValueAtTime(0.12, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.15);
        
        osc.start();
        osc.stop(this.ctx.currentTime + 0.15);
    }
    playHitBrick() {
        this.init();
        if (!this.ctx) return;
        let osc = this.ctx.createOscillator();
        let gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(440, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(880, this.ctx.currentTime + 0.1);
        
        gain.gain.setValueAtTime(0.18, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.12);
        
        osc.start();
        osc.stop(this.ctx.currentTime + 0.12);
    }
    playLoseLife() {
        this.init();
        if (!this.ctx) return;
        let osc = this.ctx.createOscillator();
        let gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(300, this.ctx.currentTime);
        osc.frequency.linearRampToValueAtTime(100, this.ctx.currentTime + 0.4);
        
        gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.4);
        
        osc.start();
        osc.stop(this.ctx.currentTime + 0.4);
    }
}
const audio = new GameAudio();

// --- Real-time Massless Elastic Circular Membrane Physics Solver ---
class MembranePhysics {
    constructor(size = 21, tension = 30.0) {
        this.size = size;
        this.u = new Float32Array(size * size);
        this.tension = tension; // Physical membrane tension (T)
        this.cDamping = 0.0;    // PERFECT CONSERVATIVE STATIC CHECK
        this.rC = 0.0;          // Dynamic contact radius
        this.A = 0.0;           // Dynamic logarithmic coefficient A
    }

    solveContactRadius(zB, Rball) {
        if (zB >= Rball) return 0.0;

        let rC = Rball * 0.3; // Initial guess

        for (let iter = 0; iter < 6; iter++) {
            rC = Math.max(1e-4, Math.min(rC, Rball * 0.96));
            let S = Math.sqrt(Rball * Rball - rC * rC);

            let fVal = S + (rC * rC * Math.log(rC)) / S;
            let fPrime = (rC * Math.log(rC) / S) * (2.0 + (rC * rC) / (S * S));

            let diff = fVal - zB;
            if (Math.abs(diff) < 1e-5) break;

            rC = rC - diff / fPrime;
        }

        return Math.max(0.0, Math.min(rC, Rball * 0.96));
    }

    getElasticEnergy(ballRadius) {
        if (this.rC <= 0.0) return 0.0;
        let R = ballRadius;
        let S = Math.sqrt(Math.max(1e-9, R * R - this.rC * this.rC));
        let term1 = -0.5 * this.rC * this.rC;
        let term2 = -R * R * Math.log(S / R);
        let term3 = -(Math.pow(this.rC, 4) * Math.log(this.rC)) / (S * S);
        return Math.max(0.0, Math.PI * this.tension * (term1 + term2 + term3));
    }

    updatePhysicsState(ballPos, ballRadius, px, py) {
        let bxRel = ballPos.x - px;
        let byRel = ballPos.y - py;

        let bDist = Math.sqrt(bxRel * bxRel + byRel * byRel);
        if (bDist > 1.0) {
            this.rC = 0.0;
            this.A = 0.0;
            return;
        }

        this.rC = this.solveContactRadius(ballPos.z, ballRadius);

        if (this.rC > 0.0) {
            let S = Math.sqrt(ballRadius * ballRadius - this.rC * this.rC);
            this.A = (this.rC * this.rC) / S;
        } else {
            this.A = 0.0;
        }
    }

    updateMassless(ballPos, ballRadius, px, py) {
        this.u.fill(0.0);

        if (this.rC > 0.0) {
            let bxRel = ballPos.x - px;
            let byRel = ballPos.y - py;

            for (let i = 0; i < this.size; i++) {
                for (let j = 0; j < this.size; j++) {
                    let nxRel = -1.0 + (i / (this.size - 1)) * 2.0;
                    let nyRel = -1.0 + (j / (this.size - 1)) * 2.0;

                    let rNode = Math.sqrt(nxRel * nxRel + nyRel * nyRel);
                    if (rNode >= 1.0) {
                        this.u[i * this.size + j] = 0.0;
                        continue;
                    }

                    // Möbius Conformal Mapping
                    let numX = nxRel - bxRel;
                    let numY = nyRel - byRel;
                    let denX = 1.0 - (nxRel * bxRel + nyRel * byRel);
                    let denY = nxRel * byRel - nyRel * bxRel;

                    let d2 = (numX * numX + numY * numY) / (denX * denX + denY * denY);
                    let d = Math.sqrt(d2);

                    if (d <= this.rC) {
                        let rBall = Math.sqrt((nxRel - bxRel) * (nxRel - bxRel) + (nyRel - byRel) * (nyRel - byRel));
                        this.u[i * this.size + j] = ballPos.z - Math.sqrt(Math.max(0.0, ballRadius * ballRadius - rBall * rBall));
                    } else {
                        this.u[i * this.size + j] = this.A * Math.log(d);
                    }
                }
            }
        }
    }
}

// --- Game Engine Variables ---
let scene, camera, renderer;
let membraneGrid, membraneWire, paddleFrame;
let ballMesh, ballLight;
let bricks = [];
let particles = [];

// Arena Limits
const ARENA_X = 2.2;
const ARENA_Y = 2.2;
const ARENA_Z = 9.0;

// Pure Physical Simulation State
let gameStarted = true;

// Physics objects
const membraneSize = 41;
const physics = new MembranePhysics(membraneSize, 30.0); // T = 30.0
let ball = {
    pos: new THREE.Vector3(0.0, 0.0, 4.0), // Starts at concentric drop height 4.0
    vel: new THREE.Vector3(0.0, 0.0, 0.0), // strictly vertical
    radius: 0.5,
    mass: 1.0
};

// Paddle tracking (LOCKED AT (0,0))
let px = 0.0, py = 0.0;
let pVx = 0.0, pVy = 0.0;
let lastPx = 0.0, lastPy = 0.0;

// Raycasting (DEACTIVATED in static verification mode)
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const planeZ0 = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

// --- Initialization ---
function init() {
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x09090e, 0.035);

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, -6.5, 7.5);
    camera.lookAt(0, 0.8, 2.5);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    document.getElementById('game-container').appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
    scene.add(ambientLight);

    ballLight = new THREE.PointLight(0xffd700, 1.2, 8.0);
    scene.add(ballLight);

    const arenaGeom = new THREE.BoxGeometry(ARENA_X * 2, ARENA_Y * 2, ARENA_Z);
    const arenaEdges = new THREE.EdgesGeometry(arenaGeom);
    const arenaLine = new THREE.LineSegments(arenaEdges, new THREE.LineBasicMaterial({ color: 0x1f2330 }));
    arenaLine.position.set(0, 0, ARENA_Z / 2);
    scene.add(arenaLine);

    const floorGrid = new THREE.GridHelper(ARENA_X * 2, 8, 0x11131a, 0x11131a);
    floorGrid.rotation.x = Math.PI / 2;
    floorGrid.position.set(0, 0, 0);
    scene.add(floorGrid);

    const planeGeom = new THREE.PlaneGeometry(2, 2, membraneSize - 1, membraneSize - 1);
    const planeMat = new THREE.MeshBasicMaterial({
        color: 0x00bfff,
        transparent: true,
        opacity: 0.38,
        side: THREE.DoubleSide
    });
    membraneGrid = new THREE.Mesh(planeGeom, planeMat);
    scene.add(membraneGrid);

    const wireframeGeom = new THREE.WireframeGeometry(planeGeom);
    membraneWire = new THREE.LineSegments(wireframeGeom, new THREE.LineBasicMaterial({
        color: 0x00bfff,
        transparent: true,
        opacity: 0.8
    }));
    scene.add(membraneWire);

    // Circular Unit Frame Outline
    const circleGeom = new THREE.BufferGeometry();
    const circlePoints = [];
    for (let i = 0; i <= 60; i++) {
        let theta = (i / 60) * Math.PI * 2;
        circlePoints.push(new THREE.Vector3(Math.cos(theta), Math.sin(theta), 0));
    }
    circleGeom.setFromPoints(circlePoints);
    paddleFrame = new THREE.LineLoop(circleGeom, new THREE.LineBasicMaterial({ color: 0x32cd32, linewidth: 2 }));
    scene.add(paddleFrame);

    const ballGeom = new THREE.SphereGeometry(ball.radius, 32, 32);
    const ballMat = new THREE.MeshStandardMaterial({
        color: 0xffd700,
        emissive: 0xffa500,
        roughness: 0.1,
        metalness: 0.1
    });
    ballMesh = new THREE.Mesh(ballGeom, ballMat);
    scene.add(ballMesh);

    bricks = [];
}

function onMouseMove(event) {
    // IGNORED IN STATIC MODE
}

function onTouchMove(event) {
    // IGNORED IN STATIC MODE
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// startGame and restartGame removed for pure simulation

let lastTime = 0;

function animate(time) {
    requestAnimationFrame(animate);
    
    let dt = (time - lastTime) / 1000.0;
    lastTime = time;
    if (isNaN(dt)) dt = 0.016;
    dt = Math.min(dt, 0.05);

    if (gameStarted) {
        px = 0.0;
        py = 0.0;
        pVx = 0.0;
        pVy = 0.0;

        paddleFrame.position.set(px, py, 0);
        membraneGrid.position.set(px, py, 0);
        membraneWire.position.set(px, py, 0);

        // 3. Substepped Physics Engine (Strict Conservation of Energy & Momentum)
        const substeps = 20; // High substepping inside WebGL for perfect visual stability
        const subDt = dt / substeps;

        for (let step = 0; step < substeps; step++) {
            ball.vel.z -= 6.2 * subDt; // gravity
            ball.pos.addScaledVector(ball.vel, subDt);
            
            // CRITICAL: We MUST update the massless membrane contact radius INSIDE the substep loop
            // to prevent phase-lag artificial energy injection!
            physics.updatePhysicsState(ball.pos, ball.radius, px, py);

            // Floor Out of Bounds (Indefinite Bouncing Reset)
            if (ball.pos.z < -1.0) {
                ball.pos.set(0.0, 0.0, 4.0);
                ball.vel.set(0.0, 0.0, 0.0);
                break;
            }

            // 4. Stable Spring-Damper coupling with Circular Slope-Normal forces
            let bxRel = ball.pos.x - px;
            let byRel = ball.pos.y - py;
            let bDist = Math.sqrt(bxRel * bxRel + byRel * byRel);

            if (bDist <= 1.0) {
                if (physics.rC > 0.0) {
                    let S = Math.sqrt(Math.max(1e-4, ball.radius * ball.radius - physics.rC * physics.rC));

                    // Upward vertical force: F_z = 2*pi * T * r_c^2 / S
                    let forceZ = 2.0 * Math.PI * physics.tension * (physics.rC * physics.rC) / S;
                    let forceDamping = -physics.cDamping * ball.vel.z;
                    let forceTotal = forceZ + forceDamping;

                    forceTotal = Math.max(0.0, forceTotal);

                    // Slope normal direction
                    let slopeMag = physics.rC / S;
                    let nx = 0, ny = 0;
                    if (bDist > 1e-4) {
                        nx = -(bxRel / bDist) * slopeMag;
                        ny = -(byRel / bDist) * slopeMag;
                    }

                    // Apply force vectors
                    ball.vel.x += (forceTotal * nx / ball.mass) * subDt;
                    ball.vel.y += (forceTotal * ny / ball.mass) * subDt;
                    ball.vel.z += (forceTotal / ball.mass) * subDt;
                    
                    if (step === 0 && Math.abs(ball.vel.z) > 0.5) {
                        audio.playBounce();
                    }
                }
            }
        }

        // Evaluate the visual membrane heights exactly ONCE per graphics frame!
        physics.updateMassless(ball.pos, ball.radius, px, py);

        // Sync visual WebGL mesh heights after the full substepped integration is complete!
        const posAttr = membraneGrid.geometry.attributes.position;
        for (let i = 0; i < membraneSize; i++) {
            for (let j = 0; j < membraneSize; j++) {
                let nxRel = -1.0 + (i / (membraneSize - 1)) * 2.0;
                let nyRel = -1.0 + (j / (membraneSize - 1)) * 2.0;
                let vertexIndex = j * membraneSize + i;

                if (nxRel*nxRel + nyRel*nyRel >= 1.0) {
                    posAttr.setZ(vertexIndex, 0.0);
                    continue;
                }

                let idx = i * membraneSize + j;
                let val = physics.u[idx];
                posAttr.setZ(vertexIndex, val);
            }
        }
        posAttr.needsUpdate = true;
        membraneGrid.geometry.computeVertexNormals();

        ballMesh.position.copy(ball.pos);
        ballLight.position.copy(ball.pos);
    }

    // Calculate real-time physical energy components
    let ballSpeedSq = ball.vel.x * ball.vel.x + ball.vel.y * ball.vel.y + ball.vel.z * ball.vel.z;
    let ke = 0.5 * ball.mass * ballSpeedSq;
    let peGrav = ball.mass * 6.2 * ball.pos.z;
    let peElastic = physics.getElasticEnergy(ball.radius);
    let total = ke + peGrav + peElastic;
    updateEnergyPlot(ke, peGrav, peElastic, total);

    camera.position.set(0, -6.5, 7.5);
    camera.lookAt(0, 0.8, 2.5);

    renderer.render(scene, camera);
}

// --- Real-time Energy Plot Renderer ---
const energyCanvas = document.getElementById('energy-canvas');
const energyCtx = energyCanvas.getContext('2d');
const energyHistory = [];
const maxHistoryLength = 200;

function updateEnergyPlot(ke, peGrav, peElastic, total) {
    // Add to history
    energyHistory.push({ ke, peGrav, peElastic, total });
    if (energyHistory.length > maxHistoryLength) {
        energyHistory.shift();
    }

    // Update UI numbers
    document.getElementById('ke-val').innerText = ke.toFixed(3);
    document.getElementById('pe-grav-val').innerText = peGrav.toFixed(3);
    document.getElementById('pe-elastic-val').innerText = peElastic.toFixed(3);
    document.getElementById('total-val').innerText = total.toFixed(3);

    // Canvas size
    const w = energyCanvas.width;
    const h = energyCanvas.height;
    
    // Clear canvas
    energyCtx.fillStyle = 'rgba(9, 9, 14, 0.9)';
    energyCtx.fillRect(0, 0, w, h);

    // Draw background grid lines (every 25% of height)
    energyCtx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    energyCtx.lineWidth = 1;
    for (let yVal = 0.25; yVal < 1.0; yVal += 0.25) {
        let yPixel = h * yVal;
        energyCtx.beginPath();
        energyCtx.moveTo(0, yPixel);
        energyCtx.lineTo(w, yPixel);
        energyCtx.stroke();
    }

    if (energyHistory.length < 2) return;

    // Find min and max for scaling
    let maxVal = 26.0; // Fit the initial total energy 24.8 with a small head margin
    for (let entry of energyHistory) {
        maxVal = Math.max(maxVal, entry.ke, entry.peGrav, entry.peElastic, entry.total);
    }
    maxVal *= 1.05; // 5% top margin

    let minVal = -1.5; // Allow for negative gravity/elastic overshoot during high stretch
    for (let entry of energyHistory) {
        minVal = Math.min(minVal, entry.ke, entry.peGrav, entry.peElastic, entry.total);
    }
    minVal -= 1.0; // bottom margin

    let range = maxVal - minVal;
    if (range < 1.0) range = 1.0;

    function getX(index) {
        // Distribute points evenly across the canvas width
        return (index / (maxHistoryLength - 1)) * w;
    }

    function getY(value) {
        // Map values smoothly to canvas heights with 10px vertical margins
        return h - 10 - ((value - minVal) / range) * (h - 20);
    }

    // Draw series
    const channels = [
        { key: 'ke', color: '#ffd700', width: 2.0 },        // Yellow/Gold
        { key: 'peGrav', color: '#32cd32', width: 2.0 },    // Lime Green
        { key: 'peElastic', color: '#00bfff', width: 2.0 }, // Cyan
        { key: 'total', color: '#ff1493', width: 2.8 }     // Deep Pink / Fuchsia
    ];

    channels.forEach(ch => {
        energyCtx.beginPath();
        energyCtx.strokeStyle = ch.color;
        energyCtx.lineWidth = ch.width;

        // Apply a subtle neon shadow glow to the Total energy line to highlight verification
        if (ch.key === 'total') {
            energyCtx.shadowBlur = 6;
            energyCtx.shadowColor = ch.color;
        } else {
            energyCtx.shadowBlur = 0;
        }

        // Draw line segments
        energyCtx.moveTo(getX(0), getY(energyHistory[0][ch.key]));
        for (let i = 1; i < energyHistory.length; i++) {
            energyCtx.lineTo(getX(i), getY(energyHistory[i][ch.key]));
        }
        energyCtx.stroke();
    });

    // Reset shadow properties for next draw call
    energyCtx.shadowBlur = 0;
}

init();
requestAnimationFrame(animate);

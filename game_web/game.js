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

        let rC = Rball * 0.5; // Initial guess

        for (let iter = 0; iter < 12; iter++) {
            rC = Math.max(1e-7, Math.min(rC, Rball * 0.9999));
            let S = Math.sqrt(Rball * Rball - rC * rC);

            let fVal = S + (rC * rC * Math.log(rC)) / S;
            let fPrime = (rC * Math.log(rC) / S) * (2.0 + (rC * rC) / (S * S));

            let diff = fVal - zB;
            if (Math.abs(diff) < 1e-10) break;

            rC = rC - diff / fPrime;
        }

        return Math.max(0.0, Math.min(rC, Rball * 0.9999));
    }

    getElasticEnergy(ballRadius) {
        if (this.rC <= 0.0) return 0.0;
        let R = ballRadius;
        let S = Math.sqrt(Math.max(1e-15, R * R - this.rC * this.rC));
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
            let rB = Math.sqrt(bxRel * bxRel + byRel * byRel);

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
                        let uLog = this.A * Math.log(d);

                        // Smooth blending fix for off-center boundary tearing
                        let dxRel = nxRel - bxRel;
                        let dyRel = nyRel - byRel;
                        let distBall = Math.sqrt(dxRel * dxRel + dyRel * dyRel);
                        let ux = 1.0, uy = 0.0;
                        if (distBall > 1e-6) {
                            ux = dxRel / distBall;
                            uy = dyRel / distBall;
                        }

                        let projA = bxRel * ux + byRel * uy;
                        let rBallBoundary = this.rC * (1.0 - rB * rB) / (1.0 + this.rC * projA);

                        let uSphereB = ballPos.z - Math.sqrt(Math.max(0.0, ballRadius * ballRadius - rBallBoundary * rBallBoundary));
                        let uLogB = this.A * Math.log(this.rC);
                        let localJump = uSphereB - uLogB;

                        let s = (d < 1.0) ? (1.0 - d) / (1.0 - this.rC) : 0.0;
                        s = Math.max(0.0, Math.min(1.0, s));

                        this.u[i * this.size + j] = uLog + s * localJump;
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
let velocityArrow, accelerationArrow;
let bricks = [];
let particles = [];

// Interactive Camera Drag Rotation State
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };
let cameraTheta = -Math.PI / 2; // horizontal rotation
let cameraPhi = 0.65; // vertical tilt
let cameraRadius = 10.0;

window.addEventListener('mousedown', (e) => {
    isDragging = true;
    previousMousePosition = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mousemove', (e) => {
    if (isDragging) {
        let deltaX = e.clientX - previousMousePosition.x;
        let deltaY = e.clientY - previousMousePosition.y;
        cameraTheta -= deltaX * 0.005;
        cameraPhi = Math.max(0.05, Math.min(Math.PI / 2 - 0.05, cameraPhi - deltaY * 0.005));
    }
    previousMousePosition = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mouseup', () => {
    isDragging = false;
});

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
    pos: new THREE.Vector3(0.2, 0.1, 2.0), // Starts at off-center drop height 2.0
    vel: new THREE.Vector3(0.3, 0.2, -0.5), // 3D gentle initial velocity
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

    // --- Technical Plotly-style Bounding Axis Box (3.0 x 3.0 x 5.2) ---
    const boxGeom = new THREE.BoxGeometry(3.0, 3.0, 5.2);
    const boxEdges = new THREE.EdgesGeometry(boxGeom);
    const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0x44444c }));
    boxLine.position.set(0, 0, 1.4);
    scene.add(boxLine);

    // Bottom Grid Plane (at z = -1.2)
    const bottomGrid = new THREE.GridHelper(3.0, 6, 0x55555c, 0x222228);
    bottomGrid.rotation.x = Math.PI / 2;
    bottomGrid.position.set(0, 0, -1.2);
    scene.add(bottomGrid);

    // Back Grid Plane (at y = 1.5)
    const backGrid = new THREE.GridHelper(5.2, 10, 0x55555c, 0x222228);
    backGrid.position.set(0, 1.5, 1.4);
    backGrid.rotation.z = Math.PI / 2;
    backGrid.rotation.x = Math.PI / 2;
    scene.add(backGrid);

    // Left Grid Plane (at x = -1.5)
    const leftGrid = new THREE.GridHelper(5.2, 10, 0x55555c, 0x222228);
    leftGrid.position.set(-1.5, 0, 1.4);
    leftGrid.rotation.y = Math.PI / 2;
    leftGrid.rotation.z = Math.PI / 2;
    scene.add(leftGrid);

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

    // Concentric Cylinder Bounding Wall Outline (R_cyl = 1.45) in neon pink
    const cylGeom = new THREE.BufferGeometry();
    const cylPoints = [];
    // --- 3D Parametric Bounding Cylinder (R_cyl = 1.45) matching the Plotly scene ---
    const cylRadius = 1.45;
    const cylHeight = 5.2; // extending from z = -1.2 to z = 4.0
    const cylinderGeom = new THREE.CylinderGeometry(cylRadius, cylRadius, cylHeight, 40, 1, true);
    cylinderGeom.rotateX(Math.PI / 2); // align along Z axis
    const cylinderMat = new THREE.MeshBasicMaterial({
        color: 0xff1493,
        transparent: true,
        opacity: 0.05,
        side: THREE.DoubleSide
    });
    const cylinderMesh = new THREE.Mesh(cylinderGeom, cylinderMat);
    cylinderMesh.position.set(0, 0, 1.4); // center at z = 1.4
    scene.add(cylinderMesh);

    // --- Interactive 3D Arrow Helpers for Velocity & Acceleration ---
    velocityArrow = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(0, 0, 0),
        0.5,
        0xff4500, // orange-red
        0.15, // head length
        0.08  // head width
    );
    scene.add(velocityArrow);

    accelerationArrow = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(0, 0, 0),
        0.5,
        0x00ffff, // cyan
        0.15,
        0.08
    );
    scene.add(accelerationArrow);

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
    initBricks();
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('touchmove', onTouchMove, { passive: false });
}

function initBricks() {
    // Clear old bricks
    bricks.forEach(b => scene.remove(b.mesh));
    bricks = [];

    const colors = [0xff1493, 0x00bfff, 0x32cd32];
    for (let row = 0; row < 3; row++) {
        let color = colors[row % colors.length];
        for (let col = 0; col < 4; col++) {
            let bx = -0.75 + col * 0.5;
            let by = -0.5 + row * 0.5;
            let bz = 4.0 + row * 0.5;

            const brickGeom = new THREE.BoxGeometry(0.4, 0.4, 0.15);
            const brickMat = new THREE.MeshStandardMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.6,
                roughness: 0.2,
                metalness: 0.1
            });
            const mesh = new THREE.Mesh(brickGeom, brickMat);
            mesh.position.set(bx, by, bz);
            scene.add(mesh);

            bricks.push({
                mesh: mesh,
                x: bx,
                y: by,
                z: bz,
                size_x: 0.4,
                size_y: 0.4,
                size_z: 0.15,
                active: true
            });
        }
    }
}

function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const target = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(planeZ0, target)) {
        px = Math.max(-0.8, Math.min(0.8, target.x));
        py = Math.max(-0.8, Math.min(0.8, target.y));
    }
}

function onTouchMove(event) {
    if (event.touches.length > 0) {
        event.preventDefault();
        mouse.x = (event.touches[0].clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.touches[0].clientY / window.innerHeight) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const target = new THREE.Vector3();
        if (raycaster.ray.intersectPlane(planeZ0, target)) {
            px = Math.max(-0.8, Math.min(0.8, target.x));
            py = Math.max(-0.8, Math.min(0.8, target.y));
        }
    }
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
        // px and py follow the mouse/touch coordinates continuously via raycasting
        paddleFrame.position.set(px, py, 0);
        membraneGrid.position.set(px, py, 0);
        membraneWire.position.set(px, py, 0);

        // 3. Substepped Physics Engine in buttery-smooth 30% slow-motion!
        const dtPhysics = 0.3 * dt;
        const substeps = 20; // High substepping inside WebGL for perfect visual stability
        const subDt = dtPhysics / substeps;

        for (let step = 0; step < substeps; step++) {
            // 1. Compute forces at current pos
            let bxRel = ball.pos.x - px;
            let byRel = ball.pos.y - py;
            let bDist = Math.sqrt(bxRel * bxRel + byRel * byRel);
            physics.updatePhysicsState(ball.pos, ball.radius, px, py);

            let forceZ = 0.0;
            let forceX = 0.0;
            let forceY = 0.0;
            if (bDist <= 1.0 && physics.rC > 0.0) {
                let S = Math.sqrt(Math.max(1e-15, ball.radius * ball.radius - physics.rC * physics.rC));
                let FzCentered = 2.0 * Math.PI * physics.tension * (physics.rC * physics.rC) / S;
                let UeCentered = physics.getElasticEnergy(ball.radius);
                
                // Off-center potential scaling factor
                let rB2 = bxRel * bxRel + byRel * byRel;
                let factor = 1.0 / (1.0 - rB2);
                
                forceZ = FzCentered * factor;
                forceX = - (2.0 * bxRel / Math.pow(1.0 - rB2, 2)) * UeCentered;
                forceY = - (2.0 * byRel / Math.pow(1.0 - rB2, 2)) * UeCentered;
            }

            // 2. Update velocity (Symplectic Euler)
            ball.vel.x += (forceX / ball.mass) * subDt;
            ball.vel.y += (forceY / ball.mass) * subDt;
            ball.vel.z += (-6.2 + forceZ / ball.mass) * subDt;
            ball.vel.clampScalar(-25.0, 25.0);

            // 3. Update position (Symplectic Euler)
            ball.pos.addScaledVector(ball.vel, subDt);

            // 4. Concentric Cylinder constraint (R_cyl = 1.45) with clamping
            const RCyl = 1.45;
            let ballR = Math.sqrt(ball.pos.x * ball.pos.x + ball.pos.y * ball.pos.y);
            if (ballR >= RCyl - ball.radius) {
                let nx = ball.pos.x / ballR;
                let ny = ball.pos.y / ballR;
                let vDotN = ball.vel.x * nx + ball.vel.y * ny;
                if (vDotN > 0) {
                    ball.vel.x -= 2.0 * vDotN * nx;
                    ball.vel.y -= 2.0 * vDotN * ny;
                    // Project back to contact boundary ONLY when colliding!
                    ball.pos.x = (RCyl - ball.radius) * nx;
                    ball.pos.y = (RCyl - ball.radius) * ny;
                }
            }

            // 5. Rigid Circular Frame Ring (R_frame = 1.0) with clamping
            const RFrame = 1.0;
            ballR = Math.sqrt(ball.pos.x * ball.pos.x + ball.pos.y * ball.pos.y);
            let pClosestX = 0.0, pClosestY = 0.0;
            if (ballR > 1e-6) {
                pClosestX = RFrame * (ball.pos.x / ballR);
                pClosestY = RFrame * (ball.pos.y / ballR);
            } else {
                pClosestX = RFrame;
                pClosestY = 0.0;
            }

            let dx = ball.pos.x - pClosestX;
            let dy = ball.pos.y - pClosestY;
            let dz = ball.pos.z - 0.0;
            let dRing = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (dRing <= ball.radius) {
                let nColX = dx / dRing;
                let nColY = dy / dRing;
                let nColZ = dz / dRing;
                let vDotCol = ball.vel.x * nColX + ball.vel.y * nColY + ball.vel.z * nColZ;
                if (vDotCol < 0) {
                    ball.vel.x -= 2.0 * vDotCol * nColX;
                    ball.vel.y -= 2.0 * vDotCol * nColY;
                    ball.vel.z -= 2.0 * vDotCol * nColZ;
                    
                    if (step === 0 && Math.abs(ball.vel.z) > 0.5) {
                        audio.playBounce();
                    }
                    // Project back to contact boundary ONLY when colliding!
                    ball.pos.x = pClosestX + ball.radius * nColX;
                    ball.pos.y = pClosestY + ball.radius * nColY;
                    ball.pos.z = ball.radius * nColZ;
                }
            }

            // --- Sphere-AABB Brick Collision Detection & Resolution ---
            for (let b of bricks) {
                if (b.active) {
                    let b_dx = Math.max(b.x - b.size_x/2, Math.min(ball.pos.x, b.x + b.size_x/2));
                    let b_dy = Math.max(b.y - b.size_y/2, Math.min(ball.pos.y, b.y + b.size_y/2));
                    let b_dz = Math.max(b.z - b.size_z/2, Math.min(ball.pos.z, b.z + b.size_z/2));

                    let b_distX = ball.pos.x - b_dx;
                    let b_distY = ball.pos.y - b_dy;
                    let b_distZ = ball.pos.z - b_dz;
                    let b_dist = Math.sqrt(b_distX*b_distX + b_distY*b_distY + b_distZ*b_distZ);

                    if (b_dist <= ball.radius) {
                        let ox = Math.abs(b_distX);
                        let oy = Math.abs(b_distY);
                        let oz = Math.abs(b_distZ);

                        if (ox > oy && ox > oz) {
                            ball.vel.x = -ball.vel.x;
                        } else if (oy > ox && oy > oz) {
                            ball.vel.y = -ball.vel.y;
                        } else {
                            ball.vel.z = -ball.vel.z;
                        }

                        b.active = false;
                        scene.remove(b.mesh);
                        audio.playHitBrick();
                        break;
                    }
                }
            }

            // Endless reset: if all bricks are destroyed, spawn them again!
            let anyActive = false;
            for (let b of bricks) {
                if (b.active) anyActive = true;
            }
            if (!anyActive) {
                initBricks();
            }

            // 6. Floor out-of-bounds reset (Reset to gentle physical state)
            if (ball.pos.z < -1.0) {
                ball.pos.set(0.2, 0.1, 2.0);
                ball.vel.set(0.3, 0.2, -0.5);
                break;
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

        // --- Update Velocity & Acceleration 3D Vector Arrows at Center of the Ball ---
        velocityArrow.position.copy(ball.pos);
        let vDir = ball.vel.clone().normalize();
        let vLen = ball.vel.length() * 0.2; // scale length (same as Plotly)
        if (vLen > 0.01) {
            velocityArrow.setDirection(vDir);
            velocityArrow.setLength(vLen, 0.12, 0.06);
            velocityArrow.visible = true;
        } else {
            velocityArrow.visible = false;
        }

        // Compute local acceleration vector based on active forces
        let bxRel = ball.pos.x - px;
        let byRel = ball.pos.y - py;
        let bDist = Math.sqrt(bxRel * bxRel + byRel * byRel);
        physics.updatePhysicsState(ball.pos, ball.radius, px, py);
        let fZ = 0.0, fX = 0.0, fY = 0.0;
        if (bDist <= 1.0 && physics.rC > 0.0) {
            let S = Math.sqrt(Math.max(1e-15, ball.radius * ball.radius - physics.rC * physics.rC));
            let FzCentered = 2.0 * Math.PI * physics.tension * (physics.rC * physics.rC) / S;
            let UeCentered = physics.getElasticEnergy(ball.radius);
            let rB2 = bxRel * bxRel + byRel * byRel;
            let factor = 1.0 / (1.0 - rB2);
            fZ = FzCentered * factor;
            fX = - (2.0 * bxRel / Math.pow(1.0 - rB2, 2)) * UeCentered;
            fY = - (2.0 * byRel / Math.pow(1.0 - rB2, 2)) * UeCentered;
        }
        let aVec = new THREE.Vector3(fX / ball.mass, fY / ball.mass, -6.2 + fZ / ball.mass);
        accelerationArrow.position.copy(ball.pos);
        let aDir = aVec.clone().normalize();
        let aLen = aVec.length() * 0.02; // scale length (same as Plotly)
        if (aLen > 0.01) {
            accelerationArrow.setDirection(aDir);
            accelerationArrow.setLength(aLen, 0.12, 0.06);
            accelerationArrow.visible = true;
        } else {
            accelerationArrow.visible = false;
        }
    }

    // Calculate real-time physical energy components
    let ballSpeedSq = ball.vel.x * ball.vel.x + ball.vel.y * ball.vel.y + ball.vel.z * ball.vel.z;
    let ke = 0.5 * ball.mass * ballSpeedSq;
    let peGrav = ball.mass * 6.2 * ball.pos.z;
    let rBEnd = Math.sqrt(ball.pos.x * ball.pos.x + ball.pos.y * ball.pos.y);
    let peElastic = 0.0;
    if (rBEnd <= 1.0) {
        let rBEnd2 = ball.pos.x * ball.pos.x + ball.pos.y * ball.pos.y;
        let factorEnd = 1.0 / (1.0 - rBEnd2);
        peElastic = physics.getElasticEnergy(ball.radius) * factorEnd;
    }
    let total = ke + peGrav + peElastic;
    updateEnergyPlot(ke, peGrav, peElastic, total);

    camera.position.x = cameraRadius * Math.cos(cameraTheta) * Math.cos(cameraPhi);
    camera.position.y = cameraRadius * Math.sin(cameraTheta) * Math.cos(cameraPhi);
    camera.position.z = cameraRadius * Math.sin(cameraPhi);
    camera.lookAt(0, 0.8, 1.5);

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

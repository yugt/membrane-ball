import numpy as np
from physics import MembranePhysics, Ball3D

def run_energy_verifier(tension=30.0, damping=0.0, gravity=6.2, initial_z=4.0, duration=5.0):
    print("=" * 70)
    print(f"ENERGY CONSERVATION VERIFIER (Damping = {damping})")
    print("=" * 70)
    
    # Initialize physics
    membrane = MembranePhysics(size=21, tension=tension)
    membrane.c_damping = damping
    
    # Ball parameters
    ball = Ball3D(x=0.0, y=0.0, z=initial_z, radius=0.5, mass=1.0)
    ball.vel = np.array([0.0, 0.0, 0.0], dtype=float)
    
    dt = 1.0 / 60.0
    substeps = 20
    sub_dt = dt / substeps
    
    total_time = 0.0
    steps = int(duration / dt)
    
    # Stored elastic potential energy in the membrane
    # (starts at 0 since ball is in the air)
    elastic_energy = 0.0
    
    # Initial total mechanical energy
    # E_0 = K_0 + Ug_0 + Ue_0
    initial_energy = 0.5 * ball.mass * (ball.vel[2]**2) + ball.mass * gravity * ball.pos[2] + elastic_energy
    print(f"Initial Total Energy E_0 = {initial_energy:.6f} J")
    print("-" * 70)
    print(f"{'Time (s)':<10}{'Ball Z':<10}{'Kinetic (K)':<15}{'Grav Pot (Ug)':<15}{'Elastic (Ue)':<15}{'Total Energy (E)':<15}")
    print("-" * 70)
    
    print_interval = int(0.1 / sub_dt) # print every 0.1 seconds
    step_count = 0
    
    for step in range(steps):
        for _ in range(substeps):
            z_old = ball.pos[2]
            
            # 1. Update ball position
            ball.update(sub_dt, gravity=gravity)
            
            # 2. Update membrane shape
            membrane.update_massless(ball.pos, ball.radius, 0.0, 0.0)
            
            # 3. Calculate contact force
            force_z = 0.0
            b_dist = np.sqrt(ball.pos[0]**2 + ball.pos[1]**2)
            if b_dist <= 1.0 and membrane.r_c > 0.0:
                S = np.sqrt(max(1e-6, ball.radius**2 - membrane.r_c**2))
                force_z = 2.0 * np.pi * membrane.tension * (membrane.r_c**2) / S
                
                # Apply conservative force (no damping for clean energy tracking)
                ball.vel[2] += (force_z / ball.mass) * sub_dt
            
            # 4. Integrate Elastic Potential Energy (Work-Energy Theorem):
            # The work done by the ball to compress the membrane is stored as potential energy:
            # dUe = - F_contact * dz
            # Since F_contact is upward (+z) and compression is downward (-z), 
            # compressing the membrane increases its potential energy.
            z_new = ball.pos[2]
            dz = z_new - z_old
            elastic_energy -= force_z * dz
            
            # Force elastic energy to be strictly positive
            elastic_energy = max(0.0, elastic_energy)
            
            # 5. Compute real-time mechanical energies
            kinetic = 0.5 * ball.mass * (ball.vel[2]**2)
            gravitational = ball.mass * gravity * ball.pos[2]
            total_energy = kinetic + gravitational + elastic_energy
            
            if step_count % print_interval == 0:
                print(f"{total_time:<10.3f}{ball.pos[2]:<10.3f}{kinetic:<15.6f}{gravitational:<15.6f}{elastic_energy:<15.6f}{total_energy:<15.6f}")
                
            step_count += 1
            total_time += sub_dt

if __name__ == "__main__":
    # Run the energy verifier for a full bounce cycle
    run_energy_verifier(tension=30.0, damping=0.0, gravity=6.2, initial_z=4.0, duration=2.5)

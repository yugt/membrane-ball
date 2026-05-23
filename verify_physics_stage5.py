import numpy as np

class MembranePhysics:
    def __init__(self, tension=30.0):
        self.tension = tension
        self.r_c = 0.0

    def solve_contact_radius(self, z_b, R_ball):
        if z_b >= R_ball:
            return 0.0
        r_c = R_ball * 0.3
        for _ in range(6):
            r_c = np.clip(r_c, 1e-4, R_ball * 0.96)
            S = np.sqrt(R_ball**2 - r_c**2)
            f_val = S + (r_c**2 * np.log(r_c)) / S
            f_prime = (r_c * np.log(r_c) / S) * (2.0 + (r_c**2) / (S**2))
            diff = f_val - z_b
            if abs(diff) < 1e-5:
                break
            r_c = r_c - diff / f_prime
        return np.clip(r_c, 0.0, R_ball * 0.96)

    def update_physics_state(self, ball_pos, ball_radius):
        z_b = ball_pos[2]
        self.r_c = self.solve_contact_radius(z_b, ball_radius)

    def get_elastic_energy(self, ball_radius):
        if self.r_c <= 0.0:
            return 0.0
        R = ball_radius
        S = np.sqrt(max(1e-9, R**2 - self.r_c**2))
        term1 = -0.5 * self.r_c**2
        term2 = -R**2 * np.log(S / R)
        term3 = -(self.r_c**4 * np.log(self.r_c)) / (S**2)
        return max(0.0, np.pi * self.tension * (term1 + term2 + term3))

def run_test():
    # Setup state
    tension = 30.0
    gravity = 6.2
    ball_mass = 1.0
    ball_radius = 0.5
    
    pos = np.array([0.2, 0.1, 4.0], dtype=float)
    vel = np.array([1.5, 1.0, 0.0], dtype=float)
    
    membrane = MembranePhysics(tension=tension)
    
    dt = 0.016
    substeps = 20
    sub_dt = dt / substeps
    
    # Track initial energy
    membrane.update_physics_state(pos, ball_radius)
    r_b_init2 = pos[0]**2 + pos[1]**2
    r_b_init = np.sqrt(r_b_init2)
    factor_init = 1.0 / (1.0 - r_b_init2) if r_b_init < 1.0 else 1.0
    ke = 0.5 * ball_mass * np.sum(vel**2)
    pe_grav = ball_mass * gravity * pos[2]
    pe_elastic = membrane.get_elastic_energy(ball_radius) * factor_init
    initial_energy = ke + pe_grav + pe_elastic
    
    print(f"Initial Energy: {initial_energy:.6f} J")
    print("Simulating 5000 substeps (approx. 4 seconds of real-time movement)...")
    
    max_dev = 0.0
    cyl_bounces = 0
    ring_bounces = 0
    membrane_contacts = 0
    
    for step in range(5000):
        # 1. Gravity acceleration
        vel[2] -= gravity * sub_dt
        
        # 2. Position integration
        pos += vel * sub_dt
        
        # 3. Concentric Cylinder constraint (Velocity reflection and clamping)
        R_cyl = 0.6
        ball_r = np.sqrt(pos[0]**2 + pos[1]**2)
        if ball_r >= R_cyl - ball_radius:
            nx = pos[0] / ball_r
            ny = pos[1] / ball_r
            v_dot_n = vel[0] * nx + vel[1] * ny
            if v_dot_n > 0:
                vel[0] -= 2.0 * v_dot_n * nx
                vel[1] -= 2.0 * v_dot_n * ny
                cyl_bounces += 1
                pos[0] = (R_cyl - ball_radius) * nx
                pos[1] = (R_cyl - ball_radius) * ny
        
        # 4. Rigid Circular Frame Ring (Velocity reflection only, no position clamping! - DISABLED FOR ISOLATION TEST)
        R_frame = 1.0
        r_b = np.sqrt(pos[0]**2 + pos[1]**2)
        if r_b > 1e-6:
            p_closest_x = R_frame * (pos[0] / r_b)
            p_closest_y = R_frame * (pos[1] / r_b)
        else:
            p_closest_x = R_frame
            p_closest_y = 0.0
            
        dx = pos[0] - p_closest_x
        dy = pos[1] - p_closest_y
        dz = pos[2] - 0.0
        d_ring = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Ring collisions disabled for this isolation check
        if False and d_ring <= ball_radius:
            n_col_x = dx / d_ring
            n_col_y = dy / d_ring
            n_col_z = dz / d_ring
            v_dot_col = vel[0] * n_col_x + vel[1] * n_col_y + vel[2] * n_col_z
            if v_dot_col < 0:
                vel[0] -= 2.0 * v_dot_col * n_col_x
                vel[1] -= 2.0 * v_dot_col * n_col_y
                vel[2] -= 2.0 * v_dot_col * n_col_z
                ring_bounces += 1
        
        # 5. Membrane active force coupling (only when inside frame)
        membrane.update_physics_state(pos, ball_radius)
        if r_b <= 1.0 and membrane.r_c > 0.0:
            membrane_contacts += 1
            Ue_centered = membrane.get_elastic_energy(ball_radius)
            S = np.sqrt(max(1e-4, ball_radius**2 - membrane.r_c**2))
            Fz_centered = 2.0 * np.pi * tension * (membrane.r_c**2) / S
            
            r_b2 = pos[0]**2 + pos[1]**2
            factor = 1.0 / (1.0 - r_b2) if r_b < 1.0 else 1.0
            
            force_z = Fz_centered * factor
            force_x = - (2.0 * pos[0] / (1.0 - r_b2)**2) * Ue_centered if Ue_centered > 0.0 else 0.0
            force_y = - (2.0 * pos[1] / (1.0 - r_b2)**2) * Ue_centered if Ue_centered > 0.0 else 0.0
                
            vel[0] += (force_x / ball_mass) * sub_dt
            vel[1] += (force_y / ball_mass) * sub_dt
            vel[2] += (force_z / ball_mass) * sub_dt
            
        # 6. Calculate total energy and verify
        ke = 0.5 * ball_mass * np.sum(vel**2)
        pe_grav = ball_mass * gravity * pos[2]
        r_b_end2 = pos[0]**2 + pos[1]**2
        r_b_end = np.sqrt(r_b_end2)
        if r_b_end <= 1.0:
            factor_end = 1.0 / (1.0 - r_b_end2)
            pe_elastic = membrane.get_elastic_energy(ball_radius) * factor_end
        else:
            pe_elastic = 0.0
        current_energy = ke + pe_grav + pe_elastic
        
        dev = abs(current_energy - initial_energy)
        max_dev = max(max_dev, dev)

    print("\n--- Numerical Results ---")
    print(f"Cylinder Bounces: {cyl_bounces}")
    print(f"Rigid Ring Bounces: {ring_bounces}")
    print(f"Membrane Stretches: {membrane_contacts}")
    print(f"Maximum Energy Deviation: {max_dev:.8f} J")
    print(f"Relative Energy Leak: {(max_dev / initial_energy) * 100:.6f}%")

if __name__ == "__main__":
    run_test()

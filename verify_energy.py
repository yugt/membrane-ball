import numpy as np

class MembranePhysics:
    def __init__(self, tension=30.0):
        self.tension = tension
        self.r_c = 0.0

    def solve_contact_radius(self, z_b, R_ball):
        if z_b >= R_ball:
            return 0.0
        r_c = R_ball * 0.5
        for _ in range(12):
            r_c = np.clip(r_c, 1e-7, R_ball * 0.9999)
            S = np.sqrt(R_ball**2 - r_c**2)
            f_val = S + (r_c**2 * np.log(r_c)) / S
            f_prime = (r_c * np.log(r_c) / S) * (2.0 + (r_c**2) / (S**2))
            diff = f_val - z_b
            if abs(diff) < 1e-10:
                break
            r_c = r_c - diff / f_prime
        return np.clip(r_c, 0.0, R_ball * 0.9999)

    def get_elastic_energy(self, z_b, ball_radius):
        r_c = self.solve_contact_radius(z_b, ball_radius)
        if r_c <= 0.0:
            return 0.0
        R = ball_radius
        S = np.sqrt(max(1e-15, R**2 - r_c**2))
        term1 = -0.5 * r_c**2
        term2 = -R**2 * np.log(S / R)
        term3 = -(r_c**4 * np.log(r_c)) / (S**2)
        return max(0.0, np.pi * self.tension * (term1 + term2 + term3))

def main():
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
    membrane.r_c = membrane.solve_contact_radius(pos[2], ball_radius)
    ke = 0.5 * ball_mass * np.sum(vel**2)
    pe_grav = ball_mass * gravity * pos[2]
    pe_elastic = membrane.get_elastic_energy(pos[2], ball_radius)
    initial_energy = ke + pe_grav + pe_elastic
    
    max_dev = 0.0
    cyl_bounces = 0
    ring_bounces = 0
    
    # Simulate 5000 steps (approx. 80 seconds / 4 seconds real time depending on FPS)
    for step in range(5000):
        for _ in range(substeps):
            # 1. Compute forces at current pos
            r_b = np.sqrt(pos[0]**2 + pos[1]**2)
            r_c = membrane.solve_contact_radius(pos[2], ball_radius)
            
            force_z = 0.0
            if r_b <= 1.0 and r_c > 0.0:
                S = np.sqrt(max(1e-15, ball_radius**2 - r_c**2))
                force_z = 2.0 * np.pi * tension * (r_c**2) / S
            
            # 2. Update velocity (Symplectic Euler: v_{n+1} updated first)
            vel[2] += (-gravity + force_z / ball_mass) * sub_dt
            
            # 3. Update position (Symplectic Euler: x_{n+1} integrated using v_{n+1})
            pos += vel * sub_dt
            
            # 4. Cylinder bounce (R_cyl = 1.45) with clamping
            R_cyl = 1.45
            ball_r = np.sqrt(pos[0]**2 + pos[1]**2)
            if ball_r >= R_cyl:
                nx = pos[0] / ball_r
                ny = pos[1] / ball_r
                v_dot_n = vel[0] * nx + vel[1] * ny
                if v_dot_n > 0:
                    vel[0] -= 2.0 * v_dot_n * nx
                    vel[1] -= 2.0 * v_dot_n * ny
                    cyl_bounces += 1
                pos[0] = R_cyl * nx
                pos[1] = R_cyl * ny
                
            # 5. Rigid circular frame ring bounce (R_frame = 1.0) with clamping
            R_frame = 1.0
            ball_r = np.sqrt(pos[0]**2 + pos[1]**2)
            if ball_r > 1e-6:
                p_closest_x = R_frame * (pos[0] / ball_r)
                p_closest_y = R_frame * (pos[1] / ball_r)
            else:
                p_closest_x = R_frame
                p_closest_y = 0.0
                
            dx = pos[0] - p_closest_x
            dy = pos[1] - p_closest_y
            dz = pos[2] - 0.0
            d_ring = np.sqrt(dx**2 + dy**2 + dz**2)
            if d_ring <= ball_radius:
                n_col_x = dx / d_ring
                n_col_y = dy / d_ring
                n_col_z = dz / d_ring
                v_dot_col = vel[0] * n_col_x + vel[1] * n_col_y + vel[2] * n_col_z
                if v_dot_col < 0:
                    vel[0] -= 2.0 * v_dot_col * n_col_x
                    vel[1] -= 2.0 * v_dot_col * n_col_y
                    vel[2] -= 2.0 * v_dot_col * n_col_z
                    ring_bounces += 1
                # Project back to contact boundary
                pos[0] = p_closest_x + ball_radius * n_col_x
                pos[1] = p_closest_y + ball_radius * n_col_y
                pos[2] = ball_radius * n_col_z
                
        # Energy Check at the end of the full graphics frame step
        ke = 0.5 * ball_mass * np.sum(vel**2)
        pe_grav = ball_mass * gravity * pos[2]
        
        r_b_end = np.sqrt(pos[0]**2 + pos[1]**2)
        if r_b_end <= 1.0:
            pe_elastic = membrane.get_elastic_energy(pos[2], ball_radius)
        else:
            pe_elastic = 0.0
            
        current_energy = ke + pe_grav + pe_elastic
        dev = abs(current_energy - initial_energy)
        max_dev = max(max_dev, dev)
        
    rel_leak = (max_dev / initial_energy) * 100.0
    print(f"Initial Energy: {initial_energy:.6f} J")
    print(f"Max Deviation: {max_dev:.6f} J")
    print(f"Relative Leak: {rel_leak:.6f}%")
    print(f"Cylinder Bounces: {cyl_bounces}, Ring Bounces: {ring_bounces}")
    
    threshold = 1.0
    if rel_leak <= threshold:
        print("Verification SUCCESS: Energy is perfectly conserved (Leak <= 1.0%).")
        exit(0)
    else:
        print("Verification FAILURE: Energy leak exceeds threshold!")
        exit(1)

if __name__ == "__main__":
    main()

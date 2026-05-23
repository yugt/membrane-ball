import numpy as np

class MembranePhysics:
    def __init__(self, size=21, tension=30.0):
        self.size = size
        self.u = np.zeros((size, size))
        self.tension = tension      # Physical membrane tension (T)
        self.c_damping = 8.0        # Kelvin-Voigt damping coefficient (c)
        self.r_c = 0.0              # Current contact radius
        self.A = 0.0                # Current logarithmic coefficient
        
    def solve_contact_radius(self, z_b, R_ball):
        # Solve the smooth contact transcendental equation for r_c:
        # f(r_c) = sqrt(R^2 - r_c^2) + (r_c^2 * ln(r_c)) / sqrt(R^2 - r_c^2) = z_b
        # Newton's method is extremely fast and exact.
        
        # If ball is above the contact limit, there is no contact
        if z_b >= R_ball:
            return 0.0
            
        # Initial guess: start near the center
        r_c = R_ball * 0.3
        
        for _ in range(6):
            r_c = np.clip(r_c, 1e-4, R_ball * 0.96)
            S = np.sqrt(R_ball**2 - r_c**2)
            
            # f(r_c)
            f_val = S + (r_c**2 * np.log(r_c)) / S
            
            # f'(r_c)
            f_prime = (r_c * np.log(r_c) / S) * (2.0 + (r_c**2) / (S**2))
            
            diff = f_val - z_b
            if abs(diff) < 1e-5:
                break
                
            # Newton step
            r_c = r_c - diff / f_prime
            
        return np.clip(r_c, 0.0, R_ball * 0.96)

    def get_elastic_energy(self, ball_radius):
        if self.r_c <= 0.0:
            return 0.0
        R = ball_radius
        S = np.sqrt(max(1e-9, R**2 - self.r_c**2))
        term1 = -0.5 * self.r_c**2
        term2 = -R**2 * np.log(S / R)
        term3 = -(self.r_c**4 * np.log(self.r_c)) / (S**2)
        return max(0.0, np.pi * self.tension * (term1 + term2 + term3))

    def update_physics_state(self, ball_pos, ball_radius, px, py):
        # Ball relative coordinates to paddle center
        bx_rel = ball_pos[0] - px
        by_rel = ball_pos[1] - py
        
        # Check if the ball center is within the unit circle frame (radius 1.0)
        b_dist = np.sqrt(bx_rel**2 + by_rel**2)
        if b_dist > 1.0:
            self.r_c = 0.0
            self.A = 0.0
            return
            
        z_b = ball_pos[2]
        
        # Solve for contact radius r_c
        self.r_c = self.solve_contact_radius(z_b, ball_radius)
        
        if self.r_c > 0.0:
            # Coefficient A matching slopes smoothly: A = r_c^2 / sqrt(R^2 - r_c^2)
            S = np.sqrt(ball_radius**2 - self.r_c**2)
            self.A = (self.r_c**2) / S
        else:
            self.A = 0.0

    def update_massless(self, ball_pos, ball_radius, px, py):
        # Reset grid to 0
        self.u.fill(0.0)
        
        if self.r_c > 0.0:
            bx_rel = ball_pos[0] - px
            by_rel = ball_pos[1] - py
            z_b = ball_pos[2]
            
            # Evaluate exact shape across the grid
            for i in range(self.size):
                for j in range(self.size):
                    # Node coordinates relative to paddle center [-1.0, 1.0]
                    nx_rel = -1.0 + (i / (self.size - 1)) * 2.0
                    ny_rel = -1.0 + (j / (self.size - 1)) * 2.0
                    
                    # Boundary condition: circular frame of radius 1.0
                    r_node = np.sqrt(nx_rel**2 + ny_rel**2)
                    if r_node >= 1.0:
                        self.u[i, j] = 0.0
                        continue
                        
                    # --- Exact Möbius Conformal Mapping ---
                    # Maps off-center contact point to the origin to solve Laplace equation exactly
                    num_x = nx_rel - bx_rel
                    num_y = ny_rel - by_rel
                    den_x = 1.0 - (nx_rel * bx_rel + ny_rel * by_rel)
                    den_y = nx_rel * by_rel - ny_rel * bx_rel
                    
                    d2 = (num_x**2 + num_y**2) / (den_x**2 + den_y**2)
                    d = np.sqrt(d2)
                    
                    # Inside the mapped contact patch: conforms to the sphere bottom
                    if d <= self.r_c:
                        r_ball = np.sqrt((nx_rel - bx_rel)**2 + (ny_rel - by_rel)**2)
                        self.u[i, j] = z_b - np.sqrt(max(0.0, ball_radius**2 - r_ball**2))
                    else:
                        # Outside contact patch: exact harmonic logarithmic funnel
                        self.u[i, j] = self.A * np.log(d)
        else:
            self.A = 0.0


class Ball3D:
    def __init__(self, x=0.0, y=0.0, z=4.0, radius=0.15, mass=1.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.radius = radius
        self.mass = mass
        
    def update(self, dt, gravity=9.8):
        # Apply gravity
        self.vel[2] -= gravity * dt
        
        # Integrate position
        self.pos += self.vel * dt
        
        # Max terminal velocity
        self.vel = np.clip(self.vel, -25.0, 25.0)

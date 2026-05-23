import pygame
import numpy as np
import sys
from physics import MembranePhysics, Ball3D

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 1024, 768
FPS = 60
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Membrane Breakout 3D (Massless Membrane - LOCKED FRAME VERIFICATION)")
CLOCK = pygame.time.Clock()

# Color Palette (Premium Sleek Neon)
COLOR_BG = (15, 15, 22)
COLOR_WALLS = (40, 45, 60)
COLOR_PADDLE_FRAME = (50, 220, 120)       # Glowing green
COLOR_BALL = (255, 230, 80)               # Vibrant yellow/gold
COLOR_TEXT = (240, 240, 250)
COLOR_MEMBRANE_DEFAULT = (0, 170, 255)    # Neon cyan
COLOR_MEMBRANE_STRETCH = (255, 50, 100)   # Neon pink/red

# Game State
game_started = True

# Arena limits
ARENA_X = 2.0
ARENA_Y = 2.0
ARENA_Z = 8.0

# 3D projection parameters
CAMERA_ROTATION = 45.0 * np.pi / 180.0
CAMERA_TILT = 35.0 * np.pi / 180.0
SCALE = 140.0

def project(x, y, z):
    cos_r = np.cos(CAMERA_ROTATION)
    sin_r = np.sin(CAMERA_ROTATION)
    rx = x * cos_r - y * sin_r
    ry = x * sin_r + y * cos_r
    
    cos_t = np.cos(CAMERA_TILT)
    sin_t = np.sin(CAMERA_TILT)
    
    px = rx * SCALE
    py = (ry * cos_t - z * sin_t) * SCALE
    
    screen_x = int(WIDTH // 2 + px)
    screen_y = int(HEIGHT // 2 + py + 120)
    return screen_x, screen_y

class Brick3D:
    def __init__(self, x, y, z, color, size_x=0.5, size_y=0.5, size_z=0.2):
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.active = True

    def get_vertices(self):
        dx, dy, dz = self.size_x / 2, self.size_y / 2, self.size_z / 2
        return [
            (self.x - dx, self.y - dy, self.z - dz),
            (self.x + dx, self.y - dy, self.z - dz),
            (self.x + dx, self.y + dy, self.z - dz),
            (self.x - dx, self.y + dy, self.z - dz),
            (self.x - dx, self.y - dy, self.z + dz),
            (self.x + dx, self.y - dy, self.z + dz),
            (self.x + dx, self.y + dy, self.z + dz),
            (self.x - dx, self.y + dy, self.z + dz),
        ]

    def draw(self, surface):
        if not self.active:
            return
        vertices = self.get_vertices()
        proj_vertices = [project(*v) for v in vertices]
        
        faces = [
            (0, 1, 2, 3), # Bottom
            (4, 5, 6, 7), # Top
            (0, 1, 5, 4), # Front-left
            (1, 2, 6, 5), # Front-right
            (2, 3, 7, 6), # Back-right
            (3, 0, 4, 7)  # Back-left
        ]
        
        for face in faces:
            points = [proj_vertices[i] for i in face]
            pygame.draw.polygon(surface, self.color, points)
            pygame.draw.polygon(surface, (255, 255, 255), points, 1)

class Particle3D:
    def __init__(self, x, y, z, color):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.random.uniform(-3.0, 3.0, 3)
        self.vel[2] = np.random.uniform(2.0, 6.0)
        self.color = color
        self.life = 1.0
        self.gravity = 8.0

    def update(self, dt):
        self.vel[2] -= self.gravity * dt
        self.pos += self.vel * dt
        self.life -= dt * 1.5

    def draw(self, surface):
        if self.life <= 0:
            return
        px, py = project(*self.pos)
        r = max(1, int(4 * self.life))
        c = tuple(max(0, min(255, int(channel * self.life))) for channel in self.color)
        pygame.draw.circle(surface, c, (px, py), r)

def init_bricks():
    return []

# Real-time Physical Energy Charting
energy_history = []
max_history_len = 200

def draw_physical_verifier(surface, ke, pe_grav, pe_elastic, total):
    global energy_history
    energy_history.append({'ke': ke, 'pe_grav': pe_grav, 'pe_elastic': pe_elastic, 'total': total})
    if len(energy_history) > max_history_len:
        energy_history.pop(0)
        
    width, height = surface.get_size()
    
    # Draw Background Panel in bottom-left
    panel_w, panel_h = 360, 240
    panel_x, panel_y = 30, height - panel_h - 30
    
    # Semitransparent glassmorphic card surface
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((18, 18, 30, 205))  # semitransparent dark card
    
    # Draw glowing cyan border
    pygame.draw.rect(panel_surf, (0, 191, 255, 60), (0, 0, panel_w, panel_h), 1, border_radius=16)
    
    # Title
    font_title = pygame.font.SysFont("Outfit, Arial, Helvetica", 14, bold=True)
    txt_title = font_title.render("PHYSICAL VERIFIER (ENERGY CONSERVATION)", True, (138, 159, 196))
    panel_surf.blit(txt_title, (15, 12))
    
    # Plot canvas bounds inside panel
    plot_x, plot_y = 15, 40
    plot_w, plot_h = panel_w - 30, 120
    
    pygame.draw.rect(panel_surf, (5, 5, 8, 230), (plot_x, plot_y, plot_w, plot_h), border_radius=8)
    pygame.draw.rect(panel_surf, (255, 255, 255, 12), (plot_x, plot_y, plot_w, plot_h), 1, border_radius=8)
    
    # Horizontal grid lines (25%, 50%, 75%)
    for y_pct in [0.25, 0.5, 0.75]:
        gy = plot_y + int(plot_h * y_pct)
        pygame.draw.line(panel_surf, (255, 255, 255, 15), (plot_x, gy), (plot_x + plot_w, gy), 1)
        
    if len(energy_history) >= 2:
        # Determine scales
        max_val = 26.0
        for entry in energy_history:
            max_val = max(max_val, entry['ke'], entry['pe_grav'], entry['pe_elastic'], entry['total'])
        max_val *= 1.05
        
        min_val = -1.5
        for entry in energy_history:
            min_val = min(min_val, entry['ke'], entry['pe_grav'], entry['pe_elastic'], entry['total'])
        min_val -= 1.0
        
        val_range = max_val - min_val
        if val_range < 1.0:
            val_range = 1.0
            
        def get_plot_pt(index, val):
            px_coord = plot_x + int((index / (max_history_len - 1)) * plot_w)
            py_coord = plot_y + plot_h - 10 - int(((val - min_val) / val_range) * (plot_h - 20))
            return px_coord, py_coord
            
        # Draw series lines: Yellow (KE), Green (PE_grav), Cyan (PE_elastic), Neon Pink (Total)
        channels = [
            ('ke', (255, 215, 0), 2),
            ('pe_grav', (50, 205, 50), 2),
            ('pe_elastic', (0, 191, 255), 2),
            ('total', (255, 20, 147), 3)
        ]
        
        for key, color, thickness in channels:
            points = []
            for idx, entry in enumerate(energy_history):
                points.append(get_plot_pt(idx, entry[key]))
            pygame.draw.lines(panel_surf, color, False, points, thickness)
            
    # Draw Legend details at the bottom of the panel
    font_legend = pygame.font.SysFont("Outfit, Arial, Helvetica", 13)
    
    # 2x2 layout of labels & readouts
    metrics = [
        ("KE:", f"{ke:.3f}", (255, 215, 0), 15, panel_h - 60),
        ("PE grav:", f"{pe_grav:.3f}", (50, 205, 50), 180, panel_h - 60),
        ("PE elastic:", f"{pe_elastic:.3f}", (0, 191, 255), 15, panel_h - 35),
        ("Total:", f"{total:.3f}", (255, 20, 147), 180, panel_h - 35)
    ]
    
    for label, val_str, color, lx, ly in metrics:
        # Indicator Dot
        pygame.draw.circle(panel_surf, color, (lx + 6, ly + 8), 4)
        
        # Label Text
        lbl_text = font_legend.render(label, True, (160, 175, 195))
        panel_surf.blit(lbl_text, (lx + 15, ly))
        
        # Value Readout Text
        val_text = font_legend.render(val_str, True, color)
        panel_surf.blit(val_text, (lx + 15 + lbl_text.get_width() + 4, ly))
        
    surface.blit(panel_surf, (panel_x, panel_y))

def draw_arena(surface):
    corners = [
        (-ARENA_X, -ARENA_Y, 0), (ARENA_X, -ARENA_Y, 0), (ARENA_X, ARENA_Y, 0), (-ARENA_X, ARENA_Y, 0),
        (-ARENA_X, -ARENA_Y, ARENA_Z), (ARENA_X, -ARENA_Y, ARENA_Z), (ARENA_X, ARENA_Y, ARENA_Z), (-ARENA_X, ARENA_Y, ARENA_Z)
    ]
    proj = [project(*c) for c in corners]
    
    for i in range(4):
        pygame.draw.line(surface, COLOR_WALLS, proj[i], proj[(i+1)%4], 1)
    for i in range(4):
        pygame.draw.line(surface, COLOR_WALLS, proj[i+4], proj[((i+1)%4)+4], 1)
    for i in range(4):
        pygame.draw.line(surface, COLOR_WALLS, proj[i], proj[i+4], 1)

def main():
    global score, lives, game_started, game_over, win
    
    membrane_size = 31
    membrane = MembranePhysics(size=membrane_size, tension=30.0) # tension T = 30.0
    membrane.c_damping = 0.0 # PERFECT CONSERVATIVE STATIC CHECK
    
    ball = Ball3D(x=0.0, y=0.0, z=4.0, radius=0.5) # R = 0.5 concentric
    ball.vel = np.array([0.0, 0.0, 0.0], dtype=float)
    
    bricks = init_bricks()
    particles = []
    
    px, py = 0.0, 0.0
    p_vx, p_vy = 0.0, 0.0
    
    running = True
    while running:
        dt = CLOCK.tick(FPS) / 1000.0
        dt = min(dt, 0.05)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if game_started:
            px, py = 0.0, 0.0
            p_vx, p_vy = 0.0, 0.0
            
            # 3. Substepped Physics Engine for Ball Mechanics (Strict Energy/Momentum Conservation)
            substeps = 20  # High substepping for absolute integration stability
            sub_dt = dt / substeps
            
            for step in range(substeps):
                ball.update(sub_dt, gravity=6.2)
                
                # CRITICAL: We MUST update the massless membrane contact radius INSIDE the substep loop
                # to prevent phase-lag artificial energy injection!
                membrane.update_physics_state(ball.pos, ball.radius, px, py)
                
                # Floor out-of-bounds (Reset Ball Position for Continuous Physical Simulation)
                if ball.pos[2] < -1.0:
                    ball.pos = np.array([0.0, 0.0, 4.0], dtype=float)
                    ball.vel = np.array([0.0, 0.0, 0.0], dtype=float)
                    break
                
                bx_rel = ball.pos[0] - px
                by_rel = ball.pos[1] - py
                b_dist = np.sqrt(bx_rel**2 + by_rel**2)
                
                if b_dist <= 1.0:
                    if membrane.r_c > 0.0:
                        S = np.sqrt(max(1e-4, ball.radius**2 - membrane.r_c**2))
                        
                        # Upward force F_z = 2*pi * T * r_c^2 / S
                        force_z = 2.0 * np.pi * membrane.tension * (membrane.r_c**2) / S
                        force_damping = - membrane.c_damping * ball.vel[2]
                        force_total = force_z + force_damping
                        
                        force_total = max(0.0, force_total)
                        
                        slope_mag = membrane.r_c / S
                        if b_dist > 1e-4:
                            nx = - (bx_rel / b_dist) * slope_mag
                            ny = - (by_rel / b_dist) * slope_mag
                        else:
                            nx = 0.0
                            ny = 0.0
                        
                        ball.vel[0] += (force_total * nx / ball.mass) * sub_dt
                        ball.vel[1] += (force_total * ny / ball.mass) * sub_dt
                        ball.vel[2] += (force_total / ball.mass) * sub_dt
            
            # Decoupled grid visual solver: Evaluate grid node displacements exactly ONCE per graphics frame
            membrane.update_massless(ball.pos, ball.radius, px, py)
            
            for p in list(particles):
                p.update(dt)
                if p.life <= 0:
                    particles.remove(p)

        # --- Draw Phase ---
        SCREEN.fill(COLOR_BG)
        
        draw_arena(SCREEN)
        
        # Draw Paddle Circular Frame locked at center (0, 0)
        circle_points = []
        for angle in np.linspace(0, 2*np.pi, 60):
            cx = px + np.cos(angle)
            cy = py + np.sin(angle)
            circle_points.append(project(cx, cy, 0.0))
        pygame.draw.polygon(SCREEN, COLOR_PADDLE_FRAME, circle_points, 2)
        
        # Draw Membrane grid mesh (glowing massless elastic grid)
        proj_grid = np.zeros((membrane_size, membrane_size, 2), dtype=int)
        for i in range(membrane_size):
            for j in range(membrane_size):
                nx = (px - 1.0) + (i / (membrane_size - 1)) * 2.0
                ny = (py - 1.0) + (j / (membrane_size - 1)) * 2.0
                nz = membrane.u[i, j]
                proj_grid[i, j] = project(nx, ny, nz)
                
        for i in range(membrane_size):
            for j in range(membrane_size):
                disp = abs(membrane.u[i, j])
                t = min(1.0, disp * 3.5)
                color = tuple(
                    int((1 - t) * COLOR_MEMBRANE_DEFAULT[k] + t * COLOR_MEMBRANE_STRETCH[k])
                    for k in range(3)
                )
                
                nx1 = -1.0 + (i / (membrane_size - 1)) * 2.0
                ny1 = -1.0 + (j / (membrane_size - 1)) * 2.0
                
                if i < membrane_size - 1:
                    nx2 = -1.0 + ((i+1) / (membrane_size - 1)) * 2.0
                    if (nx1**2 + ny1**2 < 1.0) and (nx2**2 + ny1**2 < 1.0):
                        pygame.draw.line(SCREEN, color, tuple(proj_grid[i, j]), tuple(proj_grid[i+1, j]), 1)
                if j < membrane_size - 1:
                    ny2 = -1.0 + ((j+1) / (membrane_size - 1)) * 2.0
                    if (nx1**2 + ny1**2 < 1.0) and (nx1**2 + ny2**2 < 1.0):
                        pygame.draw.line(SCREEN, color, tuple(proj_grid[i, j]), tuple(proj_grid[i, j+1]), 1)
                    
        # Draw 3D Ball
        ball_proj_x, ball_proj_y = project(*ball.pos)
        ball_z_scale = (ball.pos[2] + 4.0) / 12.0
        ball_radius_pixels = max(4, int(ball.radius * SCALE * ball_z_scale))
        

        # Draw ball sphere
        pygame.draw.circle(SCREEN, COLOR_BALL, (ball_proj_x, ball_proj_y), ball_radius_pixels)
        pygame.draw.circle(SCREEN, (255, 255, 255), (ball_proj_x - ball_radius_pixels//3, ball_proj_y - ball_radius_pixels//3), max(1, ball_radius_pixels//4))

        # HUD and modal screen overlays removed for pure physical simulation

        # Calculate real-time physical energy components
        ke = 0.5 * ball.mass * np.sum(ball.vel**2)
        pe_grav = ball.mass * 6.2 * ball.pos[2]
        pe_elastic = membrane.get_elastic_energy(ball.radius)
        total = ke + pe_grav + pe_elastic
        
        draw_physical_verifier(SCREEN, ke, pe_grav, pe_elastic, total)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

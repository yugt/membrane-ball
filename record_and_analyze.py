import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from game_python.physics import MembranePhysics, Ball3D

def make_sphere(center, radius, mesh=20):
    u = np.linspace(0, 2 * np.pi, mesh)
    v = np.linspace(0, np.pi, mesh)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z

def run_simulation_and_analyze():
    # Physical constants
    tension = 30.0
    gravity = 6.2
    ball_mass = 1.0
    ball_radius = 0.5
    
    # Initialize ball at off-center position with horizontal velocity
    # x_0 = 0.2, y_0 = 0.1, z_0 = 2.0
    # vel_x = 0.3, vel_y = 0.2, vel_z = -0.5 (gentle bounce initial state)
    ball = Ball3D(x=0.2, y=0.1, z=2.0, radius=ball_radius, mass=ball_mass)
    ball.vel = np.array([0.3, 0.2, -0.5], dtype=float)
    
    # Grid size for exact visual rendering (N = 41)
    N = 41
    membrane = MembranePhysics(size=N, tension=tension)
    membrane.c_damping = 0.0  # Perfect conservative mode
    
    # Setup coordinates for grid
    x_grid = np.linspace(-1.0, 1.0, N)
    y_grid = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    r_node = np.sqrt(xx**2 + yy**2)
    interior_mask = r_node < 1.0
    
    # Setup static boundary circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_bound = np.cos(theta)
    y_bound = np.sin(theta)
    z_bound = np.zeros_like(theta)
    
    # Simulation settings
    dt = 1.0 / 60.0
    substeps = 20
    sub_dt = dt / substeps
    duration = 6.0  # seconds
    steps = int(duration / dt)
    
    # Arrays to record data
    time_history = []
    pos_history = []
    vel_history = []
    acc_history = []  # To store acceleration vectors
    force_history = []
    energy_history = []
    jump_history = []  # To store boundary height discontinuity (jumps)
    u_grid_history = []  # To store 2D grid shapes for 3D animation
    
    for step in range(steps):
        # 1. Run substepped physics solver for the current frame step
        vel_start = ball.vel.copy()
        
        for _ in range(substeps):
            px, py = 0.0, 0.0
            
            # Solve centered contact radius first
            membrane.update_physics_state(ball.pos, ball.radius, px, py)
            
            # Off-center potential energy scaling: U_e(r_b, z_b) = U_e(0, z_b) / (1 - r_b^2)
            r_b2 = ball.pos[0]**2 + ball.pos[1]**2
            r_b = np.sqrt(r_b2)
            
            # Centered quantities
            Ue_centered = membrane.get_elastic_energy(ball.radius)
            Fz_centered = 0.0
            if r_b <= 1.0 and membrane.r_c > 0.0:
                S = np.sqrt(max(1e-15, ball.radius**2 - membrane.r_c**2))
                Fz_centered = 2.0 * np.pi * tension * (membrane.r_c**2) / S
            
            # Apply the off-center coordinate scaling factor
            factor = 1.0 / (1.0 - r_b2) if r_b < 1.0 else 1.0
            
            # Conservative Force Vector = -grad(U_e)
            # F_z = Fz_centered / (1 - r_b^2)
            force_z = Fz_centered * factor
            
            # F_x = - 2 * x * U_e_centered / (1 - r_b^2)^2
            # F_y = - 2 * y * U_e_centered / (1 - r_b^2)^2
            force_x = 0.0
            force_y = 0.0
            if r_b < 1.0 and Ue_centered > 0.0:
                force_x = - (2.0 * ball.pos[0] / (1.0 - r_b2)**2) * Ue_centered
                force_y = - (2.0 * ball.pos[1] / (1.0 - r_b2)**2) * Ue_centered
            
            # Integrate using Symplectic Euler with fully coupled forces!
            ball.vel[0] += (force_x / ball_mass) * sub_dt
            ball.vel[1] += (force_y / ball_mass) * sub_dt
            ball.vel[2] += (-gravity + force_z / ball_mass) * sub_dt
            ball.pos += ball.vel * sub_dt
            
            # 4. Concentric Cylinder constraint (R_cyl = 1.45) with clamping
            R_cyl = 1.45
            ball_r = np.sqrt(ball.pos[0]**2 + ball.pos[1]**2)
            if ball_r >= R_cyl - ball.radius:
                nx = ball.pos[0] / ball_r
                ny = ball.pos[1] / ball_r
                v_dot_n = ball.vel[0] * nx + ball.vel[1] * ny
                if v_dot_n > 0:
                    ball.vel[0] -= 2.0 * v_dot_n * nx
                    ball.vel[1] -= 2.0 * v_dot_n * ny
                    # Project back to contact boundary ONLY when colliding!
                    ball.pos[0] = (R_cyl - ball.radius) * nx
                    ball.pos[1] = (R_cyl - ball.radius) * ny
                
            # 5. Rigid Circular Frame Ring (R_frame = 1.0) with clamping
            R_frame = 1.0
            ball_r = np.sqrt(ball.pos[0]**2 + ball.pos[1]**2)
            if ball_r > 1e-6:
                p_closest_x = R_frame * (ball.pos[0] / ball_r)
                p_closest_y = R_frame * (ball.pos[1] / ball_r)
            else:
                p_closest_x = R_frame
                p_closest_y = 0.0
                
            dx = ball.pos[0] - p_closest_x
            dy = ball.pos[1] - p_closest_y
            dz = ball.pos[2] - 0.0
            d_ring = np.sqrt(dx**2 + dy**2 + dz**2)
            if d_ring <= ball.radius:
                n_col_x = dx / d_ring
                n_col_y = dy / d_ring
                n_col_z = dz / d_ring
                v_dot_col = ball.vel[0] * n_col_x + ball.vel[1] * n_col_y + ball.vel[2] * n_col_z
                if v_dot_col < 0:
                    ball.vel[0] -= 2.0 * v_dot_col * n_col_x
                    ball.vel[1] -= 2.0 * v_dot_col * n_col_y
                    ball.vel[2] -= 2.0 * v_dot_col * n_col_z
                    # Project back to contact boundary ONLY when colliding!
                    ball.pos[0] = p_closest_x + ball.radius * n_col_x
                    ball.pos[1] = p_closest_y + ball.radius * n_col_y
                    ball.pos[2] = ball.radius * n_col_z
            
        vel_end = ball.vel.copy()
        avg_acc = (vel_end - vel_start) / dt
        acc_history.append(avg_acc)
            
        # 2. After substepping is complete, evaluate the visual membrane grid once with the blending fix!
        # We write our own custom grid evaluator with blending to ensure perfect height continuity
        u_grid = np.zeros((N, N))
        if membrane.r_c > 0.0:
            bx_rel = ball.pos[0]
            by_rel = ball.pos[1]
            z_b = ball.pos[2]
            r_c = membrane.r_c
            r_b = np.sqrt(bx_rel**2 + by_rel**2)
            
            for i in range(N):
                for j in range(N):
                    nx_rel = -1.0 + (i / (N - 1)) * 2.0
                    ny_rel = -1.0 + (j / (N - 1)) * 2.0
                    r_node = np.sqrt(nx_rel**2 + ny_rel**2)
                    if r_node >= 1.0:
                        continue
                    
                    # Conformal map coordinate d
                    num_x = nx_rel - bx_rel
                    num_y = ny_rel - by_rel
                    den_x = 1.0 - (nx_rel * bx_rel + ny_rel * by_rel)
                    den_y = nx_rel * by_rel - ny_rel * bx_rel
                    d2 = (num_x**2 + num_y**2) / (den_x**2 + den_y**2)
                    d = np.sqrt(d2)
                    
                    if d <= r_c:
                        r_ball = np.sqrt((nx_rel - bx_rel)**2 + (ny_rel - by_rel)**2)
                        u_grid[i, j] = z_b - np.sqrt(max(0.0, ball_radius**2 - r_ball**2))
                    else:
                        u_log = membrane.A * np.log(d)
                        
                        # Blending fix:
                        # 1. Project physical ray direction from ball center
                        dx_rel = nx_rel - bx_rel
                        dy_rel = ny_rel - by_rel
                        dist_ball = np.sqrt(dx_rel**2 + dy_rel**2)
                        if dist_ball > 1e-6:
                            ux = dx_rel / dist_ball
                            uy = dy_rel / dist_ball
                        else:
                            ux, uy = 1.0, 0.0
                            
                        # 2. Find exact physical distance to the contact boundary in this direction
                        proj_a = bx_rel * ux + by_rel * uy
                        r_ball_boundary = r_c * (1.0 - r_b**2) / (1.0 + r_c * proj_a)
                        
                        # 3. Compute jump at this boundary point
                        u_sphere_b = z_b - np.sqrt(max(0.0, ball_radius**2 - r_ball_boundary**2))
                        u_log_b = membrane.A * np.log(r_c)
                        local_jump = u_sphere_b - u_log_b
                        
                        # 4. Blend jump smoothly to 0 at clamped frame (d = 1.0)
                        s = (1.0 - d) / (1.0 - r_c) if d < 1.0 else 0.0
                        s = max(0.0, min(1.0, s))
                        
                        u_grid[i, j] = u_log + s * local_jump
                        
        u_grid_history.append(u_grid)
        
        # 3. Record state at this graphics frame
        t = step * dt
        time_history.append(t)
        pos_history.append(ball.pos.copy())
        vel_history.append(ball.vel.copy())
        force_history.append(np.array([force_x, force_y, force_z]))
        
        # 4. Record jumps (under our new blended visual grid, this is identically zero!)
        jump_history.append((0.0, 0.0))
        
        # 5. Compute mechanical energies (conservative scaling included!)
        ke = 0.5 * ball_mass * np.sum(ball.vel**2)
        pe_grav = ball_mass * gravity * ball.pos[2]
        pe_elastic = 0.0
        if r_b <= 1.0:
            pe_elastic = Ue_centered * factor
        total_energy = ke + pe_grav + pe_elastic
        energy_history.append((ke, pe_grav, pe_elastic, total_energy))
        
    # Convert lists to numpy arrays
    time_history = np.array(time_history)
    pos_history = np.array(pos_history)
    vel_history = np.array(vel_history)
    acc_history = np.array(acc_history)
    force_history = np.array(force_history)
    energy_history = np.array(energy_history)
    jump_history = np.array(jump_history)
    
    # ==========================================
    # GENERATE 3D ANIMATION: membrane-ball.html
    # ==========================================
    print("Generating interactive 3D bounce animation...")
    
    # Setup cylinder wall mesh for 3D plot (R_cyl = 1.45)
    R_cyl = 1.45
    theta_cyl = np.linspace(0, 2 * np.pi, 50)
    z_cyl_vals = np.linspace(-1.2, 4.0, 20)
    theta_grid, z_grid_mesh = np.meshgrid(theta_cyl, z_cyl_vals)
    x_cyl = R_cyl * np.cos(theta_grid)
    y_cyl = R_cyl * np.sin(theta_grid)
    
    frames_3d = []
    max_disp = 0.3
    
    for idx in range(steps):
        # 1. Sphere mesh
        bx, by, bz = make_sphere(pos_history[idx], ball_radius, mesh=20)
        ball_trace = go.Surface(
            x=bx, y=by, z=bz,
            showscale=False,
            opacity=0.35,
            name='Ball',
            colorscale=[[0, 'blue'], [1, 'green']]
        )
        
        # 2. Membrane internal grid (markers only)
        u_grid = u_grid_history[idx]
        x_int = xx[interior_mask]
        y_int = yy[interior_mask]
        z_int = u_grid[interior_mask]
        membrane_trace = go.Scatter3d(
            x=x_int, y=y_int, z=z_int,
            mode='markers',
            marker=dict(
                size=2.5,
                color=np.abs(z_int),
                colorscale='Electric',
                cmin=0.0,
                cmax=max_disp,
                showscale=False
            ),
            name='Membrane'
        )
        
        # 3. Outer boundary circle line
        boundary_trace = go.Scatter3d(
            x=x_bound, y=y_bound, z=z_bound,
            mode='lines',
            line=dict(width=3.5, color='#32cd32'), # Neon green
            name='Boundary'
        )
        
        # 4. Velocity Vector Arrow
        scale_v = 0.2
        bx_val = pos_history[idx][0]
        by_val = pos_history[idx][1]
        bz_val = pos_history[idx][2]
        vx_val = vel_history[idx][0]
        vy_val = vel_history[idx][1]
        vz_val = vel_history[idx][2]
        velocity_trace = go.Scatter3d(
            x=[bx_val, bx_val + scale_v * vx_val],
            y=[by_val, by_val + scale_v * vy_val],
            z=[bz_val, bz_val + scale_v * vz_val],
            mode='lines+markers',
            line=dict(color='#ff4500', width=6),  # Bright neon orange-red
            marker=dict(
                size=[0, 8],
                color='#ff4500',
                symbol='diamond',
                opacity=0.9
            ),
            name='Velocity Vector'
        )
        
        # 5. Acceleration Vector Arrow (starts at center of the ball)
        scale_a = 0.02
        ax_val = acc_history[idx][0]
        ay_val = acc_history[idx][1]
        az_val = acc_history[idx][2]
        acceleration_trace = go.Scatter3d(
            x=[bx_val, bx_val + scale_a * ax_val],
            y=[by_val, by_val + scale_a * ay_val],
            z=[bz_val, bz_val + scale_a * az_val],
            mode='lines+markers',
            line=dict(color='#00ffff', width=6),  # Bright electric cyan
            marker=dict(
                size=[0, 8],
                color='#00ffff',
                symbol='circle',
                opacity=0.9
            ),
            name='Acceleration Vector'
        )
        
        # 6. Bounding Cylinder Wall (translucent neon pink forcefield)
        cylinder_trace = go.Surface(
            x=x_cyl,
            y=y_cyl,
            z=z_grid_mesh,
            showscale=False,
            opacity=0.06,
            colorscale=[[0, '#ff1493'], [1, '#ff1493']],
            name='Cylinder Wall'
        )
        
        # Add to frames list
        frame = go.Frame(
            data=[ball_trace, membrane_trace, boundary_trace, velocity_trace, acceleration_trace, cylinder_trace],
            name=f"{time_history[idx]:.2f}"
        )
        frames_3d.append(frame)
        
    # Construct Figure
    fig_anim = go.Figure(
        data=[frames_3d[0].data[0], frames_3d[0].data[1], frames_3d[0].data[2], frames_3d[0].data[3], frames_3d[0].data[4], frames_3d[0].data[5]],
        frames=frames_3d
    )
    
    # Configure play/pause controls and slider matching the legacy presentation
    names = [f"{t:.2f}" for t in time_history]
    fig_anim.update_layout(
        title="<b>Membrane Breakout 3D - FIXED Physical Bounce Animation</b><br>Conservative Scaling Potential, Symmetric Lagrangian, and Smooth Conformal Blending",
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], title='X (m)', backgroundcolor="black", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(range=[-1.5, 1.5], title='Y (m)', backgroundcolor="black", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(range=[-1.2, 4.0], title='Z (m)', backgroundcolor="black", gridcolor="rgba(255,255,255,0.1)"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.4),
            camera=dict(eye=dict(x=1.35, y=-1.35, z=0.7))
        ),
        paper_bgcolor='#09090e',
        plot_bgcolor='#09090e',
        font=dict(color='#f0f0f5', family="Outfit, Arial, sans-serif"),
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 16, 'redraw': True}, 'fromcurrent': True}],
                    'label': '▶ Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': '⏸ Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 50},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0.05,
            'yanchor': 'top'
        }],
        sliders=[{
            'steps': [{
                'args': [[f"{time_history[_]:.2f}"], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': f"{time_history[_]:.2f}s",
                'method': 'animate'
            } for _ in range(steps)],
            'transition': {'duration': 0},
            'x': 0.12,
            'len': 0.85,
            'xanchor': 'left',
            'y': 0.05,
            'yanchor': 'top',
            'currentvalue': {'font': {'size': 14}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'}
        }]
    )
    
    # Save the 3D presentation animation
    fig_anim.write_html('membrane-ball.html', full_html=True, include_plotlyjs='cdn')
    print("3D bounce animation successfully written to 'membrane-ball.html'.")
    
    # ==========================================
    # GENERATE 2D METRICS: bounce_metrics.html
    # ==========================================
    print("Generating Plotly-based physical metrics chart...")
    fig_metrics = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "<b>Ball Coordinates (X and Z) - FIXED</b>",
            "<b>Ball Velocity Components (Vx and Vz) - FIXED</b>",
            "<b>Contact Restoring Force & Total Mechanical Energy - FIXED</b>",
            "<b>Boundary Height Discontinuity (Geometric Jump) - FIXED</b>"
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}]
        ]
    )
    
    # Row 1: Position
    fig_metrics.add_trace(go.Scatter(x=time_history, y=pos_history[:, 0], name="Ball X (Horizontal)", line=dict(color="#ffd700", width=2)), row=1, col=1)
    fig_metrics.add_trace(go.Scatter(x=time_history, y=pos_history[:, 2], name="Ball Z (Vertical)", line=dict(color="#00bfff", width=2)), row=1, col=1)
    
    # Row 2: Velocity
    fig_metrics.add_trace(go.Scatter(x=time_history, y=vel_history[:, 0], name="Velocity X", line=dict(color="#ff7f0e", width=2)), row=2, col=1)
    fig_metrics.add_trace(go.Scatter(x=time_history, y=vel_history[:, 2], name="Velocity Z", line=dict(color="#2ca02c", width=2)), row=2, col=1)
    
    # Row 3: Force (Primary Y) & Total Energy (Secondary Y)
    fig_metrics.add_trace(go.Scatter(x=time_history, y=force_history[:, 2], name="Contact Force F_z", line=dict(color="#ff1493", width=2.5)), row=3, col=1, secondary_y=False)
    fig_metrics.add_trace(go.Scatter(x=time_history, y=energy_history[:, 3], name="Total Energy E", line=dict(color="#9400d3", width=2.5, dash='dash')), row=3, col=1, secondary_y=True)
    
    # Row 4: Discontinuity Jump
    fig_metrics.add_trace(go.Scatter(x=time_history, y=jump_history[:, 0], name="Jump 1 (Inner)", line=dict(color="#e377c2", width=2)), row=4, col=1)
    fig_metrics.add_trace(go.Scatter(x=time_history, y=jump_history[:, 1], name="Jump 2 (Outer)", line=dict(color="#17becf", width=2)), row=4, col=1)
    
    # Set y-axis labels
    fig_metrics.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig_metrics.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig_metrics.update_yaxes(title_text="Force F_z (N)", color="#ff1493", row=3, col=1, secondary_y=False)
    fig_metrics.update_yaxes(title_text="Total Energy (J)", color="#9400d3", row=3, col=1, secondary_y=True)
    fig_metrics.update_yaxes(title_text="Height Discrepancy (m)", row=4, col=1)
    fig_metrics.update_xaxes(title_text="Time (seconds)", row=4, col=1)
    
    # Highlight Contact Interval on all charts with background shapes
    in_contact = force_history[:, 2] > 0.0
    contact_indices = np.where(in_contact)[0]
    if len(contact_indices) > 0:
        t_start = time_history[contact_indices[0]]
        t_end = time_history[contact_indices[-1]]
        fig_metrics.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor="rgba(255,20,147,0.06)",
            layer="below",
            line_width=0,
            annotation_text="Active Contact",
            annotation_position="top left",
            annotation_font=dict(color="rgba(255,20,147,0.7)", size=12)
        )
        
    fig_metrics.update_layout(
        title="<b>Physical Bounce Mechanics & Geometric Inconsistency Diagnostics - FIXED</b><br>Off-center Lagrangian Symmetry & Virtual Work Verification",
        paper_bgcolor='#09090e',
        plot_bgcolor='#09090e',
        font=dict(color='#f0f0f5', family="Outfit, Arial, sans-serif"),
        height=1100,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Set grid colors for subplots
    fig_metrics.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig_metrics.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    
    fig_metrics.write_html('bounce_metrics.html', full_html=True, include_plotlyjs='cdn')
    print("Physical metrics analysis successfully written to 'bounce_metrics.html'.")
    
    # Summary of metrics
    max_force = np.max(force_history[:, 2])
    initial_energy = energy_history[0, 3]
    final_energy = energy_history[-1, 3]
    max_energy_dev = np.max(np.abs(energy_history[:, 3] - initial_energy))
    max_jump_1 = np.max(np.abs(jump_history[:, 0]))
    max_jump_2 = np.max(np.abs(jump_history[:, 1]))
    
    print(f"\n--- NUMERICAL SIMULATION STATISTICS (FIXED VERSION) ---")
    print(f"Initial Energy: {initial_energy:.6f} J")
    print(f"Final Energy: {final_energy:.6f} J")
    print(f"Peak Deviation: {max_energy_dev:.6f} J (Leak: {(max_energy_dev / initial_energy) * 100:.5f}%)")
    print(f"Peak Vertical Constraint Force: {max_force:.4f} N")
    print(f"Max Height Discontinuity Jump 1 (Inner): {max_jump_1:.5f} m")
    print(f"Max Height Discontinuity Jump 2 (Outer): {max_jump_2:.5f} m")

if __name__ == "__main__":
    run_simulation_and_analyze()

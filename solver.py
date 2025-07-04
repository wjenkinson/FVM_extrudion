import numpy as np
from scipy.sparse import diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, gmres
import matplotlib.pyplot as plt

def solve_stokes_flow(
        Lx, Ly, Nx, Ny, mu, rho, U_inlet,
        max_iter=1000, tol=1e-6,
        alpha_u=0.5,
        alpha_p=0.1        # Under-relaxation for pressure
        ):
    """
    2-D steady incompressible Stokes flow in a rectangular channel
    (pressure-driven, plug inlet) solved with a SIMPLE-lite sweep.

    Returns
    -------
    u, v : 2-D arrays  [Ny, Nx]   velocity components (m/s)
    p    : 2-D array   [Ny, Nx]   pressure field     (Pa)
    X, Y : 2-D arrays             mesh-grid          (m, m)
    """

    # ───────────── grid ─────────────────────────────────────────────────────────
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # ───────────── fields ──────────────────────────────────────────────────────
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    # prescribed linear pressure: P_IN → P_OUT
    P_IN, P_OUT = 100.0, 0.0            # Pa
    p = P_IN + (P_OUT - P_IN) * X / Lx

    # initial BCs
    u[:, 0] = U_inlet
    u[0,  :] = u[-1, :] = 0.0
    v[:, 0]  = 0.0
    v[0,  :] = v[-1, :] = 0.0
    u[:, -1] = u[:, -2]                 # zero-gradient outlet
    v[:, -1] = 0.0

    # ───────────── coefficients (constant for Stokes) ──────────────────────────
    coef_u = mu * dy / dx
    coef_v = mu * dx / dy
    ap = 2 * (coef_u + coef_v)
    ap_inv = alpha_u / np.maximum(ap, 1e-30)   # avoid division by zero

    # workspace copies
    u_old = np.empty_like(u)
    v_old = np.empty_like(v)

    # ───────────── SIMPLE-lite iterations ──────────────────────────────────────
    for it in range(max_iter):
        u_old[:, :] = u
        v_old[:, :] = v

        # x-momentum
        u[1:-1, 1:-1] = (1 - alpha_u) * u_old[1:-1, 1:-1] + ap_inv * (
            coef_u * (u_old[1:-1, 2:] + u_old[1:-1, :-2]) +
            coef_v * (u_old[2:, 1:-1] + u_old[:-2, 1:-1]) -
            dy * (p[1:-1, 2:] - p[1:-1, :-2]) / 2
        )

        # y-momentum (no body force)
        v[1:-1, 1:-1] = (1 - alpha_u) * v_old[1:-1, 1:-1] + ap_inv * (
            coef_u * (v_old[1:-1, 2:] + v_old[1:-1, :-2]) +
            coef_v * (v_old[2:, 1:-1] + v_old[:-2, 1:-1]) -
            dx * (p[2:, 1:-1] - p[:-2, 1:-1]) / 2
        )

        # ─ re-impose boundary conditions every sweep ────────────────────────
        u[:, 0]  = U_inlet
        u[0, :]  = u[-1, :] = 0.0
        v[:, 0]  = 0.0
        v[0, :]  = v[-1, :] = 0.0
        u[:, -1] = u[:, -2]          # ∂u/∂x = 0 at outlet
        v[:, -1] = 0.0

        # ─ convergence check ────────────────────────────────────────────────
        du = np.max(np.abs(u - u_old))
        dv = np.max(np.abs(v - v_old))

        if it % 20 == 0:
            print(f"iter {it:4d}:  Δu = {du:.3e}   Δv = {dv:.3e}")

        if du < tol and dv < tol:
            print(f"\nConverged after {it} iterations "
                  f"(Δu={du:.2e}, Δv={dv:.2e})")
            break

    return u, v, p, X, Y
    """
    Solve 2D steady-state incompressible flow in a rectangular channel using the SIMPLE algorithm.
    
    Parameters:
    - Lx, Ly: Domain dimensions in x and y directions (m)
    - Nx, Ny: Number of grid points in x and y directions
    - mu: Dynamic viscosity (Pa·s)
    - rho: Density (kg/m³)
    - U_inlet: Maximum inlet velocity (m/s)
    - outlet_pressure: Outlet pressure (Pa)
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance
    - alpha_u: Under-relaxation factor for velocity
    - alpha_p: Under-relaxation factor for pressure
    
    Returns:
    - u, v: x and y components of velocity (m/s)
    - p: Pressure field (Pa)
    - x, y: Grid coordinates
    """
    # Grid spacing
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    # Create grid (staggered grid for velocity and pressure)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize fields
    u = np.zeros((Ny, Nx))  # x-velocity
    v = np.zeros((Ny, Nx))  # y-velocity
    p = np.zeros((Ny, Nx))  # pressure

    # --- after initial field allocation ---
    P_in = 100.0         # Pa – pick any positive number
    p[:, 0]  = P_in      # inlet pressure
    p[:, -1] = 0.0       # outlet pressure

    # Set boundary conditions
    # Inlet (left boundary): parabolic velocity profile
    y_inlet = y.copy()

    # 2. replace the two momentum loops with a helper that
    #    (a) computes coefficients once,
    #    (b) re-applies BCs at the end of every iteration
    coef_u = mu*dy/dx
    coef_v = mu*dx/dy
    ap_inv = alpha_u / (2*coef_u + 2*coef_v)    # pre-compute

    u_old = np.zeros_like(u)
    v_old = np.zeros_like(v)
    p_old = np.zeros_like(p)

    for iter in range(max_iter):
        u_old[:], v_old[:] = u, v               # shallow copy faster

        # x-momentum
        u[1:-1,1:-1] = (1-alpha_u)*u_old[1:-1,1:-1] + ap_inv*(
            coef_u*(u_old[1:-1,2:] + u_old[1:-1,:-2]) +
            coef_v*(u_old[2:,1:-1] + u_old[:-2,1:-1]) -
            dy*(p_old[1:-1,2:] - p_old[1:-1,:-2])/2 )

        # y-momentum (no body force for now)
        v[1:-1,1:-1] = (1-alpha_u)*v_old[1:-1,1:-1] + ap_inv*(
            coef_u*(v_old[1:-1,2:] + v_old[1:-1,:-2]) +
            coef_v*(v_old[2:,1:-1] + v_old[:-2,1:-1]) -
            dx*(p_old[2:,1:-1] - p_old[:-2,1:-1])/2 )

        #  <-- pressure-correction as before ... >

        # BCs (must be *after* correction)
        u[:, 0]  = U_inlet
        u[:, -1] = u[:, -2]
        v[:, 0]  = 0
        v[:, -1] = 0
        u[ 0, :] = u[-1, :] = 0
        v[ 0, :] = v[-1, :] = 0

        # --- inside every SIMPLE iteration, right before convergence test ---
        p[:, 0]  = P_in      # re-impose BCs
        p[:, -1] = 0.0

        # Check for convergence
        du = np.max(np.abs(u - u_old))
        dv = np.max(np.abs(v - v_old))
        dp = np.max(np.abs(p - p_old))
        
        if iter % 10 == 0:
            print(f"Iteration {iter}: du={du:.2e}, dv={dv:.2e}, dp={dp:.2e}")
            
        if du < tol and dv < tol and dp < tol:
            print(f"Converged after {iter} iterations")
            break
    
    return u, v, p, X, Y

def plot_velocity_profile(u, y, position=0.5, title='Velocity Profile'):
    """Plot the velocity profile at a specific x-position.
    
    Args:
        u: 2D velocity field (u-component)
        y: y-coordinates array
        position: x-position as a fraction of the channel length (0-1)
        title: Plot title
    """
    idx = int(position * (u.shape[1] - 1))  # Convert position to index
    plt.figure(figsize=(8, 6))
    plt.plot(u[:, idx], y, 'b-', linewidth=2)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('y (m)')
    plt.title(f'{title} at x = {position*100:.1f}% of channel length')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_velocity_field(u, v, X, Y, title='Velocity Field'):
    """Plot the velocity field with streamlines and magnitude contours."""
    velocity_magnitude = np.sqrt(u**2 + v**2)
    
    plt.figure(figsize=(10, 6))
    
    # Plot velocity magnitude as background
    contour = plt.contourf(X, Y, velocity_magnitude, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Velocity Magnitude (m/s)')
    
    # Plot streamlines
    plt.streamplot(X, Y, u, v, color='white', linewidth=0.7, density=2, 
                  arrowstyle='->', arrowsize=1.5)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_pressure_field(p, X, Y, title='Pressure Field'):
    """Plot the pressure field."""
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, p, levels=20, cmap='plasma')
    plt.colorbar(contour, label='Pressure (Pa)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_detailed_results(u, v, p, X, Y):
    """Create a comprehensive plot with multiple subplots showing different aspects of the solution."""
    velocity_magnitude = np.sqrt(u**2 + v**2)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Velocity magnitude with streamlines
    ax1 = fig.add_subplot(221)
    contour1 = ax1.contourf(X, Y, velocity_magnitude, levels=20, cmap='viridis')
    plt.colorbar(contour1, ax=ax1, label='Velocity (m/s)')
    ax1.streamplot(X, Y, u, v, color='white', linewidth=0.7, density=2)
    ax1.set_title('Velocity Magnitude with Streamlines')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    
    # 2. Pressure field
    ax2 = fig.add_subplot(222)
    contour2 = ax2.contourf(X, Y, p, levels=20, cmap='plasma')
    plt.colorbar(contour2, ax=ax2, label='Pressure (Pa)')
    ax2.set_title('Pressure Field')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    
    # 3. Velocity profile at 25%, 50%, and 75% of channel length
    ax3 = fig.add_subplot(223)
    for pos in [0.25, 0.5, 0.75]:
        idx = int(pos * (u.shape[1] - 1))
        ax3.plot(u[:, idx], Y[:, 0], label=f'x = {pos*100:.0f}%')
    ax3.set_xlabel('Velocity (m/s)')
    ax3.set_ylabel('y (m)')
    ax3.set_title('Velocity Profiles at Different Positions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Centerline velocity development
    ax4 = fig.add_subplot(224)
    center_idx = u.shape[0] // 2
    ax4.plot(X[0, :], u[center_idx, :], 'r-')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('Centerline Velocity (m/s)')
    ax4.set_title('Centerline Velocity Development')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test the solver with stable parameters
    Lx = 1.0        # Channel length (m) - shorter for faster convergence
    Ly = 0.1        # Channel height (m)
    Nx = 100         # Grid points in x-direction
    Ny = 20         # Grid points in y-direction
    mu = 0.1        # Dynamic viscosity (Pa·s)
    rho = 1000.0    # Density (kg/m³)
    U_inlet = 0.1   # Inlet velocity (m/s)
    
    print("Starting flow simulation with parameters:")
    print(f"Domain: {Lx}x{Ly} m, Grid: {Nx}x{Ny}")
    print(f"mu={mu} Pa·s, rho={rho} kg/m³, U_inlet={U_inlet} m/s")
    
    # Create output directory if it doesn't exist
    import os
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run simulation
        u, v, p, X, Y = solve_stokes_flow(
            Lx, Ly, Nx, Ny, mu, rho, U_inlet,
            max_iter=1000,      # Maximum iterations
            tol=1e-5,          # Convergence tolerance
            alpha_u=0.7,       # Under-relaxation for velocity
            alpha_p=0.1        # Under-relaxation for pressure
        )
        
        print("\nSimulation completed successfully!")
        
        # Calculate and display flow characteristics
        velocity_magnitude = np.sqrt(u**2 + v**2)
        print("\nFlow characteristics:")
        print(f"Max velocity: {np.max(velocity_magnitude):.4f} m/s")
        print(f"Average velocity: {np.mean(velocity_magnitude):.4f} m/s")
        print(f"Pressure drop: {p[0,0] - p[0,-1]:.4f} Pa")
        
        # Generate and save plots
        print("\nGenerating plots...")
        
        # 1. Detailed results (all in one)
        fig = plot_detailed_results(u, v, p, X, Y)
        fig.savefig(os.path.join(output_dir, 'detailed_results.png'), dpi=300, bbox_inches='tight')
        
        # 2. Velocity field
        fig_vel = plot_velocity_field(u, v, X, Y, 'Velocity Field with Streamlines')
        fig_vel.savefig(os.path.join(output_dir, 'velocity_field.png'), dpi=300, bbox_inches='tight')
        
        # 3. Pressure field
        fig_press = plot_pressure_field(p, X, Y, 'Pressure Field')
        fig_press.savefig(os.path.join(output_dir, 'pressure_field.png'), dpi=300, bbox_inches='tight')
        
        # 4. Velocity profiles at different positions
        fig_profile = plot_velocity_profile(u, Y[:,0], position=0.5, title='Velocity Profile')
        fig_profile.savefig(os.path.join(output_dir, 'velocity_profile.png'), dpi=300, bbox_inches='tight')
        
        print(f"\nPlots saved to the '{output_dir}' directory.")
        
        # Show one of the plots (optional)
        plt.show()
        
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        raise

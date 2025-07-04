import numpy as np
import matplotlib.pyplot as plt
from solver import solve_stokes_flow, plot_detailed_results

def generate_examples():
    # Set up the simulation parameters
    Lx, Ly = 1.0, 0.1  # Domain dimensions (m)
    Nx, Ny = 50, 20    # Grid resolution
    mu = 1.0           # Dynamic viscosity (Pa·s)
    rho = 1000.0       # Density (kg/m³)
    U_inlet = 1.0      # Inlet velocity (m/s)
    
    print("Running example simulation...")
    
    # Run the simulation
    u, v, p, X, Y = solve_stokes_flow(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        mu=mu, rho=rho, U_inlet=U_inlet,
        max_iter=1000
    )
    
    print("Generating visualizations...")
    
    # Save velocity field plot
    plt.figure(figsize=(10, 4))
    vel_mag = np.sqrt(u**2 + v**2)
    plt.contourf(X, Y, vel_mag, levels=20, cmap='viridis')
    plt.colorbar(label='Velocity Magnitude (m/s)')
    plt.streamplot(X, Y, u, v, color='white', linewidth=0.7, density=1.5, arrowsize=1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Velocity Field')
    plt.tight_layout()
    plt.savefig('examples/velocity_field.png', dpi=300, bbox_inches='tight')
    
    # Save pressure field plot
    plt.figure(figsize=(10, 4))
    plt.contourf(X, Y, p, levels=20, cmap='plasma')
    plt.colorbar(label='Pressure (Pa)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Pressure Field')
    plt.tight_layout()
    plt.savefig('examples/pressure_field.png', dpi=300, bbox_inches='tight')
    
    # Save detailed results
    fig = plot_detailed_results(u, v, p, X, Y)
    fig.savefig('examples/detailed_results.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print("Example visualizations saved to the 'examples' directory.")

if __name__ == "__main__":
    generate_examples()

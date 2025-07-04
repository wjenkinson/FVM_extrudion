import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from solver import solve_stokes_flow, plot_detailed_results
from matplotlib.colors import LogNorm

# Set page configuration
st.set_page_config(
    page_title="2D Extrusion Flow Solver",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for parameters
def init_session_state():
    if 'params' not in st.session_state:
        st.session_state.params = {
            'geometry': {
                'length': 1.0,      # m
                'diameter': 1.0,    # m
                'Nx': 50,          # Horizontal grid resolution
                'Ny': 20           # Vertical grid resolution
            },
            'simulation': {
                'dt': 0.001,       # Time step size (s)
                'T': 1.0,          # Total simulation time (s)
                'max_iter': 1000    # Maximum iterations
            },
            'material': {
                'viscosity': 1.0,  # Pa.s
                'density': 1000.0   # kg/m³
            }
        }

# Initialize the session state
init_session_state()

def main():
    # Main title
    st.title("2D Extrusion Flow Solver")
    st.write("A finite volume solver for viscous fluid flow in extrusion processes")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    # Left column for input parameters
    with col1:
        st.header("Simulation Parameters")
        
        # Geometry parameters
        st.subheader("Geometry")
        geometry_col1, geometry_col2 = st.columns(2)
        with geometry_col1:
            st.session_state.params['geometry']['length'] = st.number_input(
                "Channel Length (m)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.params['geometry']['length'],
                step=0.1,
                format="%.2f",
                help="Length of the channel in meters"
            )
            
        with geometry_col2:
            st.session_state.params['geometry']['diameter'] = st.number_input(
                "Channel Diameter (m)",
                min_value=0.01,
                max_value=1.0,
                value=st.session_state.params['geometry']['diameter'],
                step=0.01,
                format="%.2f",
                help="Diameter of the channel in meters"
            )
        
        st.session_state.params['geometry']['Nx'] = st.slider(
            "Horizontal Grid Resolution",
            min_value=10,
            max_value=200,
            value=st.session_state.params['geometry']['Nx'],
            step=5,
            help="Number of grid points in the x-direction"
        )
        
        st.session_state.params['geometry']['Ny'] = st.slider(
            "Vertical Grid Resolution",
            min_value=5,
            max_value=100,
            value=st.session_state.params['geometry']['Ny'],
            step=1,
            help="Number of grid points in the y-direction"
        )
        
        # Simulation parameters
        st.subheader("Simulation Settings")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.session_state.params['simulation']['dt'] = st.number_input(
                "Time Step (s)",
                min_value=1e-6,
                max_value=0.1,
                value=st.session_state.params['simulation']['dt'],
                step=1e-4,
                format="%.6f",
                help="Time step size in seconds"
            )
            
        with sim_col2:
            st.session_state.params['simulation']['T'] = st.number_input(
                "Total Simulation Time (s)",
                min_value=0.1,
                max_value=100.0,
                value=st.session_state.params['simulation']['T'],
                step=0.1,
                format="%.2f",
                help="Total simulation time in seconds"
            )
    
        # Material properties
        st.subheader("Material Properties")
        material_col1, material_col2 = st.columns(2)
        with material_col1:
            st.session_state.params['material']['viscosity'] = st.number_input(
                "Viscosity (Pa·s)",
                min_value=1e-6,
                max_value=1000.0,
                value=st.session_state.params['material']['viscosity'],
                step=0.1,
                format="%.6f"
            )
        with material_col2:
            st.session_state.params['material']['density'] = st.number_input(
                "Density (kg/m³)",
                min_value=1.0,
                max_value=10000.0,
                value=st.session_state.params['material']['density'],
                step=1.0
            )
        
        # Run simulation button
        if st.button("Run Simulation", type="primary"):
            st.session_state.run_simulation = True
            
    with col2:
        st.header("Simulation Output")
        
        # Status messages
        st.subheader("Simulation Parameters")
        
        # Display current parameters in a more organized way
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Geometry**")
            st.write(f"- Length: {st.session_state.params['geometry']['length']:.2f} m")
            st.write(f"- Diameter: {st.session_state.params['geometry']['diameter']:.2f} m")
            st.write(f"- Grid: {st.session_state.params['geometry']['Nx']} × {st.session_state.params['geometry']['Ny']}")
            
            st.write("\n**Material**")
            st.write(f"- Viscosity: {st.session_state.params['material']['viscosity']} Pa·s")
            st.write(f"- Density: {st.session_state.params['material']['density']} kg/m³")
        
        with col2:
            st.write("**Simulation**")
            st.write(f"- Time step: {st.session_state.params['simulation']['dt']:.6f} s")
            st.write(f"- Total time: {st.session_state.params['simulation']['T']} s")
            num_steps = int(st.session_state.params['simulation']['T'] / st.session_state.params['simulation']['dt'])
            st.write(f"- Total steps: {num_steps:,}")
        
        if st.session_state.run_simulation:
            with st.status("Simulation in progress...", expanded=True) as status:
                st.write("\n🚀 **Running simulation...**")
                progress_bar = st.progress(0, "Initializing...")
                
                # Get parameters from session state
                params = st.session_state.params
                Lx = params['geometry']['length']
                Ly = params['geometry']['diameter']
                Nx = params['geometry']['Nx']
                Ny = params['geometry']['Ny']
                mu = params['material']['viscosity']
                rho = params['material']['density']
                dt = params['simulation']['dt']
                max_iter = params['simulation']['max_iter']
                U_inlet = 1.0  # Default inlet velocity (m/s)
                
                # Update progress
                progress_bar.progress(20, "Running solver...")
                
                try:
                    # Call the solver
                    u, v, p, X, Y = solve_stokes_flow(
                        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, 
                        mu=mu, rho=rho, U_inlet=U_inlet,
                        max_iter=max_iter
                    )
                    
                    # Store results in session state
                    st.session_state.results = {
                        'u': u, 'v': v, 'p': p, 'X': X, 'Y': Y
                    }
                    
                    progress_bar.progress(100, "Simulation complete!")
                    st.success("✅ Simulation complete!")
                    
                except Exception as e:
                    st.error(f"❌ Simulation failed: {str(e)}")
                    st.session_state.run_simulation = False
                    return
        
        # Display results if available
        if 'results' in st.session_state and st.session_state.run_simulation:
            results = st.session_state.results
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Velocity Field", "Pressure Field", "Detailed View"])
            
            with tab1:
                st.subheader("Velocity Field")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Calculate velocity magnitude
                vel_mag = np.sqrt(results['u']**2 + results['v']**2)
                
                # Plot velocity magnitude
                c = ax.contourf(results['X'], results['Y'], vel_mag, levels=20, cmap='viridis')
                plt.colorbar(c, label='Velocity Magnitude (m/s)')
                
                # Add streamlines
                ax.streamplot(
                    results['X'], results['Y'], results['u'], results['v'], 
                    color='white', linewidth=0.7, density=1.5, arrowsize=1
                )
                
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_aspect('equal')
                st.pyplot(fig)
                
            with tab2:
                st.subheader("Pressure Field")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Plot pressure field
                c = ax.contourf(
                    results['X'], results['Y'], results['p'], 
                    levels=20, cmap='plasma'
                )
                plt.colorbar(c, label='Pressure (Pa)')
                
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_aspect('equal')
                st.pyplot(fig)
                
            with tab3:
                st.subheader("Detailed Results")
                fig = plot_detailed_results(
                    results['u'], results['v'], results['p'], 
                    results['X'], results['Y']
                )
                st.pyplot(fig)
            
            # Data export
            st.subheader("Data Export")
            
            # Create a DataFrame for better CSV export
            import pandas as pd
            import io
            
            # Create a DataFrame with the data
            df = pd.DataFrame({
                'x (m)': results['X'].flatten(),
                'y (m)': results['Y'].flatten(),
                'u (m/s)': results['u'].flatten(),
                'v (m/s)': results['v'].flatten(),
                'p (Pa)': results['p'].flatten()
            })
            
            # Convert DataFrame to CSV string
            csv = df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="velocity_field.csv",
                mime="text/csv",
                help="Export X, Y, U, V, P data as CSV"
            )
        else:
            # Show placeholder when no results are available
            st.subheader("Results Visualization")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Click 'Run Simulation' to see results",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            st.pyplot(fig)

if __name__ == "__main__":
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    main()

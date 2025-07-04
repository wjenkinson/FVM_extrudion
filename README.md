# 2D Extrusion Flow Solver

A Streamlit-based interface for simulating 2D steady-state Stokes flow in a rectangular channel using the finite volume method.

![Try it here](https://fvsimulator.streamlit.app/)

## Features

- Interactive parameter adjustment through a user-friendly interface
- Real-time visualization of velocity and pressure fields
- Multiple view modes including velocity field, pressure field, and detailed results
- Data export functionality for further analysis
- Responsive design that works on different screen sizes

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/2d-extrusion-flow-solver.git
   cd 2d-extrusion-flow-solver
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

Then open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Project Structure

- `app.py` - Main Streamlit application with the user interface
- `solver.py` - Core finite volume solver implementation
- `requirements.txt` - Python dependencies
- `roadmap.txt` - Development plan and progress tracking
- `examples/` - Sample visualizations and output files
- `.gitignore` - Specifies intentionally untracked files to ignore

## Usage

1. Adjust the simulation parameters in the left sidebar:
   - **Geometry**: Set channel dimensions and grid resolution
   - **Simulation Settings**: Configure time step and maximum iterations
   - **Material Properties**: Define fluid viscosity and density

2. Click "Run Simulation" to start the calculation

3. View the results in the main panel:
   - Switch between different visualization tabs
   - Download the simulation data as CSV

## Example Visualizations



## Development Status

- Phase 1: Basic Streamlit skeleton application (✅ Complete)
- Phase 2: Implementation of parameter sliders (✅ Complete)
- Phase 3: Finite Volume Solver with Fixed Parameters (✅ Complete)
- Phase 4: Integrate Solver with Streamlit (✅ Complete)
- Phase 5-8: Parameter Hook-Up
- Phase 9: Future Extensions, Bingham Plastic, Conical Nozzle

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Scientific computing with [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/)
- Visualization with [Matplotlib](https://matplotlib.org/)

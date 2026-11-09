# AdapFSM

Code and data for the numerical simulations of quantum state estimation with the adaptive Fisher-Symmetric Measurements (AdapFSM) protocol.

The repository contains the implementation of the protocol, the simulation scripts, the generated data and the analysis notebooks used to study its performance for pure states and for slightly mixed states.

### Repository structure

The folders `PureStates` and `Slightly-Mixed-States` contains the code, data and analysis for the estimation of pure states and sligthly mixed states respectively.

Each folder contains:

- Python module with the protocol: FSM construction, simulated measurements, MLE and the adaptive routine (AdapFSM_V2 for pure states, AdapFSM_mixed_V2 for mixed states).
- Simulation notebook that runs the protocol over a grid of dimensions and ensemble sizes and stores the results as CSV files (fid1 for the first estimation, fid2 for the final one; median, mean and standard deviation of the fidelity per random state).
- Data analysis notebook that loads the CSV files, plots the infidelity against N and d together with the Gill-Massar bound, and fits the scaling law.

### Requirements
- Python 3
- NumPy, SciPy, pandas, Matplotlib
- Jupyter
  
### Usage
- Run the simulation notebook of the corresponding folder to generate the CSV files (the output folder can be set with the carpeta argument of the simulation function).
- Open the data analysis notebook, which reads the CSV files and produces the figures and the curve fits.

Note: the modules are imported with from <module> import *, so the notebooks must be run from the same folder as the .py file (and the file name must not contain hyphens).

import numpy as np
import os
import pandas as pd
from AdapFSM_mixed_V2 import*

def AdapFSM_mixed_Simulations(d, n, lamda, n_random_states, n_estimations, n_random_starts=None, carpeta="carpeta"):
    """
    Simulate the AdapFSM algorithm for mixed states and save the results to CSV files.
    
    Parámetros:
    d (int): Dimension of the Hilbert space.
    n (float): Power that determines the number of samples.
    lamda (float): Parameter for the mixed state.
    n_random_states (int): Number of random states to simulate.
    n_estimations (int): Number of estimations for each random state.
    n_random_starts (int, optional): Number of random starts for the optimization. Default is None.
    """
    print(f"Simulating for d={d}, N={n}")

    # Initialize lists to store the data

    datos1 = [] # First stage fidelity data
    datos2 = [] # Second stage fidelity data

    for i in range(n_random_states): # random states of dimension d
        psi = RandomState(d)

        Fid1 = np.zeros(n_estimations)
        Fid2 = np.zeros(n_estimations)

        for j in range(n_estimations): # fidelities for each stage
            Fid1[j] , Fid2[j] = AdapFSM_mixed_V2(psi , lamda , 10**n, n_random_starts)

       # First stage fidelity data
        mean_fid1 = np.mean(Fid1)         ## mean of fidelity
        median_fid1 = np.median(Fid1)   ## median of fidelity
        std_fid1 = np.std(Fid1)           ## standard deviation
        datos1.append([median_fid1 , mean_fid1, std_fid1])

        # Second stage fidelity data
        mean_fid2 = np.mean(Fid2)         ## mean of fidelity
        median_fid2 = np.median(Fid2)   ## median of fidelity
        std_fid2 = np.std(Fid2)           ## standard deviation

        datos2.append([median_fid2 , mean_fid2, std_fid2])

      # Create data frame
    df1 = pd.DataFrame(datos1, columns=["Median", "Mean", "Standard Deviation"])
    df2 = pd.DataFrame(datos2, columns=["Median", "Mean", "Standard Deviation"])

    # Save as CSV
    df1.to_csv(f'{carpeta}/d={d},N={n},fid1.csv', index=False)  
    df2.to_csv(f'{carpeta}/d={d},N={n},fid2.csv', index=False)  

    print("CSV save succesfully.")

    return 

def Simulation(dimensions, powers, lamda, carpeta="carpeta"):
    # Create folder to save CSV files

    os.makedirs(carpeta, exist_ok=True)  # Create the folder if it doesn't exist

    for d in dimensions:
        for n in powers:
            AdapFSM_mixed_Simulations(d, n, lamda, n_random_states=10, n_estimations=10, n_random_starts=0, carpeta=carpeta)

    return
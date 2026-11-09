import numpy as np
import pandas as pd
import os
from AdapFSM_Functions_V2 import*

def AdapFSM_Pure_Simulations(d, n, n_random_states, n_estimations, carpeta='TEST'):
    """
    Simulate the AdapFSM algorithm for pure states with MLE and save the results in CSV files.
    
    Parameters:
    d (int): Dimension of the Hilbert space.
    n (float): Power that determines the number of samples.
    n_random_states (int): Number of different random states to simulate.
    n_estimations (int): Number of different runs for each random state.
    """
    print(f"Simulating for d={d}, N={n}")

    # Initialize lists to store the data

    datos1 = [] # data fidelity of first stage
    datos2 = [] # data fidelity of second stage

    for i in range(n_random_states): # n different random states of dimension d
        psi = RandomState(d)

        Fid1 = np.zeros(n_estimations)
        Fid2 = np.zeros(n_estimations)

        for j in range(n_estimations): # fidelity for n different runs
            Fid1[j] , Fid2[j] = AdapFSM_V2(psi , 10**n)

       #Data for first stage fidelity
        mean_fid1 = np.mean(Fid1)         ## mean fidelity
        median_fid1 = np.median(Fid1)   ## median fidelity
        std_fid1 = np.std(Fid1)           ## standard deviation

        datos1.append([median_fid1 , mean_fid1, std_fid1])

        #Data for second stage fidelity
        mean_fid2 = np.mean(Fid2)         ## mean fidelity
        median_fid2 = np.median(Fid2)   ## median fidelity
        std_fid2 = np.std(Fid2)           ## standard deviation

        datos2.append([median_fid2 , mean_fid2, std_fid2])

      # Create data frame
    df1 = pd.DataFrame(datos1, columns=["Median", "Mean", "Standard Deviation"])
    df2 = pd.DataFrame(datos2, columns=["Median", "Mean", "Standard Deviation"])

    # Save as CSV
    df1.to_csv(f'{carpeta}/d={d},N={n},fid1.csv', index=False)  
    df2.to_csv(f'{carpeta}/d={d},N={n},fid2.csv', index=False)  

    print("CSV saved succesfully.")

    return 

def Simulation(dimensions, powers, n_random_states, n_estimations, carpeta="TEST"):
    # Create folder to save CSV files

    os.makedirs(carpeta, exist_ok=True)  # Create the folder if it doesn't exist

    for d in dimensions:
        for n in powers:
            AdapFSM_Pure_Simulations(d, n, n_random_states, n_estimations, carpeta=carpeta)

    return
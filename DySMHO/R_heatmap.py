## Adding DySMHO repositories to the environment
import sys
# Insert path to directory here
path_to_dysmho = 
sys.path.insert(0, path_to_dysmho+'model')
sys.path.insert(0, path_to_dysmho+'data')

# Loading functions and packages (3D models used for Lorenz) 
import model_3D
import utils_3D
import R_data_generation
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
import pandas as pd
import os

## Basis functions
# Basis functions for dynamics of state 1 
basis_functions_y0 = [lambda y0, y1, y2: 1, 
                      lambda y0, y1, y2: y0,
                      lambda y0, y1, y2: y1,
                      lambda y0, y1, y2: y2,  
                      lambda y0, y1, y2: y0*y1,
                      lambda y0, y1, y2: y0*y2,
                      lambda y0, y1, y2: y1*y2,
                      lambda y0, y1, y2: y0**2,
                      lambda y0, y1, y2: y1**2,
                      lambda y0, y1, y2: y2**2]
# Basis functions for dynamics of state 2
basis_functions_y1 = [lambda y0, y1, y2: 1, 
                      lambda y0, y1, y2: y0,
                      lambda y0, y1, y2: y1,
                      lambda y0, y1, y2: y2,  
                      lambda y0, y1, y2: y0*y1,
                      lambda y0, y1, y2: y0*y2,
                      lambda y0, y1, y2: y1*y2,
                      lambda y0, y1, y2: y0**2,
                      lambda y0, y1, y2: y1**2,
                      lambda y0, y1, y2: y2**2]
# Basis functions for dynamics of state 3
basis_functions_y2 = [lambda y0, y1, y2: 1, 
                      lambda y0, y1, y2: y0,
                      lambda y0, y1, y2: y1,
                      lambda y0, y1, y2: y2,  
                      lambda y0, y1, y2: y0*y1,
                      lambda y0, y1, y2: y0*y2,
                      lambda y0, y1, y2: y1*y2,
                      lambda y0, y1, y2: y0**2,
                      lambda y0, y1, y2: y1**2,
                      lambda y0, y1, y2: y2**2]


# Basis function names
basis_functions_names_y0 = ['1','y0', 'y1', 'y2', 'y0*y1', 'y0*y2', 'y1*y2', 'y0^2', 'y1^2', 'y2^2']
basis_functions_names_y1 = ['1','y0', 'y1', 'y2', 'y0*y1', 'y0*y2', 'y1*y2', 'y0^2', 'y1^2', 'y2^2']
basis_functions_names_y2 = ['1','y0', 'y1', 'y2', 'y0*y1', 'y0*y2', 'y1*y2', 'y0^2', 'y1^2', 'y2^2']
basis_y0 = {'functions': basis_functions_y0, 'names': basis_functions_names_y0} 
basis_y1 = {'functions': basis_functions_y1, 'names': basis_functions_names_y1}
basis_y2 = {'functions': basis_functions_y2, 'names': basis_functions_names_y2}

## Target Sparsity
non_zero_target = [2, 3, 11, 12, 20, 23, 25]
theta_target = pd.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.2, 0.0, 0.0, -5.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

## Simulation parameters
sigma_array = np.linspace(0.025, 0.4, 16)
time_steps_array = np.linspace(10.0, 30.0, 11)
ii = int(os.environ['ii'])
# ii=6
idt =  (ii-1)%11 
ieps = (ii-1)//11 

# Number of test
nTest = 20

# Horizon length for optimization problem (arbitrary time units) 
horizon_length = 5.0  
# Number of sampling time periods taken per MHE step
time_steps = time_steps_array[idt] - horizon_length
# Time step size
dt = 0.005
# Define initial conditions for the 3 states 
y_init = [-6, 5, 0]

xs = np.linspace(0.0, 30.0, int(30.0/dt) + 1)
t_nf, y_nf = R_data_generation.data_gen(xs, y_init, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], False)

# Noise level
sigma = np.std(y_nf,axis=None) * sigma_array[ieps] # Noise level

# Initialize arrays to store results
Success = np.zeros(nTest, dtype=bool)
ModelError = np.zeros(nTest)
ValidationError = np.zeros(nTest)

for i in range(nTest):
    try:
        ## Data Generation
        # Data generation (time grid)
        xs = np.linspace(0.0, horizon_length + time_steps, int((horizon_length + time_steps)/dt) + 1)
        # Data generation (simulating true dynamics on the time grid with addition of white noise )
        t, y = R_data_generation.data_gen(xs, y_init, [0.0, sigma, 0.0, sigma, 0.0, sigma], False) # Noise - Normal [mu, sigma, mu, sigma, mu, sigma]
        # No noise data
        t_nf, y_nf = R_data_generation.data_gen(xs, y_init, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], False)

        ## Learning
        # Creating MHL class (note 3D model used) 
        L_example = model_3D.threeD_MHL(y,t, [basis_y0,basis_y1,basis_y2])
        # Applying SV smoothing 
        L_example.smooth()
        # Pre-processing 1: generates features and tests for Granger Causality 
        L_example.pre_processing_1()
        # Pre-processing 2: uses OLS for initialization and for bounding parameters
        L_example.pre_processing_2(significance = 0.7, plot = False)
        # Calling for main discovery task
        L_example.discover(horizon_length,
                        time_steps,
                        data_step = 100, # Number of time steps between two windows
                        optim_options = {'nfe':80, 'ncp':5}, # ncp = number of collocation pt in FE, nfe = number of FE in the time horizon
                        thresholding_frequency = 10, # Small omega, Big omega is hard coded into line 553 in utils_3D as 4*(small omega)
                        thresholding_tolerance = 1) # This is the threshold, default to 1

        # Check for sparsity
        if L_example.non_zero == non_zero_target :
            Success[i] = True

        # Coefficients Error
        theta_values = pd.DataFrame(L_example.theta_values)
        theta_values.loc[theta_values.iloc[:,-1] == 0, :] = 0
        mean_theta = theta_values.iloc[:,-5:-1].mean(axis=1).to_numpy()
        ModelError[i] = np.linalg.norm(theta_target - mean_theta)/np.linalg.norm(theta_target)
        # Evaluation Error (sort of pointless)
        #L_example.validate(xs, y_nf,plot = False)
        #ValidationError[i] = L_example.error
    except:
        print('Error in test %d' % i)
        Success[i] = np.nan
        ModelError[i] = np.nan
        #ValidationError[i] = np.nan

from scipy.io import savemat
# Save results
results = {
    'Success': Success,
    'ModelError': ModelError,
    'sigma': sigma,
    'time_steps': time_steps,
    'nTest': nTest
}
savemat(str(ii)+'_Rossler_DySMHO.mat', results)

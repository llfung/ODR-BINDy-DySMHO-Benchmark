## Adding DySMHO repositories to the environment
import sys
# Insert path to directory here
path_to_dysmho = 
sys.path.insert(0, path_to_dysmho+'model')
sys.path.insert(0, path_to_dysmho+'data')

# Loading functions and packages (3D models used for Lorenz) 
import model_2D
import utils
import VDP_data_generation
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
import pandas as pd
import os

## Basis functions
# Basis functions for dynamics of state 1 
basis_functions_y0 = [lambda y0,y1: 1, 
                  lambda y0, y1: y0,
                  lambda y0, y1: y1, 
                  lambda y0, y1: y0*y1,
                  lambda y0, y1: y0**2,
                  lambda y0, y1: y1**2,
                  lambda y0, y1: (y0**2)*y1,
                  lambda y0, y1: y0*(y1**2),
                  lambda y0, y1: y0**3,
                  lambda y0, y1: y1**3]

# Basis functions for dynamics of state 2
basis_functions_y1 = [lambda y0,y1: 1, 
                  lambda y0, y1: y0,
                  lambda y0, y1: y1, 
                  lambda y0, y1: y0*y1,
                  lambda y0, y1: y0**2,
                  lambda y0, y1: y1**2,
                  lambda y0, y1: (y0**2)*y1,
                  lambda y0, y1: y0*(y1**2),
                  lambda y0, y1: y0**3,
                  lambda y0, y1: y1**3]

# Basis function names
basis_functions_names_y0 = ['1','y0', 'y1', 'y0*y1', 'y0^2', 'y1^2', '(y0^2)*y1', 'y0*(y1^2)', ' y0^3',' y1^3']
basis_functions_names_y1 = ['1','y0', 'y1', 'y0*y1', 'y0^2', 'y1^2', '(y0^2)*y1', 'y0*(y1^2)', ' y0^3',' y1^3']
basis_y0 = {'functions': basis_functions_y0, 'names': basis_functions_names_y0} 
basis_y1 = {'functions': basis_functions_y1, 'names': basis_functions_names_y1}

## Target Sparsity
non_zero_target = [2,11,12,16]
theta_target = pd.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0,  -1.0, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0])

## Simulation parameters
sigma_array = np.linspace(0.025, 0.4, 16)
time_steps_array = np.linspace(1.0, 12.0, 12)
ii = int(os.environ['ii'])
idt =  (ii-1)%12 
ieps = (ii-1)//12 

# Number of test
nTest = 50
# Noise level
sigma = 1.4492 * sigma_array[ieps] # Noise level
# Number of sampling time periods taken per MHE step
time_steps = time_steps_array[idt] -2.0 

dt = 0.001

# Initialize arrays to store results
Success = np.zeros(nTest, dtype=bool)
ModelError = np.zeros(nTest)
ValidationError = np.zeros(nTest)

for i in range(nTest):
    try:
        ## Data Generation
        # Define initial conditions for the 3 states 
        y_init = [-2,1]

        # Horizon length for optimization problem (arbitrary time units) 
        horizon_length = 2.0  

        # Data generation (time grid)
        xs = np.linspace(0.0, horizon_length + time_steps, int((horizon_length + time_steps)/dt) + 1)
        # Data generation (simulating true dynamics on the time grid with addition of white noise )
        t, y = VDP_data_generation.data_gen(xs, y_init, [0.0, sigma, 0.0, sigma], False) # Noise - Normal [mu, sigma, mu, sigma]
        # No noise data
        t_nf, y_nf = VDP_data_generation.data_gen(xs, y_init, [0.0, 0.0, 0.0, 0.0], False)

        ## Learning
        # Creating MHL class (note 3D model used) 
        VDP_example = model_2D.DySMHO(y,t, [basis_y0,basis_y1])
        # Applying SV smoothing 
        VDP_example.smooth()
        # Pre-processing 1: generates features and tests for Granger Causality 
        VDP_example.pre_processing_1()
        # Pre-processing 2: uses OLS for initialization and for bounding parameters
        VDP_example.pre_processing_2(significance = 0.7, plot = False)
        # Calling for main discovery task
        VDP_example.discover(horizon_length,
                        time_steps,
                        data_step = 100, # Number of time steps between two windows
                        optim_options = {'nfe':80, 'ncp':15}, # ncp = number of collocation pt in FE, nfe = number of FE in the time horizon
                        thresholding_frequency = 10, # Small omega, Big omega is hard coded into line 553 in utils_3D as 4*(small omega)
                        thresholding_tolerance = 1) # This is the threshold, default to 1

        # Check for sparsity
        if VDP_example.non_zero == non_zero_target :
            Success[i] = True

        # Coefficients Error
        theta_values = pd.DataFrame(VDP_example.theta_values)
        theta_values.loc[theta_values.iloc[:,-1] == 0, :] = 0
        mean_theta = theta_values.iloc[:,-5:-1].mean(axis=1).to_numpy()
        ModelError[i] = np.linalg.norm(theta_target - mean_theta)/np.linalg.norm(theta_target)
        # Evaluation Error (sort of pointless)
        #VDP_example.validate(xs, y_nf,plot = False)
        #ValidationError[i] = VDP_example.error
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
savemat(str(ii)+'_VanDerPol_DySMHO.mat', results)
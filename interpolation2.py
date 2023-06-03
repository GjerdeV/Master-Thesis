import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import bspline
import pandas as pd

# B-spline interpolation

def interpolate(eeg_df, bad_channels=[]):  
    for i in range(len(bad_channels)):
        missing_index = eeg_df.columns.get_loc(bad_channels[i])

        left_index = missing_index - 1
        right_index = missing_index + 1

    # Get the values of the neighboring channels
        signal1 = eeg_df.iloc[:, left_index].values
        signal2 = eeg_df.iloc[:, right_index].values

        # new_signal_length = len(signal1)

        # k1 = int(np.ceil(len(signal1) / 2))
        # k2 = int(np.ceil(len(signal2) / 2))
        # t = [0] * k1 + list(range(len(signal1) - k1)) + [len(signal1) - 1] * k2 + list(range(len(signal1), len(signal1) + k2)) + [len(signal1) + len(signal2) - 2] * k1
        
        # # Create a matrix of control points based on the input signals
        # c1 = signal1.tolist() + [0] * k2
        # c2 = [0] * k1 + signal2.tolist()
        # c = np.array([c1, c2])
        
        # # Use the bspline algorithm to interpolate a new signal
        # t_new = np.linspace(0, len(signal1) + len(signal2) - 2, new_signal_length)
        
        # new_signal = np.zeros(new_signal_length)
        # for i in range(c.shape[0]):
        #     new_signal += c[i] * bspline(t_new - t[i], 3)
        k1 = int(np.ceil(len(signal1) / 2))
        k2 = int(np.ceil(len(signal2) / 2))
        t = [0] * k1 + list(range(len(signal1) - k1)) + [len(signal1) - 1] * k2 + list(range(len(signal1), len(signal1) + k2)) + [len(signal1) + len(signal2) - 2] * k1
        
        # Create a matrix of control points based on the input signals
        c1 = signal1.tolist() + [0] * k2
        c2 = [0] * k1 + signal2.tolist()
        c = np.array([c1, c2])
        
        # If new_signal_length is not specified, set it to the length of the input signals
        new_signal_length = len(signal1) + len(signal2) - 1
        
        # Use the bspline algorithm to interpolate a new signal
        t_new = np.linspace(0, new_signal_length)
        new_signal = np.zeros(new_signal_length)
        for i in range(c.shape[0]):
            new_signal += c[i] * bspline(t_new - t[i], 3)
        

        # Replace the missing channel in the DataFrame
        eeg_df.iloc[:, missing_index] = new_signal
    
    return eeg_df
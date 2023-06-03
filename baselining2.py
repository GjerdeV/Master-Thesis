import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import linalg
import numpy as np
from numpy.linalg import norm

def baselining(df, fs=500):
    baseline_start = 0  # Start time of baseline period (in milliseconds)
    baseline_end = 2500    # End time of baseline period (in milliseconds)

    # Calculate baseline mean for each channel
    bsl_eeg = df.copy()
    baseline_mean = df.loc[(bsl_eeg['Timestamp'] >= baseline_start) & (bsl_eeg['Timestamp'] < baseline_end)].mean()

    # Subtract baseline mean from each data point in each channel
    for channel in bsl_eeg.columns:
        if channel != 'Timestamp':  # Skip the time column
            bsl_eeg[channel] -= baseline_mean[channel]
            
    bsl_eeg = bsl_eeg.drop(columns='Timestamp')


    baseline_start_idx = 0
    baseline_end_idx = 2500
    data_start_idx = 0
    data_end_idx = 300000

    # Extract baseline and data segments
    baseline_data = bsl_eeg.iloc[:, baseline_start_idx:baseline_end_idx].values
    data = bsl_eeg.iloc[:, data_start_idx:data_end_idx].values

    # Compute the average of each channel's baseline segment
    baseline_averages = np.mean(baseline_data, axis=1)

    # Subtract the average baseline from each channel's data segment
    baseline_corrected_data = data - baseline_averages[:, np.newaxis]
    baseline_corrected_eeg_data = pd.DataFrame(baseline_corrected_data)
    
    # bsl_eeg = baseline_corrected_eeg_data.copy()
    
    # baseline_mean = df.loc[(bsl_eeg['Timestamp'] >= 0) & (bsl_eeg['Timestamp'] < 300000)].mean()

    # # Subtract baseline mean from each data point in each channel
    # for channel in bsl_eeg.columns:
    #     if channel != 'Timestamp':  # Skip the time column
    #         bsl_eeg[channel] -= baseline_mean[channel]
            
    # bsl_eeg = bsl_eeg.drop(columns='Timestamp')
    # return bsl_eeg
    
    baseline_corrected_eeg_data.columns = [f"Ch{col+1}" for col in range(len(baseline_corrected_eeg_data.columns))]
    
    return baseline_corrected_eeg_data

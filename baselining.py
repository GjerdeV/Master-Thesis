import numpy as np
import pandas as pd

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
    return bsl_eeg
    
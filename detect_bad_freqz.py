import pandas as pd
import numpy as np
from scipy import signal


def detect_bad_channels_freqz(dataframe,fs, window, threshold):
    psd_data = []
    
    # Estimating PSD using Welch method
    for i in range(32):
        f, Pxx = signal.welch(dataframe.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
        psd_data.append({'Ch' + str(i+1): Pxx})
    psd_df = pd.DataFrame(psd_data)
    
    # Converting values to floats
    for i in range(32):
        psd_df['Ch'+str(i+1)] = dataframe['Ch'+str(i+1)].astype(float)
    
    # Compute the standard deviation of each channel
    channel_std = np.std(psd_df.values, axis=0)

    # Compute the interquartile range of the standard deviation
    q1, q3 = np.percentile(channel_std, [25, 75])
    iqr = q3 - q1

    # Define the threshold for bad channels
    threshold_std = q3 + threshold * iqr
    threshold_std2 = q1 - threshold * iqr

    # Find the bad channels
    bad_channels = []
    for channel, std_value in zip(dataframe.columns, channel_std):
        if std_value > threshold_std or std_value < threshold_std2:
            bad_channels.append(channel)

    return bad_channels
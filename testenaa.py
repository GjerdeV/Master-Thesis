import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import interp1d
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

# Defining sample frequency (one sample every 2ms)
fs = 1/0.002
fmin = 0
fmax = 50
window = 'hann'

# Reading the .csv-file and specifying which columns and rows to use to fetch correct data
df = pd.read_csv('eeg_data/001_anon_3.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,45), *range(301015,301018)])
raw_eeg_df = df.drop(columns='Timestamp') # Removing the timestamp-column to work with the data easier
n_channels = len(raw_eeg_df)
# plt.figure(1)

# plt.subplot(311)
# plt.plot(df['Timestamp'], raw_eeg_df) # Plotting of raw data
# plt.title('Raw EEG data')
# plt.xlabel('Time [ms]')
# plt.ylabel('uV')

# plt.subplot(312)
# for i in range(32):
#     f, Pxx = signal.periodogram(raw_eeg_df.iloc[:,i], fs=fs, window=window, scaling='density')
#     mask = (f >= fmin) & (f <= fmax)
#     f = f[mask]
#     Pxx = Pxx[mask]
#     plt.semilogy(f, Pxx)
# plt.title('PSD of raw data using periodogram')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.subplot(313)
# for i in range(32):    
#     f, Pxx = signal.welch(raw_eeg_df.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     plt.semilogy(f, Pxx)
# plt.title('PSD of raw data using Welch method')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.tight_layout()

############################################################
################## Baseline correction #####################

baseline_start = 0  # Start time of baseline period (in milliseconds)
baseline_end = 2000    # End time of baseline period (in milliseconds)

# Calculate baseline mean for each channel
bsl_eeg = df.copy()
baseline_mean = df.loc[(bsl_eeg['Timestamp'] >= baseline_start) & (bsl_eeg['Timestamp'] < baseline_end)].mean()

# Subtract baseline mean from each data point in each channel
for channel in bsl_eeg.columns:
    if channel != 'Timestamp':  # Skip the time column
        bsl_eeg[channel] -= baseline_mean[channel]
        
bsl_eeg = bsl_eeg.drop(columns='Timestamp')

# plt.figure(2)

# plt.subplot(211)
# plt.plot(df['Timestamp'], bsl_eeg)
# plt.title('Baseline corrected data')
# plt.xlabel('Time [ms]')
# plt.ylabel('uV')

# plt.subplot(212)
# window = 'hann'
# for i in range(32):
#     f, Pxx = signal.welch(bsl_eeg.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     plt.semilogy(f, Pxx, label='Channel {}'.format(i+1))
# plt.title('PSD for baseling corrected data')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.tight_layout()

##############################################################
# Re-referencing not necessary as already done in iMotions before data export, through earclip
################ Detection of bad channels ###################

# def detect_bad_channels(dataframe, threshold):

#     # Compute the standard deviation of each channel
#     channel_std = np.std(dataframe.values, axis=0)

#     # Compute the interquartile range of the standard deviation
#     q1, q3 = np.percentile(channel_std, [25, 75])
#     iqr = q3 - q1

#     # Define the threshold for bad channels
#     threshold_std = q3 + threshold * iqr

#     # Find the bad channels
#     bad_channels = []
#     for channel, std_value in zip(dataframe.columns, channel_std):
#         if std_value > threshold_std:
#             bad_channels.append(channel)

#     if len(bad_channels) == 0:
#         print("No bad channels detected.")
#     else:
#         print("The following channels are potentially bad:")
#         print(bad_channels)

#     return bad_channels

# bad_channels = detect_bad_channels(bsl_eeg, threshold=0.5)

##############################################################
################# FREQZ ######################################
def detect_bad_channels_freqz(dataframe, threshold):
    psd_data = []
    for i in range(32):
        f, Pxx = signal.welch(dataframe.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
        psd_data.append({'Ch' + str(i+1): Pxx})
    print(psd_data)
    psd_df = pd.DataFrame(psd_data)
    for i in range(32):
        psd_df['Ch'+str(i+1)] = bsl_eeg['Ch'+str(i+1)].astype(float)
    
    # Compute the standard deviation of each channel
    channel_std = np.std(psd_df.values, axis=0)

    # Compute the interquartile range of the standard deviation
    q1, q3 = np.percentile(channel_std, [25, 75])
    iqr = q3 - q1

    # Define the threshold for bad channels
    threshold_std = q3 + threshold * iqr
    th2 = q1 - threshold * iqr

    # Find the bad channels
    bad_channels = []
    for channel, std_value in zip(dataframe.columns, channel_std):
        if std_value > threshold_std or std_value < th2:
            bad_channels.append(channel)

    if len(bad_channels) == 0:
        print("No bad channels detected.")
    else:
        print("The following channels are potentially bad:")
        print(bad_channels)

    return bad_channels

bad_channels = detect_bad_channels_freqz(bsl_eeg, threshold=2.0)


###############################################################

# def detect_bad_channels(dataframe, threshold):
    
#     psd_data = []
#     for i in range(32):
#         f, Pxx = signal.welch(dataframe.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#         psd_data.append({'Ch' + str(i+1): Pxx})
#     print(psd_data)
#     psd_df = pd.DataFrame(psd_data)
#     for i in range(32):
#         psd_df['Ch'+str(i+1)] = df['Ch'+str(i+1)].astype(float)
    
#     # Calculate the median and median absolute deviation (MAD) for each channel
#     median = psd_df.median()
#     mad = psd_df.mad()
    
#     # Calculate the deviation from the median for each channel
#     for channel in bsl_eeg.columns:
#         dataframe[channel] -= baseline_mean[channel]
    
#     deviation = np.abs(dataframe - median)
    
#     # Calculate the z-score for each channel
#     z_score = deviation / mad
    
#     # Find the channels with z-scores above the threshold
#     bad_channels = list(z_score[z_score > threshold].index)
    
#     if len(bad_channels) == 0:
#         print("No bad channels detected.")
#     else:
#         print("The following channels are potentially bad:")
#         print(bad_channels)
    
#     return bad_channels

# bad_channels = detect_bad_channels(bsl_eeg, 3.5)


################ Visualizing bad channels ####################

plt.figure(3)
for i in range(32):
    check = 'Ch' + str(i+1)
    if check in bad_channels:
        plt.plot(df['Timestamp'], bsl_eeg[check], 'k-') 
    else:
        plt.plot(df['Timestamp'], bsl_eeg[check])
    plt.label='Channel{}'.format(i+1)
plt.title('Detection of bad channels in time domain')
plt.xlabel('Time [ms]')
plt.ylabel('uV')
plt.legend()

plt.figure(4)
for i in range(32):
    check = 'Ch' + str(i+1)
    f, Pxx = signal.welch(bsl_eeg.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
    if check in bad_channels:
        plt.semilogy(f, Pxx, 'k-') 
    else:
        plt.semilogy(f, Pxx)
plt.title('Detection of bad channels in frequency domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

##############################################################
################ Clearing bad channels #######################

eeg_df = bsl_eeg.copy()
# eeg_df[bad_channels] = np.nan
# eeg_df.info()

##############################################################
################ Channel interpolation #######################


# for i in range(len(bad_channels)): # loop is new
#     missing_index = eeg_df.columns.get_loc(bad_channels[i]) # in a list, have to check for all that is missing

# # Find the indices of the neighboring channels, (((((((((((((if next index is nan, then check next)))))))))))))))))
#     left_index = missing_index - 1
#     right_index = missing_index + 1

#     # Get the values of the neighboring channels
#     left_channel = eeg_df.iloc[:, left_index].values
    
#     right_channel = eeg_df.iloc[:, right_index].values

#     # Interpolate the missing channel
#     interp_func = interp1d([left_index, right_index], [left_channel, right_channel], axis=0)
#     missing_channel_values = interp_func(missing_index)

#     # Replace the missing channel in the DataFrame
#     eeg_df.iloc[:, missing_index] = missing_channel_values

# plt.figure(5)
# for i in range(32):
#     check = 'Ch' + str(i+1)
#     if check in bad_channels:
#         plt.plot(df['Timestamp'], eeg_df[check], 'k-') 
#     else:
#         plt.plot(df['Timestamp'], eeg_df[check])
#     plt.label='Channel{}'.format(i+1)
# plt.xlabel('Time [ms]')
# plt.ylabel('uV')
# plt.legend()

# plt.figure(6)
# for i in range(32):
#     check = 'Ch' + str(i+1)
#     f, Pxx = signal.welch(eeg_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     if check in bad_channels:
#         plt.semilogy(f, Pxx, 'k-') 
#     else:
#         plt.semilogy(f, Pxx)
#     plt.label='Channel{}'.format(i+1)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')
# plt.legend()

##############################################################




plt.show()

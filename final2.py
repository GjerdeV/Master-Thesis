import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from baselining2 import baselining
from detect_bad_chnl import detect_bad_channels
from detect_bad_freqz import detect_bad_channels_freqz
# from ica import run_ica
from interpolation import interpolate
from epoching import epoch_signal

####### Notice ######
# This file is only used for testing different methods while developing code. Use this to test out baselining 2

# Defining sample frequency (one sample every 2ms)
fs = 1/0.002
fmin = 0
fmax = 50
window = 'hann'

# Reading the .csv-file and specifying which columns and rows to use to fetch correct data

# Anon 1
# df = pd.read_csv('eeg_data/001_anon_1.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,340), *range(2257553,2257556)])
# Godt eksempel på fjerning av bad channels med threshold: 2 og 2:
# df = pd.read_csv('eeg_data/001_anon_1.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,340), *range(300340,2257556)])

# Anon 2, range not set yet
# df = pd.read_csv('eeg_data/001_anon_2.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(301015,2257264)])
# Finner ikke bad channel + vil ikke filtreres, finner bad channels ved threshold=0.5, men fortsatt ikke filtreres:
# df = pd.read_csv('eeg_data/001_anon_2.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(300050,2257264)])

# Anon 3
# df = pd.read_csv('eeg_data/001_anon_3.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,45), *range(301015,301018)])
# test:
df = pd.read_csv('eeg_data/001_anon_3.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,45), *range(300045,301018)])

# Anon 4 - plotting doesn't start at 0????
# df = pd.read_csv('eeg_data/001_anon_4.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(301023,301026)])

# Anon 5 - bad channels freqz 0.5 atm, 1.5 vanlig, plotting doesn't start at 0 here either
# df = pd.read_csv('eeg_data/001_anon_5.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,44), *range(301022,301026)])

df['Timestamp'] -= df['Timestamp'][0] # Setting time to start at 0
df['Timestamp'] = df['Timestamp']/1000 # Converting time unit from ms to s
raw_eeg_df = df.drop(columns='Timestamp') # Removing the timestamp-column to work with the data easier
n_channels = len(raw_eeg_df)

plt.figure(1)

plt.subplot(311)
plt.plot(df['Timestamp'], raw_eeg_df) # Plotting of raw data
plt.title('Raw EEG data')
plt.xlabel('Time [s]')
plt.ylabel('uV')

plt.subplot(312)
for i in range(32):
    f, Pxx = signal.periodogram(raw_eeg_df.iloc[:,i], fs=fs, window=window, scaling='density')
    mask = (f >= fmin) & (f <= fmax)
    f = f[mask]
    Pxx = Pxx[mask]
    plt.semilogy(f, Pxx)
plt.title('PSD of raw data using periodogram')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

plt.subplot(313)
for i in range(32):    
    f, Pxx = signal.welch(raw_eeg_df.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
    plt.semilogy(f, Pxx)
plt.title('PSD of raw data using Welch method')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

plt.tight_layout()

############################################################
################## Baseline correction #####################

# bsl_eeg = baselining(df)

from BaselineRemoval import BaselineRemoval

polynomial_degree=2 #only needed for Modpoly and IModPoly algorithm
bsl_eeg = raw_eeg_df.copy()
for channel in range(1, 33):
    channel_name = 'Ch' + str(channel)
    baseObj=BaselineRemoval(df[channel_name])
    bsl_eeg[channel_name] = Zhangfit_output=baseObj.ZhangFit()
    

# baseObj=BaselineRemoval(df['Ch22'])
# Modpoly_output=baseObj.ModPoly(polynomial_degree)
# Imodpoly_output=baseObj.IModPoly(polynomial_degree)
# Zhangfit_output=baseObj.ZhangFit()

# bsl_eeg = Zhangfit_output

# import numpy as np
# from scipy import sparse
# from scipy.sparse.linalg import spsolve

# def baseline_als(y, lam, p, niter=10):
#   L = len(y)
#   D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
#   w = np.ones(L)
#   for i in range(niter):
#     W = sparse.spdiags(w, 0, L, L)
#     Z = W + lam * D.dot(D.transpose())
#     z = spsolve(Z, w*y)
#     w = p * (y > z) + (1-p) * (y < z)
#   return z

# bsl_eeg = baseline_als(raw_eeg_df['Ch1'], 10000000, 0.1)
# bsl_eeg = baselining(raw_eeg_df)

plt.figure(2)

plt.subplot(211)
plt.plot(df['Timestamp'], bsl_eeg)
plt.title('Baseline corrected data')
plt.xlabel('Time [s]')
plt.ylabel('uV')

plt.subplot(212)
for i in range(32):
    f, Pxx = signal.welch(bsl_eeg.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
    plt.semilogy(f, Pxx, label='Channel {}'.format(i+1))
plt.title('PSD for baseling corrected data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

plt.tight_layout()

bsl_eeg.info()
# plt.show()
##############################################################
# Re-referencing not necessary as already done in iMotions before data export, through earclip
################ Detection of bad channels ###################
bad_channels = []

# Anon 5, Threshold = 1.5
bad_channels.extend(detect_bad_channels(bsl_eeg, threshold=2.0))

###### Automatic detection of bad channels based on PSD ######
# Anon 5, Threshold = 0.5
bad_channels.extend(detect_bad_channels_freqz(bsl_eeg, fs, window, threshold=2.0))

if len(bad_channels) == 0:
    print("No bad channels detected.")
else:
    print("The following channels are potentially bad:")
    print(bad_channels)

################ Visualizing bad channels ####################

plt.figure(3)
for i in range(32):
    check = 'Ch' + str(i+1)
    if check in bad_channels:
        # plt.plot(df['Timestamp'], bsl_eeg[check], 'k-') 
        plt.plot(bsl_eeg[check], 'k-') 
    else:
        # plt.plot(df['Timestamp'], bsl_eeg[check])
        plt.plot(bsl_eeg[check])
    plt.label='Channel{}'.format(i+1)
plt.title('Detection of bad channels in time domain')
plt.xlabel('Time [s]')
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
    plt.label='Channel{}'.format(i+1)
plt.title('Detection of bad channels in frequency domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

##############################################################
################ Clearing bad channels #######################

eeg_df = bsl_eeg.copy()
eeg_df[bad_channels] = np.nan
# eeg_df.info()

##############################################################
################ Channel interpolation #######################

eeg_df = interpolate(eeg_df, bad_channels)
eeg_df.info()

plt.figure(5)
for i in range(32):
    check = 'Ch' + str(i+1)
    if check in bad_channels:
        # plt.plot(df['Timestamp'], eeg_df[check], 'k-') 
        plt.plot(eeg_df[check], 'k-') 
    else:
        # plt.plot(df['Timestamp'], eeg_df[check])
        plt.plot(eeg_df[check])
    plt.label='Channel{}'.format(i+1)
plt.xlabel('Time [s]')
plt.title('EEG data with interpolated values')
plt.ylabel('uV')
plt.legend()

plt.figure(6)
for i in range(32):
    check = 'Ch' + str(i+1)
    f, Pxx = signal.welch(eeg_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
    if check in bad_channels:
        plt.semilogy(f, Pxx, 'k-') 
    else:
        plt.semilogy(f, Pxx)
    plt.label='Channel{}'.format(i+1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
# plt.legend()

##############################################################
################### Notch filter, Hz = 50Hz ##################

notch_freq = 50 # Hz
Q = 30.0  # Quality factor
w0 = notch_freq / (fs / 2)  # Normalized frequency

# Create the notch filter
d, c = signal.iirnotch(w0, Q)

# Apply the notch filter to each channel of the EEG data
notchf_eeg_df = eeg_df.copy()
for channel in range(1, 33):
    channel_name = 'Ch' + str(channel)
    notchf_eeg_df[channel_name] = signal.filtfilt(d, c, eeg_df[channel_name])

fig7 = plt.figure(7)
fig7.suptitle('Notch filtered EEG data at 50Hz')

plt.subplot(211)
plt.plot(df['Timestamp'], notchf_eeg_df)
plt.title('Notch filtered EEG data')
plt.xlabel('Time [s]')
plt.ylabel('uV')

plt.subplot(212)
for i in range(32):
    f, Pxx = signal.welch(notchf_eeg_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
    mask = (f >= fmin) & (f <= fmax)
    f = f[mask]
    Pxx = Pxx[mask]
    plt.semilogy(f, Pxx)
plt.title('Notch filtered EEG data in frequency domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
    
#################################################################
################### Bandpass filtering ##########################

f_low = 4  # in Hz
f_high = 45  # in Hz<
order = 4

b, a = signal.butter(order, [f_low/(fs/2), f_high/(fs/2)], btype='bandpass')

filtered_eeg_df = notchf_eeg_df.copy()
# filtered_eeg_df = eeg_df.copy()
for chnl in range(1, 33):
    chnl_name = 'Ch' + str(chnl)
    filtered_eeg_df[chnl_name] = signal.filtfilt(b, a, notchf_eeg_df[chnl_name])
    # filtered_eeg_df[chnl_name] = signal.filtfilt(b, a, eeg_df[chnl_name])

fig8 = plt.figure(8)
fig8.suptitle('Bandpass filtered data in range: ' + str(f_low) + '-' + str(f_high) + 'Hz')

plt.subplot(211)
plt.plot(df['Timestamp'], filtered_eeg_df)
# plt.plot(filtered_eeg_df)
plt.title('Filtered EEG data')
plt.xlabel('Time [s]')
plt.ylabel('uV')

plt.subplot(212)
for i in range(32):
    f, Pxx = signal.welch(filtered_eeg_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
    # mask = (f >= fmin) & (f <= fmax)
    # f = f[mask]
    # Pxx = Pxx[mask]
    plt.semilogy(f, Pxx)
plt.title('Filtered EEG data in frequency domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

############### Showing why notchf is necessary #################
start = 0 #0
stop = 300000 #1000

filtnonotch = eeg_df.copy()

for chnl in range(1, 33):
    chnl_name = 'Ch' + str(chnl)
    filtnonotch[chnl_name] = signal.filtfilt(b, a, eeg_df[chnl_name])

fig9 = plt.figure(9)
fig9.suptitle('Bandpass filtered data in range: ' + str(f_low) + '-' + str(f_high) + 'Hz')

plt.subplot(221)
plt.plot(df['Timestamp'].iloc[start:stop], filtered_eeg_df['Ch1'].iloc[start:stop])
plt.title('Filtered EEG data')
plt.xlabel('Time [s]')
plt.ylabel('uV')

plt.subplot(222)
plt.plot(df['Timestamp'].iloc[start:stop], filtnonotch['Ch1'].iloc[start:stop])
plt.title('Filtered EEG data without notch filter')
plt.xlabel('Time [s]')
plt.ylabel('uV')

plt.subplot(223)
f11, Pxx11 = signal.welch(filtered_eeg_df['Ch1'].iloc[start:stop], fs, window=window, scaling='density')
plt.semilogy(f11, Pxx11)
plt.title('Filtered EEG data in frequency domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

plt.subplot(224)
f22, Pxx22 = signal.welch(filtnonotch['Ch1'].iloc[start:stop], fs, window=window, scaling='density')
plt.semilogy(f22, Pxx22)
plt.title('Filtered EEG data in frequency domain without notch filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

#################################################################
######################## Average signal #########################

average_eeg = np.mean(filtered_eeg_df, axis=1)
# test_df = average_eeg
# average_eeg = np.mean(eeg_df, axis=1)
test_df = pd.DataFrame({"Average Signal": average_eeg})

# test_df = test_df2.copy()

# test_df['Average Signal'] = signal.filtfilt(b, a, test_df['Average Signal'])

plt.figure(10)
plt.plot(df['Timestamp'], test_df['Average Signal'])
plt.title('Averaged EEG data in time domain')
plt.xlabel('Time [s]')
plt.ylabel('uV')

plt.figure(11)
f_avg, Pxx_avg = signal.welch(test_df['Average Signal'], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
plt.semilogy(f_avg, Pxx_avg)
plt.title('Averaged EEG data in frequency domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')

#################################################################
######################### Epoching ##############################
#################################################################
######################### ICA ###################################

plt.show()
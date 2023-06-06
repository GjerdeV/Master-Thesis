import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import interp1d
from scipy import signal
from sklearn.decomposition import FastICA
from baselining import baselining
from detect_bad_chnl import detect_bad_channels
from detect_bad_freqz import detect_bad_channels_freqz
from interpolation import interpolate
from epoching import epoch_signal
from ssqueezepy import ssq_cwt, cwt
from ssqueezepy.visuals import imshow
from ssqueezepy.experimental import scale_to_freq
from feature_extraction import featext, featest


######## Notice #########
# This file is meant for signal processing and feature extraction for the 300 000 sample data
# Before running, make sure that the correct data is imported, correct thresholds are chosen and
# correct .npy-file name is chosen.
# For the longer data sets, please see the file "longer_data_preprocessing.py"

# Defining sample frequency (one sample every 2ms)
fs = 1/0.002
# fmin and fmax are used if just certain frequencies are of interest in the PSD plots
fmin = 0
fmax = 50
# Defining window function as hann
window = 'hann'

# Reading the .csv-file and specifying which columns and rows to use to fetch correct data

# Anon 1
# All data:
# df = pd.read_csv('eeg_data/001_anon_1.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,340), *range(2257553,2257556)])
# Godt eksempel på fjerning av bad channels med threshold: 2 og 2:
# 300 000 sample data:
# df = pd.read_csv('eeg_data/001_anon_1.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,340), *range(300340,2257556)])

# Anon 2
# All data:
# df = pd.read_csv('eeg_data/001_anon_2.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(301015,2257264)])
# Finner ikke bad channel + vil ikke filtreres, finner bad channels ved threshold=0.5, men fortsatt ikke filtreres:
# df = pd.read_csv('eeg_data/001_anon_2.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(300050,2257264)])

# Anon 3
# All data:
# df = pd.read_csv('eeg_data/001_anon_3.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,45), *range(301015,301018)])
# 300 000 sample data:
# df = pd.read_csv('eeg_data/001_anon_3.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,45), *range(300045,301018)])

# Anon 4
# All data:
# df = pd.read_csv('eeg_data/001_anon_4.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(301023,301026)])
# 300 000 sample data:
# df = pd.read_csv('eeg_data/001_anon_4.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(300050,301026)])

# Anon 5
# All data:
# df = pd.read_csv('eeg_data/001_anon_5.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,44), *range(301022,301026)])
# 300 000 sample data:
df = pd.read_csv('eeg_data/001_anon_5.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,44), *range(300044,301026)])

df['Timestamp'] -= df['Timestamp'][0] # Setting time to start at 0
df['Timestamp'] = df['Timestamp']/1000 # Converting time unit from ms to s
raw_eeg_df = df.drop(columns='Timestamp') # Removing the timestamp-column to work with the data easier
n_channels = len(raw_eeg_df)

# plt.figure(1)

# plt.subplot(311)
# plt.plot(df['Timestamp'], raw_eeg_df) # Plotting of raw data
# plt.grid(True)
# plt.title('Raw EEG data')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')

# plt.subplot(312)
# for i in range(32):
#     f, Pxx = signal.periodogram(raw_eeg_df.iloc[:,i], fs=fs, window=window, scaling='density')
#     ## If a smaller representation of the periodogram is desired, then uncomment lines below:
#     # mask = (f >= fmin) & (f <= fmax)
#     # f = f[mask]
#     # Pxx = Pxx[mask]
#     plt.semilogy(f, Pxx)
# plt.grid(True)
# plt.title('PSD of raw data using periodogram')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.subplot(313)
# for i in range(32):    
#     f, Pxx = signal.welch(raw_eeg_df.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     plt.semilogy(f, Pxx)
# plt.grid(True)
# plt.title('PSD of raw data using Welch method')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.tight_layout()
# plt.show()
############################################################
################## Baseline correction #####################

bsl_eeg = baselining(df)

# plt.figure(2)

# plt.subplot(211)
# plt.plot(df['Timestamp'], bsl_eeg)
# plt.title('Baseline corrected data')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')

# plt.subplot(212)
# for i in range(32):
#     f, Pxx = signal.welch(bsl_eeg.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     plt.semilogy(f, Pxx, label='Channel {}'.format(i+1))
# plt.title('PSD for baseling corrected data')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.tight_layout()

##############################################################
################ Detection of bad channels ###################
bad_channels = []

# Anon 1, Threshold = 2.0
# Anon 2, Threshold = 2.0
# Anon 3, Threshold = 4.0
# Anon 4, Threshold = 2.0
# Anon 5, Threshold = 1.5
bad_channels.extend(detect_bad_channels(bsl_eeg, threshold=1.5))

###### Automatic detection of bad channels based on PSD ######
# Anon 1, Threshold = 2.0
# Anon 2, Threshold = 2.0
# Anon 3, Threshold = 4.0
# Anon 4, Threshold = 2.0
# Anon 5, Threshold = 0.5
bad_channels.extend(detect_bad_channels_freqz(bsl_eeg, fs, window, threshold=0.5))

# bad_channels = list(set(bad_channels)) # Removing any duplicates

if len(bad_channels) == 0:
    print("No bad channels detected.")
else:
    print("The following channels are potentially bad:")
    print(bad_channels)

################ Visualising bad channels ####################

# plt.figure(3)
# for i in range(32):
#     check = 'Ch' + str(i+1)
#     if check in bad_channels:
#         plt.plot(df['Timestamp'], bsl_eeg[check], 'k-') 
#     else:
#         plt.plot(df['Timestamp'], bsl_eeg[check])
#     plt.label='Channel{}'.format(i+1)
# plt.title('Detection of bad channels in time domain')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')
# plt.legend()

# plt.figure(4)
# for i in range(32):
#     check = 'Ch' + str(i+1)
#     f, Pxx = signal.welch(bsl_eeg.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     if check in bad_channels:
#         plt.semilogy(f, Pxx, 'k-') 
#     else:
#         plt.semilogy(f, Pxx)
#     plt.label='Channel{}'.format(i+1)
# plt.title('Detection of bad channels in frequency domain')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.show()
##############################################################
################ Clearing bad channels #######################

eeg_df = bsl_eeg.copy()
eeg_df[bad_channels] = np.nan
eeg_df.info()

##############################################################
################ Channel interpolation #######################

eeg_df = interpolate(eeg_df, bad_channels)
eeg_df.info()

# plt.figure(5)
# for i in range(32):
#     check = 'Ch' + str(i+1)
#     if check in bad_channels:
#         plt.plot(df['Timestamp'], eeg_df[check], 'k-') 
#     else:
#         plt.plot(df['Timestamp'], eeg_df[check])
#     plt.label='Channel{}'.format(i+1)
# plt.xlabel('Time [s]')
# plt.title('EEG data with interpolated values')
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

# plt.show()
##############################################################
################### Notch filter, Hz = 50Hz ##################

notch_freq = 50 # Frequency to notch filter out of the data
Q = 30.0  # Quality factor
w0 = notch_freq / (fs / 2)  # Normalized frequency

# Create the notch filter
d, c = signal.iirnotch(w0, Q)

# Apply the notch filter to each channel of the EEG data
notchf_eeg_df = eeg_df.copy()
for channel in range(1, 33):
    channel_name = 'Ch' + str(channel)
    notchf_eeg_df[channel_name] = signal.filtfilt(d, c, eeg_df[channel_name])

# fig7 = plt.figure(7)
# fig7.suptitle('Notch filtered EEG data at 50Hz')

# plt.subplot(211)
# plt.plot(df['Timestamp'], notchf_eeg_df)
# plt.title('Notch filtered EEG data')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')

# plt.subplot(212)
# for i in range(32):
#     f, Pxx = signal.welch(notchf_eeg_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     ## If a smaller representation of the periodogram is desired, then uncomment lines below:
#     # mask = (f >= fmin) & (f <= fmax)
#     # f = f[mask]
#     # Pxx = Pxx[mask]
#     plt.semilogy(f, Pxx)
# plt.title('Notch filtered EEG data in frequency domain')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (dB/Hz)')

# plt.show()
#################################################################
################### Bandpass filtering ##########################

f_low = 4  # in Hz
f_high = 45  # in Hz<
order = 4

b, a = signal.butter(order, [f_low/(fs/2), f_high/(fs/2)], btype='bandpass')
# b, a = signal.butter(order, 45/(fs/2), 'low')

filtered_eeg_df = notchf_eeg_df.copy()
for chnl in range(1, 33):
    chnl_name = 'Ch' + str(chnl)
    # filtered_eeg_df[chnl_name] = signal.lfilter(b, a, notchf_eeg_df[chnl_name])
    filtered_eeg_df[chnl_name] = signal.filtfilt(b, a, notchf_eeg_df[chnl_name])
    # filtered_eeg_df[chnl_name] = signal.sosfiltfilt(sos, notchf_eeg_df[chnl_name])


# fig8 = plt.figure(8)
# fig8.suptitle('Bandpass filtered data in range: ' + str(f_low) + '-' + str(f_high) + 'Hz')

# plt.subplot(311)
# plt.plot(df['Timestamp'], filtered_eeg_df)
# plt.grid(True)
# plt.title('Filtered EEG data')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')

# plt.subplot(312)
# for i in range(32):
#     f, Pxx = signal.welch(filtered_eeg_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     ## If a smaller representation of the periodogram is desired, then uncomment lines below:
#     # mask = (f >= fmin) & (f <= fmax)
#     # f = f[mask]
#     # Pxx = Pxx[mask]
#     plt.semilogy(f, Pxx)
#     plt.grid(True)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.subplot(313)
# for i in range(32):
#     f, Pxx = signal.welch(filtered_eeg_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     ## If a smaller representation of the periodogram is desired, then uncomment lines below:
#     mask = (f >= fmin) & (f <= fmax)
#     f = f[mask]
#     Pxx = Pxx[mask]
#     plt.semilogy(f, Pxx)
#     plt.grid(True)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.show()
############### Showing why notchf is necessary #################
start = 0
stop = 1000

filtnonotch = eeg_df.copy()

for chnl in range(1, 33):
    chnl_name = 'Ch' + str(chnl)
    filtnonotch[chnl_name] = signal.filtfilt(b, a, eeg_df[chnl_name])

# fig9 = plt.figure(9)
# fig9.suptitle('Bandpass filtered data in range: ' + str(f_low) + '-' + str(f_high) + 'Hz')

# plt.subplot(221)
# plt.plot(df['Timestamp'].iloc[start:stop], filtered_eeg_df['Ch1'].iloc[start:stop])
# plt.title('Filtered EEG data')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')

# plt.subplot(222)
# plt.plot(df['Timestamp'].iloc[start:stop], filtnonotch['Ch1'].iloc[start:stop])
# plt.title('Filtered EEG data without notch filter')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')

# plt.subplot(223)
# f11, Pxx11 = signal.welch(filtered_eeg_df['Ch1'].iloc[start:stop], fs, window=window, scaling='density')
# plt.semilogy(f11, Pxx11)
# plt.title('Filtered EEG data in frequency domain')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

# plt.subplot(224)
# f22, Pxx22 = signal.welch(filtnonotch['Ch1'].iloc[start:stop], fs, window=window, scaling='density')
# plt.semilogy(f22, Pxx22)
# plt.title('Filtered EEG data in frequency domain without notch filter')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

#################################################################
######################## Average signal #########################

average_eeg = np.mean(filtered_eeg_df, axis=1)
avg_df = pd.DataFrame({"Average Signal": average_eeg})

# plt.figure(10)
# plt.plot(df['Timestamp'], avg_df['Average Signal'], color='b')
# plt.title('Averaged EEG data in time domain')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')

# plt.figure(11)
# f_avg, Pxx_avg = signal.welch(avg_df['Average Signal'], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
# plt.semilogy(f_avg, Pxx_avg)
# plt.title('Averaged EEG data in frequency domain')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')

#################################################################
######################### Epoching ##############################

epoch_duration = 5  # seconds
epochs = epoch_signal(avg_df, fs, epoch_duration)

# num_epochs_per_figure = 5
# total_epochs = len(epochs)
# num_figures = (total_epochs + num_epochs_per_figure - 1) // num_epochs_per_figure

# for i in range(num_figures):
#     start_epoch = i * num_epochs_per_figure
#     end_epoch = min((i + 1) * num_epochs_per_figure, total_epochs)
    
#     num_epochs_in_figure = end_epoch - start_epoch
#     fig, axs = plt.subplots(num_epochs_in_figure, 1, figsize=(8, 2*num_epochs_in_figure))
    
#     for j in range(num_epochs_in_figure):
#         epoch_index = start_epoch + j
#         epoch = epochs[epoch_index]
#         time = np.arange(len(epoch)) / fs
#         start_time = epoch_index * epoch_duration
#         end_time = start_time + epoch_duration
#         axs[j].plot(time + start_time, epoch)
#         axs[j].set_ylabel(f'Epoch {epoch_index+1}')
#         axs[j].grid(True)
    
#     plt.xlabel('Time[s]')
#     plt.tight_layout()

fig, ax = plt.subplots(figsize=(12, 4))

# Plotting the averaged signal
time = np.arange(len(avg_df)) / fs
ax.plot(time, avg_df, label='Group Averaged Signal')

# Plotting red vertical lines for epoch boundaries
for i in range(len(epochs) - 1):
    epoch_end_time = (i + 1) * epoch_duration
    ax.axvline(x=epoch_end_time, color='red', linestyle='-', alpha=0.8)

ax.set_xlabel('Time [s]')
ax.set_ylabel('uV')
ax.set_title('Group Averaged Signal with Epoch Boundaries')
ax.legend()

t = np.arange(0, 5, 0.002, dtype='float64') # Defining the time-axis for epochs
plt.figure()
plt.plot(t, epochs[0].iloc[:,0].to_numpy())
plt.grid(True)
plt.title('Single epoch data')
plt.xlabel('Time [s]')
plt.ylabel('uV')

# plt.show()
#######################################################################
################# Wavelet Transformation plot #########################
t = np.arange(0, 5, 0.002, dtype='float64') # Defining the time-axis for epochs
sig = epochs[82].iloc[:,0].to_numpy()
Tx, *_ = ssq_cwt(sig, 'morlet', t=t)
Wx, scales = cwt(sig, 'morlet', t=t)
freqs = scale_to_freq(scales, 'morlet', N=500, fs=500)

# Illustrating the complex Morlet wavelet
plt.figure()
# imshow(Wx, xticks=t, yticks=freqs, abs=1, ylabel="Frequency [Hz]", xlabel="Time [s]")
# Illustrating the squeezed version of complex Morlet Wavelet
plt.figure()
# imshow(Tx, abs=1, xticks=t, yticks=freqs, title="SSQ CWT", ylabel="Frequency [Hz]", xlabel="Time [s]")
#################################################################
 
# plt.show() 
#################################################################

################################ Bad epochs for anon3 ###########
# bad_epochs = np.array([1, 65, 66, 114])-1
############################ Bad epochs for anon4 ###############
# bad_epochs = np.array([31, 41, 81])-1
############################ Bad epochs for anon5 ###############
bad_epochs = np.array([1, 41])-1
################ Align epochs with stimuli ######################
Boredom = np.array([1, 2, 3, 5, 7, 9, 15, 17, 26, 27, 28, 29, 32, 33, 38, 40, 49, 50, 51, 53, 57, 61, 68, 72, 76, 77, 79, 82, 85, 86, 88, 90, 94, 96, 97, 98, 99, 100, 102, 106, 110, 113, 114, 117, 118, 120])-1
Anger = np.array([30, 69, 70, 87])-1
Joy = np.array([4, 12, 14, 18, 19, 20, 21, 23, 25, 35, 37, 39, 42, 43, 46, 48, 54, 59, 66, 67, 74, 80, 93, 105, 111, 116])-1
Admiration = np.array([13, 22, 58, 63, 64, 119])-1
Arousal = np.array([6, 11, 36, 45, 60, 75, 83, 84, 89, 91, 95, 109])-1
Disgust = np.array([31, 41, 52, 71, 73, 81, 101, 112, 115])-1
Sadness = np.array([8, 10, 16, 24, 44, 47, 55, 56, 65, 92, 103, 104])-1
Fear = np.array([34, 62, 78, 107, 108])-1

# Removing bad epochs from the stimuli epochs
for tall in bad_epochs:
    if tall in Boredom:
        Boredom = np.setdiff1d(Boredom, bad_epochs)
    if tall in Anger:
        Anger = np.setdiff1d(Anger, bad_epochs)
    if tall in Joy:
        Joy = np.setdiff1d(Joy, bad_epochs)
    if tall in Admiration:
        Admiration = np.setdiff1d(Admiration, bad_epochs)
    if tall in Arousal:
        Arousal = np.setdiff1d(Arousal, bad_epochs)
    if tall in Disgust:
        Disgust = np.setdiff1d(Disgust, bad_epochs)
    if tall in Sadness:
        Sadness = np.setdiff1d(Sadness, bad_epochs)
    if tall in Fear:
        Fear = np.setdiff1d(Fear, bad_epochs)

Positive = []
Positive.extend(Joy)
Positive.extend(Admiration)
Positive.extend(Arousal)
Negative = []
Negative.extend(Anger)
Negative.extend(Disgust)
Negative.extend(Sadness)
Negative.extend(Fear)
Neutral = []
Neutral.extend(Boredom)
Interest = []
Interest.extend(Positive)
Interest.extend(Negative)

length_diff = len(Positive) - len(Negative)

if length_diff > 0:  # list1 is longer, truncate it
    Positive = Positive[:len(Negative)]
elif length_diff < 0:  # list2 is longer, pad it
    Negative = Negative[:len(Positive)]

boredom_epochs = []
anger_epochs = []
joy_epochs = []
admiration_epochs = []
arousal_epochs = []
disgust_epochs = []
sadness_epochs = []
fear_epochs = []

positive_epochs = []
neutral_epochs = []
negative_epochs = []

interest_epochs = []

for n in Boredom:
    boredom_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Anger:
    anger_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Joy:
    joy_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Admiration:
    admiration_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Arousal:
    arousal_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Disgust:
    disgust_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Sadness:
    sadness_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Fear:
    fear_epochs.append(epochs[n].iloc[:,0].to_numpy())
    
for n in Positive:
    positive_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Neutral:
    neutral_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Negative:
    negative_epochs.append(epochs[n].iloc[:,0].to_numpy())
    
for n in Interest:
    interest_epochs.append(epochs[n].iloc[:,0].to_numpy())

# Extracting features for every epoch in their emotional context:
boredom_features = featest(boredom_epochs)
anger_features = featest(anger_epochs)
joy_features = featest(joy_epochs)
admiration_features = featest(admiration_epochs)
arousal_features = featest(arousal_epochs)
disgust_features = featest(disgust_epochs)
sadness_features = featest(sadness_epochs)
fear_features = featest(fear_epochs)

# Extracting features in positive/'neutral'/negative valence:
positive_features = featest(positive_epochs)
neutral_features = featest(neutral_epochs)
negative_features = featest(negative_epochs)

interest_features = featest(interest_epochs)

################### Storing data in new .npy-file ###########################
# Create a dictionary to store the features for each class

# For all seperate emotions:
features_dict_8 = {
    'Boredom': boredom_features,
    'Anger': anger_features,
    'Joy': joy_features,
    'Admiration': admiration_features,
    'Arousal': arousal_features,
    'Disgust': disgust_features,
    'Sadness': sadness_features,
    'Fear': fear_features
}

# For distingushing between positive/'neutral'/negative emotions
features_dict_3 = {
    'Positive': positive_features,
    'Neutral': neutral_features,
    'Negative': negative_features
}

features_dict_2 = {
    'Positive': positive_features,
    'Negative': negative_features
}



features_dict_i = {
    'Interest': interest_features,
    'Boredom': boredom_features
}

np.save('ssqpca_anon5.npy', features_dict_8)


################# Add to existing data #######################
# existing_data = np.load('features.npy', allow_pickle=True).item()


# Update the existing data with more lists for each class

# existing_data['Boredom'].extend(boredom_features)
# existing_data['Anger'].extend(anger_features)
# existing_data['Joy'].extend(joy_features)
# existing_data['Admiration'].extend(admiration_features)
# existing_data['Arousal'].extend(arousal_features)
# existing_data['Disgust'].extend(disgust_features)
# existing_data['Sadness'].extend(sadness_features)
# existing_data['Fear'].extend(fear_features)

# existing_data['Positive'].extend(positive_features)
# existing_data['Neutral'].extend(neutral_features)
# existing_data['Negative'].extend(negative_features)

# Save the updated data back to the file
# np.save('features.npy', existing_data)

# np.save('3classfeatures.npy', existing_data)
##################################################################

# plt.show()
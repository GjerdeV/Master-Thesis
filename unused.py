########################################
########################################
# THESE ARE TEST CODES OF FAILED ATTEMPTS
########################################

####### This is the original interpolation function that goes directly into script

# for i in range(len(bad_channels)):
#     missing_index = eeg_df.columns.get_loc(bad_channels[i]) # in a list, have to check for all that is missing

# # Find the indices of the neighboring channels, (((((((((((((if next index is nan, then check next))))))))))))))))), if index = 0 or 31, move more away
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
########################################
########################################
########################################





# def divide_signal(signal, epoch_length):
#     """Divide a signal into non-overlapping epochs of fixed length."""
#     num_epochs = signal.shape[0] // epoch_length
#     boundaries = np.arange(num_epochs+1)*epoch_length
#     return boundaries

# def divide_signal(signal, epoch_length):
#     """Divide a signal into non-overlapping epochs of fixed length."""
#     num_epochs = (signal.shape[0] + epoch_length - 1) // epoch_length
#     boundaries = np.arange(num_epochs+1)*epoch_length
#     return boundaries

# def divide_signal(signal, epoch_length):
#     """Divide a signal into non-overlapping epochs of fixed length."""
#     num_epochs = signal.shape[0] // epoch_length
#     boundaries = (np.arange(num_epochs+1)*epoch_length).astype(int)
#     return boundaries

# # Generate a test signal
# t = df['Timestamp']
# signal = average_eeg

# # Divide the signal into 2500-sample epochs
# epoch_length = 2500
# boundaries = divide_signal(signal, epoch_length)

# # Fetch out each epoch separately
# epochs = np.split(signal, boundaries[:-1])

# # Apply FastICA to reject bad epochs
# ica = FastICA(n_components=len(epochs), random_state=0)
# X = np.vstack(epochs).T
# S = ica.fit_transform(X)
# A = ica.mixing_

# # Identify bad epochs using kurtosis threshold
# kurtosis_threshold = 3.0
# kurtosis_values = np.apply_along_axis(lambda x: np.abs(np.kurtosis(x)), axis=0, arr=S)
# bad_epochs = np.where(kurtosis_values > kurtosis_threshold)[0]

# # Remove bad epochs and reconstruct cleaned signal
# clean_epochs = np.delete(epochs, bad_epochs, axis=0)
# clean_signal = np.concatenate(clean_epochs)

# # Visualize the original signal and cleaned signal
# plt.figure()
# fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# axs[0].plot(signal, color='blue')
# for boundary in boundaries:
#     axs[0].axvline(boundary, color='red', linewidth=0.5)
# axs[0].set_title('Original Signal')
# axs[0].set_ylabel('Amplitude')
# axs[1].plot(clean_signal, color='blue')
# clean_boundaries = divide_signal(clean_signal, epoch_length)
# for boundary in clean_boundaries:
#     axs[1].axvline(boundary, color='green', linewidth=0.5)
# axs[1].set_title('Cleaned Signal')
# axs[1].set_xlabel('Sample')
# axs[1].set_ylabel('Amplitude')

######################### ICA ###################################
# bad_epochs = detect_bad_epochs(epochs)

# # Print the indices of the detected bad epochs
# print("Detected Bad Epochs:", bad_epochs)

# # Remove the bad epochs
# clean_epochs = [epochs[i] for i in range(len(epochs)) if i not in bad_epochs]

# # Plotting the clean epochs
# fig, axs = plt.subplots(len(clean_epochs), 1, figsize=(8, 2*len(clean_epochs)))

# for i, epoch in enumerate(clean_epochs):
#     time = np.arange(len(epoch)) / fs
#     axs[i].plot(time, epoch)
#     axs[i].set_ylabel(f'Epoch {i+1}')
#     axs[i].grid(True)

#######################

# reshaped_epochs = np.array(epochs).reshape(len(epochs), -1)

# bad_epochs = detect_bad_epochs(reshaped_epochs)

# # Print the indices of the detected bad epochs
# print("Detected Bad Epochs:", bad_epochs)

# # Remove the bad epochs
# clean_epochs = [epochs[i] for i in range(len(epochs)) if i not in bad_epochs]

# # Plotting the clean epochs
# fig, axs = plt.subplots(len(clean_epochs), 1, figsize=(8, 2*len(clean_epochs)))

# for i, epoch in enumerate(clean_epochs):
#     time = np.arange(len(epoch)) / fs
#     axs[i].plot(time, epoch)
#     axs[i].set_ylabel(f'Epoch {i+1}')
#     axs[i].grid(True)

# plt.xlabel('Time (seconds)')
# # plt.tight_layout()

# cleaned_epochs = detect_bad_epochs(epochs)
# plt.figure()
# plt.plot(cleaned_epochs)

# ica = FastICA(n_components=8, random_state=0, whiten='unit_variance')
# ok = ica.fit_transform(epochs[0])
# ok /= ok.std(axis=0)  # Standardize the estimated signals
# estimated_sources = ok
# estimated_sources.shape

# plt.figure()
# # plt.plot(estimated_sources[0])
# # plt.plot(estimated_sources[1])
# # plt.plot(estimated_sources[2])
# # plt.plot(estimated_sources[3])
# # plt.plot(estimated_sources[4])
# # plt.plot(estimated_sources[5])
# # plt.plot(estimated_sources[6])
# # plt.plot(estimated_sources[7])
# # plt.title('Estimated Source Signals')
# plt.plot(estimated_sources[0])
# plt.tight_layout()
#################################################################

######################### Epoching ##############################
# signal = average_eeg
# epoch_length = 2500 # Samples, 2500 samples = 5000ms = 5s
# boundaries = epoch_signal(signal, epoch_length)

# # Visualize the dividing with vertical lines
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(signal, color='blue')
# for boundary in boundaries:
#     ax.axvline(boundary, color='red', linewidth=0.5)
# ax.set_xlabel('Sample')
# ax.set_ylabel('Amplitude')

# epochs = np.split(signal, boundaries[:-1])

# # Visualize the dividing with vertical lines
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(df['Timestamp'], signal, color='blue')
# for boundary in boundaries:
#     ax.axvline(boundary, color='red', linewidth=0.5)
# ax.set_xlabel('Sample')
# ax.set_ylabel('Amplitude')

# # Epochs viz
# num_epochs_per_group = 4
# num_groups = len(epochs) // num_epochs_per_group
# if len(epochs) % num_epochs_per_group != 0:
#     num_groups += 1

# for i in range(num_groups):
#     fig, axs = plt.subplots(num_epochs_per_group, 1, figsize=(10, 6), sharex=True)
#     for j in range(num_epochs_per_group):
#         epoch_index = i*num_epochs_per_group + j
#         if epoch_index >= len(epochs):
#             break
#         epoch = epochs[epoch_index]
#         axs[j].plot(epoch, color='blue')
#         axs[j].set_ylabel('Amplitude')
#         axs[j].axhline(0, color='gray', linewidth=0.5)
#     plt.xlabel('Sample')


# epoch_size = 2500

# # calculate the number of epochs per signal
# num_epochs = filtered_eeg_df.shape[1] // epoch_size

# # initialize an empty list to store the epoch dataframes
# epoch_dfs = []

# # loop through each signal
# for i in range(df.shape[0]):
#     # loop through each epoch
#     for j in range(num_epochs):
#         # calculate the start and end indices of the epoch
#         start_idx = j * epoch_size
#         end_idx = (j + 1) * epoch_size
        
#         # slice the signal dataframe to get the epoch dataframe
#         epoch_df = filtered_eeg_df.iloc[i, start_idx:end_idx]
        
#         # add the epoch dataframe to the list of epoch dataframes
#         epoch_dfs.append(epoch_df)
        
# # concatenate all epoch dataframes into a single dataframe
# epoch_df = pd.DataFrame(epoch_dfs)

# # transpose the dataframe so that each row represents an epoch
# epoch_df = epoch_df.T

# # reset the index to start from 0
# epoch_df = epoch_df.reset_index(drop=True)

# fig11 = plt.figure(11)
# fig11.suptitle('Epochs')

# plt.subplot(211)
# plt.plot(df['Timestamp'], epoch_df)
# plt.title('Filtered EEG data')
# plt.xlabel('Time [ms]')
# plt.ylabel('uV')

# plt.subplot(212)
# for i in range(32):
#     f, Pxx = signal.welch(epoch_df.iloc[:,i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
#     mask = (f >= fmin) & (f <= fmax)
#     f = f[mask]
#     Pxx = Pxx[mask]
#     plt.semilogy(f, Pxx)
# plt.title('Filtered EEG data in frequency domain')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [dB/Hz]')
#################################################################
# Wavelet transform
# Feature extraction



# Wavelet transform #
# wavelet = 'db8'  # Wavelet type
# levels = 4  # Number of decomposition levels

# # Compute the wavelet coefficients
# coeffs = pywt.wavedec(filtered_eeg_df['Ch1'], wavelet, level=levels)

# # Reconstruct the signal from the wavelet coefficients
# reconstructed = pywt.waverec(coeffs, wavelet)
# plt.figure(8)
# plt.plot(reconstructed)

# plt.figure(9)
# for i in range(len(coeffs)):
#     plt.subplot(levels+1, 1, i+1)
#     plt.plot(coeffs[i])
#     plt.title('Level {}'.format(i))
# plt.tight_layout()



###################### Plotting of all raw data #################
# Small snippet of signal to see effect of filtering easier
# start = 0
# stop = 1000
# n_channels = raw_eeg_df.shape[1]
# n_samples = raw_eeg_df.shape[0]

# Define the figure size and the spacing between subplots
# figsize = (12, 8)
# hspace = 0.4

# Create a new figure with subplots for each channel
# fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True, sharey=True, gridspec_kw={'hspace': hspace})

# Loop over each channel and plot it in the corresponding subplot
# for i, ax in enumerate(axes):
#     plt.figure(1)
#     ax.plot(raw_eeg_df.iloc[:, i])
#     ax.set_ylabel('{}'.format(i+1))
#     ax.set_xlim([0, n_samples])
# ########################################################################

# ############################################## Bad channel visual detection in PSD #########################################
# # Looking further into the PSD of all the channels to observe if there's any bad channels
# # Define the frequency range of interest (in Hz)
# fmin, fmax = 0, 100

# # Define the window function to use for the PSD calculation
# window = 'hann'

# # Loop over each channel in the EEG signal and calculate its PSD
# plt.figure(2)

# # Loop over each channel in the EEG signal and calculate its PSD
# for i in range(32):
#     f, Pxx = signal.welch(hpf_eeg_df.iloc[:, i], fs, window=window, nperseg=fs*2, noverlap=fs, scaling='density')
    
#     # Restrict the PSD to the frequency range of interest
#     mask = (f >= fmin) & (f <= fmax)
#     f = f[mask]
#     Pxx = Pxx[mask]
    
#     # Plot the PSD
#     plt.plot(f, 10 * np.log10(Pxx), label='Channel {}'.format(i+1))

# # Add axis labels and a title
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (dB/Hz)')
# plt.title('Power Spectral Density')

# # Add a legend
# plt.legend()
# #############################################################################################################################

# ############### Plotting of all data throughout the filtering process step-by-step ##################
# plt.figure(3)
# plt.subplot(411)
# plt.plot(raw_eeg_df, 'b-')
# plt.subplot(412)
# plt.plot(hpf_eeg_df, 'g-')
# plt.subplot(413)
# plt.plot(notchf_eeg_df, 'r-')
# plt.subplot(414)
# plt.plot(denoised_df, 'k-')
# plt.title('All data through filtering')
# ######################################################################################################

# ############### Plotting of a short sequences of one channel throughout the filtering process step-by-step ##################
# plt.figure(4)
# plt.plot(raw_eeg_df['Ch1'].iloc[start:stop], 'b-')
# plt.plot(hpf_eeg_df['Ch1'].iloc[start:stop], 'g-')
# plt.plot(notchf_eeg_df['Ch1'].iloc[start:stop], 'r-')
# plt.plot(denoised_df['Ch1'].iloc[start:stop], 'k-')
# plt.title('One channel snippet through filtering')

# plt.figure(5)
# plt.subplot(411)
# plt.plot(raw_eeg_df['Ch1'].iloc[start:stop], 'b-')
# plt.subplot(412)
# plt.plot(hpf_eeg_df['Ch1'].iloc[start:stop], 'g-')
# plt.subplot(413)
# plt.plot(notchf_eeg_df['Ch1'].iloc[start:stop], 'r-')
# plt.subplot(414)
# plt.plot(denoised_df['Ch1'].iloc[start:stop], 'k-')
# plt.title('One channel snippet through filtering')
# #############################################################################################################################

# ############################### PSD of one channel throughout the fitlering process #########################################
# [f1, Pxx_den1] = signal.periodogram(raw_eeg_df['Ch1'], fs)
# [f2, Pxx_den2] = signal.periodogram(hpf_eeg_df['Ch1'], fs)
# [f3, Pxx_den3] = signal.periodogram(notchf_eeg_df['Ch1'], fs)
# [f4, Pxx_den4] = signal.periodogram(denoised_df['Ch1'], fs)

# plt.figure(6)
# plt.subplot(411)
# plt.semilogy(f1, Pxx_den1, 'b-')
# plt.subplot(412)
# plt.semilogy(f2, Pxx_den2, 'g-')
# plt.subplot(413)
# plt.semilogy(f3, Pxx_den3, 'r-')
# plt.subplot(414)
# plt.semilogy(f4, Pxx_den4, 'k-')
# #############################################################################################################################

# ############################# Zoom in on bad channel ############################
# # Showing "bad" channel
# plt.figure(7)
# plt.subplot(211)
# plt.plot(denoised_df, 'k-')
# plt.subplot(212)
# plt.plot(denoised_df[161700:162100], 'k-')
##################################################################################

# plt.figure(8)
# plt.plot(denoised_df, 'b-')
# plt.plot(raw_eeg_df['Ch31'].iloc[161700:162100], 'b-')
# plt.plot(denoised_df['Ch23'], 'g-')
# plt.plot(denoised_df['Ch22'], 'k-')
# plt.plot(raw_eeg_df['Ch2'].iloc[161700:162100], 'r-')








# Epoching
###### Current ##########
# import numpy as np
# def epoch_signal(signal, epoch_length):
#     epoch_length = epoch_length/2
#     num_epochs = signal.shape[0] // epoch_length
#     boundaries = np.arange(num_epochs+1)*epoch_length*2
#     boundaries = boundaries/1000
#     return boundaries
##########################



# for i in range(num_figures):
#     start_epoch = i * num_epochs_per_figure
#     end_epoch = min((i + 1) * num_epochs_per_figure, total_epochs)
    
#     num_epochs_in_figure = end_epoch - start_epoch
#     fig, axs = plt.subplots(num_epochs_in_figure, 1, figsize=(8, 2*num_epochs_in_figure))
    
#     for j in range(num_epochs_in_figure):
#         epoch_index = start_epoch + j
#         epoch = epochs[epoch_index]
#         time = np.arange(len(epoch)) / fs
#         axs[j].plot(time, epoch)
#         axs[j].set_ylabel(f'Epoch {epoch_index+1}')
#         axs[j].grid(True)
    
#     plt.xlabel('Time (seconds)')
#     plt.tight_layout()

################ Current ###############
# epoch_length = 5 # in seconds
# epoch_length = epoch_length*1000

# signal = average_eeg
# boundaries = epoch_signal(signal, epoch_length)

# # Visualize the dividing with vertical lines
# fig, ax = plt.subplots(figsize=(10, 6))
# # ax.plot(df['Timestamp'], signal, color='blue')
# ax.plot(df['Timestamp'], signal, color='blue')
# for boundary in boundaries:
#     ax.axvline(boundary, color='red', linewidth=0.5)
# ax.set_xlabel('Time[s]')
# ax.set_ylabel('uV')

# sample_boundaries = boundaries*500
########## Current ################

# plt.figure()
# segments = []
# for i in range(len(boundaries) + 1):
#     if i == 0:
#         segment = signal[:boundaries[i]]
#     elif i == len(boundaries):
#         segment = signal[boundaries[i-1]:]
#     else:
#         segment = signal[boundaries[i-1]:boundaries[i]]
#     segments.append(segment)

# fig, axs = plt.subplots(len(segments), 1, figsize=(6, 2*len(segments)))

# for i, segment in enumerate(segments):
#     axs[i].plot(segment)
#     axs[i].set_ylabel(f'Epoch {i+1}')
#     axs[i].set_xticks(np.arange(len(segment)))
#     axs[i].set_xticklabels(np.arange(len(segment))+1)
#     axs[i].grid(True)

# plt.xlabel('Time')
# plt.tight_layout()


############### Epoching ##################
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import FastICA

# def divide_signal(signal, epoch_length):
#     """Divide a signal into non-overlapping epochs of fixed length."""
#     num_epochs = signal.shape[0] // epoch_length
#     boundaries = np.arange(num_epochs+1)*epoch_length
#     return boundaries

# # Generate a test signal
# t = np.linspace(0, 10*np.pi, 100000)
# signal = np.sin(t)

# # Divide the signal into 2500-sample epochs
# epoch_length = 2500
# boundaries = divide_signal(signal, epoch_length)

# # Fetch out each epoch separately
# epochs = np.split(signal, boundaries[:-1])

# # Apply FastICA to reject bad epochs
# ica = FastICA(n_components=len(epochs), random_state=0)
# X = np.vstack(epochs).T
# S = ica.fit_transform(X)
# A = ica.mixing_

# # Identify bad epochs using kurtosis threshold
# kurtosis_threshold = 3.0
# kurtosis_values = np.apply_along_axis(lambda x: np.abs(np.kurtosis(x)), axis=0, arr=S)
# bad_epochs = np.where(kurtosis_values > kurtosis_threshold)[0]

# # Remove bad epochs and reconstruct cleaned signal
# clean_epochs = np.delete(epochs, bad_epochs, axis=0)
# clean_signal = np.concatenate(clean_epochs)

# # Visualize the original signal and cleaned signal
# fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# axs[0].plot(signal, color='blue')
# for boundary in boundaries:
#     axs[0].axvline(boundary, color='red', linewidth=0.5)
# axs[0].set_title('Original Signal')
# axs[0].set_ylabel('Amplitude')
# axs[1].plot(clean_signal, color='blue')
# clean_boundaries = divide_signal(clean_signal, epoch_length)
# for boundary in clean_boundaries:
#     axs[1].axvline(boundary, color='green', linewidth=0.5)
# axs[1].set_title('Cleaned Signal')
# axs[1].set_xlabel('Sample')
# axs[1].set_ylabel('Amplitude')
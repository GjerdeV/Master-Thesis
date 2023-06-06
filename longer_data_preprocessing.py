import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from baselining import baselining
from detect_bad_chnl import detect_bad_channels
from detect_bad_freqz import detect_bad_channels_freqz
from interpolation import interpolate
from epoching import epoch_signal
from feature_extraction import featest

# Defining sample frequency (one sample every 2ms)
fs = 1/0.002
# fmin and fmax are used if just certain frequencies are of interest in the PSD plots
fmin = 0
fmax = 50
# Defining window function as hann
window = 'hann'

# Reading the .csv-file and specifying which columns and rows to use to fetch correct data

# Anon 1
# df = pd.read_csv('eeg_data/001_anon_1.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,340), *range(2257553,2257556)])
# 300 000 samples:
# df = pd.read_csv('eeg_data/001_anon_1.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,340), *range(300340,2257556)])
# All 2 250 000 samples:
df = pd.read_csv('eeg_data/001_anon_1.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,340), *range(2250340,2257556)])

# Anon 2, range not set yet
# df = pd.read_csv('eeg_data/001_anon_2.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(301015,2257264)])
# First 300 000 samples:
# df = pd.read_csv('eeg_data/001_anon_2.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(300050,2257264)])
# All 2 250 000 samples:
# df = pd.read_csv('eeg_data/001_anon_2.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(2250050,2257264)])

# Anon 3
# df = pd.read_csv('eeg_data/001_anon_3.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,45), *range(301015,301018)])
# df = pd.read_csv('eeg_data/001_anon_3.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,45), *range(300045,301018)])

# Anon 4 - plotting doesn't start at 0????
# df = pd.read_csv('eeg_data/001_anon_4.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(301023,301026)])
# df = pd.read_csv('eeg_data/001_anon_4.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,50), *range(300050,301026)])

# Anon 5 - bad channels freqz 0.5 atm, 1.5 vanlig, plotting doesn't start at 0 here either
# df = pd.read_csv('eeg_data/001_anon_5.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,44), *range(301022,301026)])
# df = pd.read_csv('eeg_data/001_anon_5.csv', sep=',', usecols=[1, *range(10,42)], skiprows=[*range(0,26), *range(27,44), *range(300044,301026)])

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

# plt.figure(1)

# plt.subplot(311)
# plt.plot(df['Timestamp'], raw_eeg_df.iloc[:, 13]) # Plotting of raw data
# plt.grid(True)
# plt.title('Raw EEG data')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')
# plt.show()
############################################################
################## Baseline correction #####################

bsl_eeg = baselining(df)

##############################################################
################ Detection of bad channels ###################
bad_channels = []

# Anon 1, Threshold = 2.0
# Anon 2, Threshold = 1.5
# Anon 3, Threshold = 4.0
# Anon 4, Threshold = 2.0
# Anon 5, Threshold = 1.5
bad_channels.extend(detect_bad_channels(bsl_eeg, threshold=2.0))

###### Automatic detection of bad channels based on PSD ######
# Anon 1, Threshold = 2.0
# Anon 2, Threshold = 0.5
# Anon 3, Threshold = 4.0
# Anon 4, Threshold = 2.0
# Anon 5, Threshold = 0.5
bad_channels.extend(detect_bad_channels_freqz(bsl_eeg, fs, window, threshold=2.0))

# For anon 2, manual bad channel detection was necessary due to unknown reasons, Channel 13 is bad:
# bad_channels.append('Ch13')

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
# plt.legend()
# plt.show()
##############################################################
################### Notch filter, Hz = 50Hz ##################
# Anon 2's data fail on this point for unknown reasons

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
#################################################################
################### Bandpass filtering ##########################

f_low = 4  # in Hz
f_high = 45  # in Hz
order = 4

b, a = signal.butter(order, [f_low/(fs/2), f_high/(fs/2)], btype='bandpass')

filtered_eeg_df = notchf_eeg_df.copy()
for chnl in range(1, 33):
    chnl_name = 'Ch' + str(chnl)
    filtered_eeg_df[chnl_name] = signal.filtfilt(b, a, notchf_eeg_df[chnl_name])

#################################################################
######################## Average signal #########################

average_eeg = np.mean(filtered_eeg_df, axis=1)
avg_df = pd.DataFrame({"Average Signal": average_eeg})

#################################################################
######################### Epoching ##############################

epoch_duration = 5  # seconds
epochs = epoch_signal(avg_df, fs, epoch_duration)

num_epochs_per_figure = 5
total_epochs = len(epochs)
num_figures = (total_epochs + num_epochs_per_figure - 1) // num_epochs_per_figure

fig, ax = plt.subplots(figsize=(12, 4))

# Plotting the averaged signal
time = np.arange(len(avg_df)) / fs
ax.plot(time, avg_df, label='Group Averaged Signal')

# Plotting red vertical lines for epoch boundaries
# for i in range(len(epochs) - 1):
#     epoch_end_time = (i + 1) * epoch_duration
#     ax.axvline(x=epoch_end_time, color='red', linestyle='-', alpha=0.8)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('uV')
# ax.set_title('Group Averaged Signal with Epoch Boundaries')
# ax.legend()

# t = np.arange(0, 5, 0.002, dtype='float64') # Defining the time-axis for epochs
# plt.figure()
# plt.plot(t, epochs[600].iloc[:,0].to_numpy())
# plt.grid(True)
# plt.title('Single epoch data')
# plt.xlabel('Time [s]')
# plt.ylabel('uV')
# plt.show()
##################################################################
################# Wavelet Transformation #########################
t = np.arange(0, 5, 0.002, dtype='float64') # Defining the time-axis for epochs
#################### Bad epochs for anon1 #######################
bad_epochs = np.array([668, 683, 827, 873])-1
##################### Bad epochs for anon2 ######################
#################################################################
####################### Bad epochs for anon3 ####################
# bad_epochs = np.array([1, 65, 66, 114])-1
############################ Bad epochs for anon4 ###############
# bad_epochs = np.array([31, 41, 81])-1
############################ Bad epochs for anon5 ###############
# bad_epochs = np.array([1, 41])-1
################ Align epochs with stimuli ######################
Boredom = np.array([1, 2, 3, 6, 9, 14, 15, 16, 28, 29, 
                       42, 43, 46, 49, 50, 51, 52, 53, 54, 55, 58, 69, 72, 73, 87, 88, 90, 92, 95, 97, 
                       98, 105, 106, 107, 109, 112, 113, 114, 115, 116, 123, 124, 125, 126, 127, 128, 129, 144, 
                       152, 153, 154, 155, 156, 161, 162, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 192, 193, 
                       194, 195, 196, 197, 198, 199, 212, 216, 220, 221, 222, 223, 224, 225, 229, 230, 232, 233, 235, 236, 
                       238, 239, 240, 242, 247, 249, 250, 285, 286, 289, 298, 299, 300, 301, 311, 312, 313, 314, 315, 316, 
                       317, 321, 323, 327, 329, 330, 331, 332, 333, 353, 354, 356, 357, 360, 361, 365, 367, 379, 381, 384, 385, 
                       386, 387, 391, 392, 393, 399, 400, 401, 402, 403, 404, 405, 410, 423, 424, 426, 427, 428, 429, 430, 431, 432, 
                       433, 435, 436, 441, 442, 446, 447, 448, 449, 450, 460, 471, 473, 474, 475, 486, 487, 489, 499, 506, 507, 514, 515, 
                       517, 519, 520, 521, 522, 523, 524, 525, 526, 536, 569, 586, 587, 588, 589, 590, 593, 594, 595, 596, 597, 598, 599, 600, 
                       601, 602, 603, 604, 605, 606, 608, 609, 612, 613, 614, 618, 619, 623, 627, 628, 629, 630, 632, 633, 634, 635, 636, 642, 
                       646, 647, 648, 650, 651, 653, 654, 655, 656, 657, 658, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 677, 
                       678, 679, 680, 681, 682, 685, 686, 692, 705, 706, 707, 716, 718, 719, 720, 721, 722, 723, 733, 744, 745, 746, 747, 748, 
                       749, 750, 751, 752, 754, 758, 760, 761, 762, 763, 764, 765, 766, 773, 774, 775, 776, 777, 778, 779, 780, 783, 786, 787, 
                       788, 792, 795, 796, 798, 800, 801, 802, 803, 806, 807, 821, 823, 824, 825, 836, 839, 840, 841, 842, 843, 844, 845, 846, 
                       854, 856, 859, 860, 861, 862, 863, 872, 873, 880, 881, 882, 884, 885, 887, 890, 891, 892, 893, 894, 895, 897, 898, 899])-1

Admiration = np.array([25, 30, 31, 59, 60, 61, 62, 63, 66, 74, 75, 77, 96, 99, 100, 145, 146, 147, 171, 217, 218, 226, 237, 241, 
                       318, 325, 334, 335, 336, 337, 338, 339, 340, 344, 345, 346, 347, 348, 349, 350, 351, 352, 371, 372, 373, 
                       374, 375, 376, 377, 378, 455, 456, 457, 458, 459, 461, 463, 464, 465, 466, 467, 468, 469, 470, 476, 477, 
                       478, 479, 480, 504, 516, 615, 617, 637, 638, 660, 661, 662, 663, 755, 756, 757, 781, 782, 797, 799, 804, 
                       805, 808, 809, 810, 811, 812, 813, 814, 815, 826, 827, 829, 830, 831, 832, 833, 834, 835, 837, 857, 858, 886])-1

Joy = np.array([4, 5, 7, 8, 10, 11, 18, 32, 33, 34, 35, 36, 37, 39, 40, 44, 45, 47, 48, 64, 65, 67, 68, 76, 78, 79, 80, 81, 103, 104, 
                111, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 157, 158, 159, 160, 165, 170, 183, 184, 186, 187, 
                188, 190, 191, 200, 201, 202, 203, 204, 205, 206, 207, 219, 251, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 263, 
                264, 266, 267, 268, 269, 270, 271, 277, 278, 279, 280, 288, 290, 291, 292, 293, 294, 295, 296, 297, 308, 355, 358, 359, 
                388, 394, 396, 397, 398, 434, 462, 472, 481, 483, 484, 488, 500, 502, 505, 508, 509, 510, 511, 512, 513, 592, 611, 616, 
                620, 621, 624, 625, 626, 652, 676, 683, 684, 704, 708, 709, 710, 717, 724, 727, 731, 732, 735, 736, 743, 822, 838, 874, 
                875, 876, 877, 878, 879, 883, 888, 900])-1

Disgust = np.array([22, 23, 24, 26, 27, 85, 166, 167, 168, 169, 228, 287, 309, 310, 437, 438, 439, 440, 485, 631, 725, 738, 740, 753, 
                    794, 816, 855])-1

Arousal = np.array([56, 57, 151, 369, 527, 528, 529, 530, 531, 532, 533, 534, 535, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 
                    547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 570, 
                    571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 622, 896])-1

Sadness = np.array([12, 13, 20, 21, 38, 41, 84, 86, 89, 91, 110, 146, 148, 149, 150, 185, 189, 209, 210, 211, 213, 214, 215, 231, 
                    234, 259, 265, 272, 273, 274, 275, 276, 281, 363, 364, 366, 368, 370, 389, 390, 395, 406, 407, 408, 409, 425, 
                    482, 490, 491, 492, 493, 494, 495, 496, 497, 498, 501, 503, 649, 688, 689, 693, 694, 695, 696, 697, 698, 699, 700, 
                    701, 702, 703])-1

Fear = np.array([17, 19, 70, 71, 82, 83, 93, 94, 101, 102, 108, 117, 118, 119, 120, 121, 122, 163, 164, 176, 208, 243, 244, 245, 246, 
                 248, 282, 283, 284, 302, 303, 305, 306, 307, 319, 320, 322, 324, 326, 328, 341, 342, 343, 362, 411, 412, 419, 420, 421, 
                 422, 443, 444, 445, 453, 454, 591, 607, 610, 639, 640, 641, 643, 644, 659, 687, 690, 691, 711, 712, 713, 714, 715, 726, 
                 728, 729, 730, 734, 739, 741, 742, 759, 767, 768, 769, 770, 771, 772, 784, 785, 791, 793, 817, 818, 819, 820, 828, 847, 
                 848, 849, 850, 851, 852, 853, 865, 866, 871, 889])-1

Anger = np.array([227, 304, 380, 382, 383, 413, 414, 415, 416, 417, 418, 451, 452, 518, 645, 737, 789, 790, 864, 867, 869, 870])-1

# Removing bad epochs from the stimuli response epochs
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

positive_epochs = []
neutral_epochs = []
negative_epochs = []

for n in Positive:
    positive_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Neutral:
    neutral_epochs.append(epochs[n].iloc[:,0].to_numpy())
for n in Negative:
    negative_epochs.append(epochs[n].iloc[:,0].to_numpy())

# Extracting features for every epoch in their emotional context using the wavelet transformation
boredom_features = featest(boredom_epochs)
anger_features = featest(anger_epochs)
joy_features = featest(joy_epochs)
admiration_features = featest(admiration_epochs)
arousal_features = featest(arousal_epochs)
disgust_features = featest(disgust_epochs)
sadness_features = featest(sadness_epochs)
fear_features = featest(fear_epochs)

positive_features = featest(positive_epochs)
neutral_features = featest(neutral_epochs)
negative_features = featest(negative_epochs)
################### Storing data in new .npy-file ###########################
# Create a dictionary to store the features for each class
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

features_dict_3 = {
    'Positive': positive_features,
    'Neutral': neutral_features,
    'Negative': negative_features
}

features_dict_2 = {
    'Positive': positive_features,
    'Negative': negative_features
}

# Save the features to a file
np.save('l.npy', features_dict_8)


################# Add to existing data #######################
# existing_data = np.load('3classfeatures.npy', allow_pickle=True).item()

# # Update the existing data with more lists for each class
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

# # Save the updated data back to the file
# np.save('3classfeatures.npy', existing_data)
##################################################################

# plt.show()
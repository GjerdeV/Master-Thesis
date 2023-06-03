import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import bspline
import pandas as pd

def interpolate(eeg_df, bad_channels=[]):
    # good_channels = [col for col in eeg_df.columns if col not in bad_channels]
    # for channel in bad_channels:
    # # Get two random signals to interpolate from the dataframe
    #     signal1 = eeg_df[good_channels].sample(1)[np.random.choice(good_channels)].values
    #     signal2 = eeg_df[good_channels].sample(1)[np.random.choice(good_channels)].values
    #     while signal1 == signal2:
    #         signal2 = eeg_df[good_channels].sample(1)[np.random.choice(good_channels)].values
    
        
    #     # Check if either signal is null, and skip interpolation if so
    #     # if pd.isnull(signal1).any() or pd.isnull(signal2).any():
    #     #     continue
        
    #     # Generate a linearly spaced x-axis for the new signal
    #     # x_new = np.linspace(0, 1, len(signal1))
    #     x_old = np.linspace(0, 1, len(signal1))
    #     x_new = np.linspace(0, 1, len(signal1) * 2)
        
    #     # Interpolate the two signals to create a new signal for the bad channel
    #     f1 = interp1d(x_old, signal1)
    #     f2 = interp1d(x_old, signal2)
    #     new_signal = (f1(x_new) + f2(x_new)) / 2.0
        
    #     # Replace the bad channel with the new interpolated signal
    #     eeg_df[channel] = new_signal[:len(signal1)]
    # return eeg_df
    
    # good_channels = [col for col in eeg_df.columns if col not in bad_channels]
    # print(good_channels)
    
    for i in range(len(bad_channels)):
        missing_index = eeg_df.columns.get_loc(bad_channels[i])
        
    #     for j in range(len(good_channels)):
    #         left_index = eeg_df.columns.get_loc(good_channels[missing_index])
    #         if j == max(range(len(good_channels))):
    #             right_index = eeg_df.columns.get_loc(good_channels[missing_index])
    #         else:
    #             right_index = eeg_df.columns.get_loc(good_channels[missing_index+1])
        
        # random_int_chnl = np.random.choice(good_channels)
        # left_index = eeg_df.columns.get_loc(random_int_chnl)
        
        # if left_index == max(good_channels):
        #     right_index = eeg_df.columns.get_loc(good_channels[random_int_chnl-1])
            
            
        # print(left_index)
        # print(right_index)
        # if missing_index == 0:
        #     left_index = missing_index
        #     right_index = missing_index + 1
        # elif missing_index == 31:
        #     left_index = missing_index - 1
        #     right_index = missing_index
        # else:
        #     left_index = missing_index - 1
        #     right_index = missing_index + 1
        
        # while left_index >= 1 or right_index <= 30:
        #     if left_index not in bad_channels and not np.isnan(eeg_df.iloc[:, left_index]).all():
        #         left_index = left_index
        #     elif right_index not in bad_channels and not np.isnan(eeg_df.iloc[:, right_index]).all():
        #         right_index = right_index
        #     else:
        #         left_index -= 1
        #         right_index += 1
        
        left_index = missing_index - 1
        right_index = missing_index + 1

    # Get the values of the neighboring channels
        left_channel = eeg_df.iloc[:, left_index].values
        right_channel = eeg_df.iloc[:, right_index].values

        # Interpolate the missing channel
        # interp_func = interp1d([left_index, right_index], [left_channel, right_channel], axis=0, fill_value="extrapolate")
        interp_func = interp1d([left_index, right_index], [left_channel,right_channel], axis=0)
        # interp_func = bspline([left_index, right_index], [left_channel,right_channel], axis=0)
        missing_channel_values = interp_func(missing_index)

        # Replace the missing channel in the DataFrame
        eeg_df.iloc[:, missing_index] = missing_channel_values
    
    return eeg_df
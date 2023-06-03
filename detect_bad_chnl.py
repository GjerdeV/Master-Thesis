import numpy as np

def detect_bad_channels(dataframe, threshold):

    # Compute the standard deviation of each channel
    channel_std = np.std(dataframe.values, axis=0)

    # Compute the interquartile range of the standard deviation
    q1, q3 = np.percentile(channel_std, [25, 75])
    iqr = q3 - q1

    # Define the threshold for bad channels
    threshold_std = q3 + threshold * iqr
    threshold_std2 = q1 - threshold * iqr

    # Find the bad channels
    bad_channels = []
    for channel, std_value in zip(dataframe.columns, channel_std):
        if std_value > threshold_std or threshold_std2 > std_value:
            bad_channels.append(channel)

    return bad_channels
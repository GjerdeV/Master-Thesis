import numpy as np
from sklearn.decomposition import FastICA
from scipy import stats

def detect_bad_epochs(epochs, threshold=2.5):
    # # Perform FastICA on the epochs
    # ica = FastICA(n_components=len(epochs), random_state=0)
    # transformed_epochs = ica.fit_transform(epochs)
    
    # # Calculate the kurtosis of the transformed epochs
    # kurtosis = np.abs(np.kurtosis(transformed_epochs, axis=0))
    
    # # Set a threshold for bad epoch detection
    # # threshold = 2.5
    
    # # Identify bad epochs based on kurtosis
    # bad_epochs = np.where(kurtosis > threshold)[0]
    
    # return bad_epochs
    
    # ica = FastICA(n_components=epochs.shape[1], random_state=0)
    # transformed_epochs = ica.fit_transform(epochs)
    
    # # Calculate the kurtosis of the transformed epochs
    # kurtosis = np.abs(stats.kurtosis(transformed_epochs, axis=0))
    
    # # Set a threshold for bad epoch detection
    # # threshold = 2.5
    
    # # Identify bad epochs based on kurtosis
    # bad_epochs = np.where(kurtosis > threshold)[0]
    
    # return bad_epochs

    num_epochs = epochs.shape()


    # Reshape the epochs array into a 2D matrix
    reshaped_epochs = np.reshape(epochs, (num_epochs, -1))

    # Perform FastICA on the reshaped epochs
    ica = FastICA()
    components = ica.fit_transform(reshaped_epochs)

    # Calculate the kurtosis of each component
    kurtosis = np.mean(np.abs(components) ** 4, axis=0) - 3

    # Find the indices of bad epochs based on kurtosis
    bad_epoch_indices = np.where(kurtosis > threshold)[0]  # Replace 'threshold' with your desired value

    # Remove the bad epochs from the array
    cleaned_epochs = np.delete(epochs, bad_epoch_indices, axis=0)

    return cleaned_epochs
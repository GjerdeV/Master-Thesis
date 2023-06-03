from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import numpy as np

def run_ica(eeg_data, n_components):
    # Initialize the FastICA object
    ica = FastICA(n_components=n_components, random_state=42)
    
    # Fit the ICA model to the data
    ica.fit(eeg_data.T)
    
    # Get the ICA components
    ica_components = ica.transform(eeg_data.T)
    
    # Identify the eye-related components by finding the ones that have high kurtosis
    kurt = np.abs(np.apply_along_axis(kurtosis, 0, ica_components))
    eye_components = np.argmax(kurt, axis=0)
    
    # Remove the eye-related components
    eeg_data_ica = ica.inverse_transform(np.delete(ica_components, eye_components, axis=1))
    
    return eeg_data_ica.T
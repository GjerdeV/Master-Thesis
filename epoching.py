def epoch_signal(signal, sampling_rate, epoch_duration):
    # Calculate the number of samples in each epoch
    epoch_samples = int(sampling_rate * epoch_duration)
    
    # Calculate the total number of epochs
    total_epochs = len(signal) // epoch_samples
    
    # Create an empty list to store the epochs
    epochs = []
    
    # Extract epochs from the signal
    for i in range(total_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        epoch = signal[start:end]
        epochs.append(epoch)
    
    return epochs
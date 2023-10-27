import numpy as np
import pandas as pd
import mne
import os

def load_data(data):
    df = pd.DataFrame(data)
    eeg_data = df.iloc[:, 0:8].to_numpy().T
    return eeg_data

def amplify_data(eeg_data, amplification_factor=10):
    amplified_data = eeg_data * amplification_factor
    return amplified_data.astype(np.float64)  # Ensure data type is float64

def notch_filter_data(amplified_data, Fs=250, freqs=[50], notch_widths=1):
    return mne.filter.notch_filter(amplified_data, Fs=Fs, freqs=freqs, notch_widths=notch_widths)

def bandpass_filter_data(notched_data, sfreq=250, l_freq=0.1, h_freq=40):
    return mne.filter.filter_data(notched_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq)

def remove_artifacts(bandpassed_data):
    ch_names = ['ch{}'.format(i) for i in range(1, 9)]
    ch_types = ['eeg'] * 8
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=250.0)
    raw = mne.io.RawArray(bandpassed_data, info)
    
    # Apply ICA for artifact removal
    ica = mne.preprocessing.ICA(n_components=5, random_state=97, max_iter=800)
    ica.fit(raw)
    
    # Assuming the first two ICs represent artifacts. This might need adjustments based on your dataset.
    ica.exclude = [0, 1]
    
    # Return data after removing artifacts
    return ica.apply(raw).get_data()

def preprocess_data(input_path):
    eeg_data = load_data(input_path)
    amplified_data = amplify_data(eeg_data)
    notched_data = notch_filter_data(amplified_data)
    bandpassed_data = bandpass_filter_data(notched_data)
    cleaned_data = remove_artifacts(bandpassed_data)
    return np.array(cleaned_data)
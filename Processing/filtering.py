import numpy as np
import pandas as pd
import mne
import os

def process_file(input_path, output_path):
    # Load the data
    data = np.load(input_path)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Extract EEG channels
    eeg_data = df.iloc[:, 0:8].to_numpy().T

    # Create MNE info structure
    ch_names = ['ch{}'.format(i) for i in range(1, 9)]
    ch_types = ['eeg'] * 8
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=250.0)

    # 1. Amplification
    amplification_factor = 10
    amplified_data = eeg_data * amplification_factor
    amplified_data = amplified_data.astype(np.float64)  # Ensure data is float64

    # 2. Notch Filter
    notched_data = mne.filter.notch_filter(amplified_data, Fs=250, freqs=[50], notch_widths=1)

    # 3. Bandpass Filter
    low_freq, high_freq = 0.1, 40
    bandpassed_data = mne.filter.filter_data(notched_data, sfreq=250, l_freq=low_freq, h_freq=high_freq)

    # Create an MNE raw object
    filtered_raw = mne.io.RawArray(bandpassed_data, info)

    # 4. Artifact Removal using ICA
    ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=800)
    ica.fit(filtered_raw)

    # Apply the ICA to the raw data to remove the components (this step was missing in the initial code)
    cleaned_data = ica.apply(filtered_raw, exclude=ica.exclude).get_data()
    

    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
    np.save(output_path, cleaned_data)

base_dir = 'C:/Users/Janvg/Documents/Hackaton/ezyZip/'
folders = ['positive', 'neutral', 'negative']

for folder in folders:
    input_folder_path = os.path.join(base_dir, folder)
    output_folder_path = os.path.join(base_dir, f"{folder}_processed")
    
    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)
    
    for file in os.listdir(input_folder_path):
        if file.endswith('.npy'):
            input_file_path = os.path.join(input_folder_path, file)
            output_file_path = os.path.join(output_folder_path, file)
            
            process_file(input_file_path, output_file_path)

import os
import numpy as np
import pandas as pd
import mne

def time_masking(raw_data, segment_length=500):
    """Apply time masking by setting random sections of the EEG signal to zero."""
    if raw_data._data.shape[1] < segment_length:
        # Skip masking or mask the entire data (based on preference)
        return raw_data

    mask = np.ones(raw_data._data.shape[1], dtype=bool)
    mask_start = np.random.randint(0, raw_data._data.shape[1] - segment_length)
    mask[mask_start:mask_start+segment_length] = 0
    raw_data._data[:, mask] = 0
    return raw_data




def jitter_signal(raw_data, max_shift=10):
    """Apply small random shifts in the time axis."""
    shift = np.random.randint(-max_shift, max_shift)
    shifted_data = np.roll(raw_data._data, shift, axis=1)
    return mne.io.RawArray(shifted_data, raw_data.info)

def scale_signal(raw_data, scale_factor_range=(0.8, 1.2)):
    """Randomly amplify or diminish the amplitude."""
    scale_factor = np.random.uniform(*scale_factor_range)
    scaled_data = raw_data._data * scale_factor
    return mne.io.RawArray(scaled_data, raw_data.info)

def channel_shuffle(raw_data):
    """Randomly shuffle the order of EEG channels."""
    shuffled_data = np.random.permutation(raw_data._data)
    return mne.io.RawArray(shuffled_data, raw_data.info)

# Map of augmentation methods to their function calls
AUGMENTATION_METHODS = {
    "channel_shuffle": channel_shuffle,
    "time_masking": time_masking,
    "jitter_signal": jitter_signal,
    "scale_signal": scale_signal
}

def process_file(input_file_path, output_folder_path):
    # Load the EEG data
    eeg_data = np.load(input_file_path)
    info = mne.create_info(ch_names=[f'ch{i+1}' for i in range(eeg_data.shape[0])], 
                           ch_types='eeg', sfreq=250.0)
    raw = mne.io.RawArray(eeg_data, info)
    
    # Apply each augmentation method and save the result
    for method_name, method_func in AUGMENTATION_METHODS.items():
        augmented_raw = method_func(raw.copy())
        
        # Define the output filename
        base_filename = os.path.basename(input_file_path)
        output_filename = f"processed_{method_name}_{base_filename}"
        output_file_path = os.path.join(output_folder_path, output_filename)
        
        # Save the augmented data
        np.save(output_file_path, augmented_raw._data)

# Your existing script, modified to fit the new process_file function
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
            
            process_file(input_file_path, output_folder_path)

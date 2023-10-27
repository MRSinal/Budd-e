import numpy as np
import os

# Function to apply data augmentation techniques
def augment_data(data):
    # Add random noise to the data
    noisy_data = data + np.random.normal(0, 0.1, data.shape)
    
    # Random time shift (positive or negative)
    shift_amount = np.random.randint(-50, 50)  # You can adjust the shift range
    shifted_data = np.roll(data, shift_amount, axis=1)
    
    # Random amplitude scaling
    scaling_factor = np.random.uniform(0.8, 1.2)  # You can adjust the scaling range
    scaled_data = data * scaling_factor
    
    # Combine augmented data
    augmented_data = np.vstack((noisy_data, shifted_data, scaled_data))
    
    return augmented_data

# Path to the folder containing .npy files
folder_path = 'C:/Users/Balint/Desktop/UNI-Stuff/Hackathon/neutral_processed'

# Loop through .npy files, augment data, and save augmented data
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        
        # Apply data augmentation
        augmented_data = augment_data(data)
        
        # Save augmented data to new .npy file
        augmented_filename = filename.replace('.npy', '_augmented.npy')
        augmented_file_path = os.path.join(folder_path, augmented_filename)
        np.save(augmented_file_path, augmented_data)

print("Data augmentation complete.")

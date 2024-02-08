import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import record_eeg
import prepro

import os 
BASE_PATH = "C:/Users/Balint/Documents/Budd-e/" # Add path to the folder containing the data
folders = ["positive_processed", "negative_processed", "neutral_processed"]
labels_map = {"positive_processed": 0, "negative_processed": 1, "neutral_processed": 2}
    

# Extract features from EEG data (for demonstration, let's compute mean and variance)
def extract_features(data):
    # Mean and variance as features (you can expand this to include more sophisticated features)
    mean = np.mean(data)
    var = np.var(data)
    return np.hstack((mean, var))


def load_and_process_data(BASE_PATH, folders):
    """Loads the EEG data, processes it, and extracts features."""
    X = []  # Features
    y = []  # Labels

    for folder in folders:
        folder_path = os.path.join(BASE_PATH, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(folder_path, filename)
                data = np.load(file_path)

                # Extract features
                features = extract_features(data)

                X.append(features)
                y.append(labels_map[folder])

    return np.array(X), np.array(y)

def rf_training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf_classifier.predict(X_test)

    return rf_classifier

def make_predictions(clf, z):
    predicted_label = clf.predict(z.reshape(1, -1))
    return predicted_label

def rec_n_pred(device_set):
    X, y = load_and_process_data(BASE_PATH, folders)
    z = extract_features(prepro.preprocess_data(record_eeg.record_data(device_set)))
    return make_predictions(rf_training(X, y), z)
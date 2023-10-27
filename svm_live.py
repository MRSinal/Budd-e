import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
import prepro
import record_eeg

BASE_PATH = "C:/Users/Balint/Desktop/UNI-Stuff/Hackathon/all_data"
folders = ["positive_processed", "negative_processed", "neutral_processed"]
labels_map = {"positive_processed": 0, "negative_processed": 1, "neutral_processed": 2}

def extract_features(data):
    """Extracts features from the data."""
    mean = np.mean(data)
    var = np.var(data)
    return np.hstack((mean, var))

def load_and_process_data():
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

def train_and_test_svm(X, y, z):
    """Trains an SVM and tests its accuracy."""
    # Splitting the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training SVM
    clf = SVC(kernel='rbf', C=1000, gamma=0.01)
    clf.fit(X_train, y_train)

    # Testing SVM
    y_pred = clf.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))
    
    predicted_label = clf.predict(z.reshape(1, -1))
    print(predicted_label)
    return clf
def svm_predicted_label(X, y, z):
    """Trains an SVM and tests its accuracy."""
    # Splitting the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training SVM
    clf = SVC(kernel='rbf', C=1000, gamma=0.01)
    clf.fit(X_train, y_train)

    # Testing SVM
    y_pred = clf.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))
    
    predicted_label = clf.predict(z.reshape(1, -1))
    print(predicted_label)
    return predicted_label

def main():
    record_eeg.connect_to_device()
    X, y = load_and_process_data()
    z = extract_features(prepro.preprocess_data(record_eeg.record_data()))
    train_and_test_svm(X, y, z)
    

if __name__ == "__main__":
    main()
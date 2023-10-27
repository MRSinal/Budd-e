import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming you have your training data (train_features, train_labels) and test data (test_features, test_labels) loaded.

# Step 1: Train SVM using sklearn
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # RBF Kernel SVM
svm.fit(train_features, train_labels)

# Step 2: Use PyTorch to convert SVM decision function to torch tensor
class SVMPredictor(nn.Module):
    def __init__(self, svm_model):
        super(SVMPredictor, self).__init__()
        self.svm_model = svm_model

    def forward(self, x):
        # Convert decision function values to tensor
        with torch.no_grad():
            decision_values = self.svm_model.decision_function(x)
            return torch.tensor(decision_values)

# Step 3: Create an instance of SVMPredictor and use it for predictions
svm_predictor = SVMPredictor(svm)
svm_predictions = svm_predictor(torch.tensor(test_features))

# Step 4: Calculate accuracy using sklearn's accuracy_score
svm_accuracy = accuracy_score(test_labels, svm_predictions > 0)
print("SVM Accuracy:", svm_accuracy)
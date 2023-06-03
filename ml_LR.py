import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression as LR
import pickle

# Load the training data from the .npy file
data = np.load('3classfeatures.npy', allow_pickle=True).item()

# Separate the features and labels
features = []
labels = []
for class_label, class_data in data.items():
    features.extend(class_data)
    labels.extend([class_label] * len(class_data))

# Convert the lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.abs(features), labels, test_size=0.2, random_state=42)

nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))

classifier = LR()
classifier.fit(d2_train_dataset, y_train)

nsamples2, nx2, ny2 = X_test.shape
d2_train_dataset2 = X_test.reshape((nsamples2,nx2*ny2))

# Predict the labels for the test set
y_pred = classifier.predict(d2_train_dataset2)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

test_features = []
test_labels = []
test_data = np.load('3classtestfeatures.npy', allow_pickle=True).item()
for test_class_label, test_class_data in test_data.items():
    test_features.extend(test_class_data)
    test_labels.extend([test_class_label] * len(test_class_data))
    
test_features = np.array(test_features)
test_labels = np.array(test_labels)

ntestsamples, nxtest, nytest = test_features.shape
d2_test_dataset = test_features.reshape((ntestsamples,nxtest*nytest))

predictions = classifier.predict(np.abs(d2_test_dataset))
for prediction in predictions:
    print("Predicted class:", prediction)

############### Save Model ####################
# pickle.dump(classifier, 'trained_SVM_model.pkl')
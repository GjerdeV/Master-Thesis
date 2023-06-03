import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt
import pickle
import joblib

# Load the training data from the .npy file
data = np.load('anon3varselect_3.npy', allow_pickle=True).item()
# model = pickle.load(open('model1.pkl', 'rb'))
model = joblib.load('modelvar3.sav')

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
X_train, X_test, y_train, y_test = train_test_split((features), labels, test_size=0.2, random_state=42)

nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))

# Make and train the classifier
# model = RFC(n_estimators=9)
# model.fit(d2_train_dataset, y_train)
# # model.fit(X_train, y_train)

nsamples2, nx2, ny2 = X_test.shape
d2_test_dataset = X_test.reshape((nsamples2,nx2*ny2))
# y_pred = model.predict(d2_train_dataset2)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print(classification_report(y_test, y_pred))


# Train existing model
accuracies = []
for i in range(1000):
    # model = pickle.load(open('model1.pkl', 'rb'))
    model.fit(d2_train_dataset, y_train)
    # model.fit(X_train, y_train)

    # nsamples2, nx2, ny2 = X_test.shape
    # d2_test_dataset = X_test.reshape((nsamples2,nx2*ny2))

    # Predict the labels for the test set
    y_pred = model.predict(d2_test_dataset)
    # y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # Printing accuracy and classification report:
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    accuracies.append(accuracy)
    # pickle.dump(model, open('model1.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(accuracies, 'b-')
plt.plot(np.poly1d(accuracies), 'r*')
plt.title('Model performance')
plt.ylabel('Accuracy')
plt.xlabel('Training iteration')
plt.grid(True)
    
    
    


############### Save Model ####################
# joblib.dump(classifier, open('model.sav', 'wb'))
joblib.dump(model, 'modelvar3.sav')
# pickle.dump(model, open('model1.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

plt.show()
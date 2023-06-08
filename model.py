import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import PassiveAggressiveClassifier as PAC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import matplotlib.pyplot as plt
import joblib

# Trained with data: l, 3 and 4
# Load the training data from the .npy file
data = np.load('ssqpca38.npy', allow_pickle=True).item()

# Load the trained model
model = joblib.load('model.sav')

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
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.05, random_state=42)

# used for direct CWT as features:
# nsamples, nx, ny = X_train.shape
# d2_train_dataset = X_train.reshape((nsamples,nx*ny))

# Make and train the classifier
# model = RFC(random_state=42, n_jobs=-1, max_depth=3, n_estimators=30, oob_score=True)


###
X_train, X_test, y_train, y_test = train_test_split(np.abs(features), labels, train_size=0.8, random_state=42)
# nsamples, nx, ny = X_train.shape
# X_train = X_train.reshape((nsamples,nx*ny))

# model.fit(X_train, y_train)
# print('Cross-validation score: ')
# print(model.oob_score_)
# params = {
#     'max_depth': [2,3,5,10,20],
#     'min_samples_leaf': [5,10,15,20,50,100,200],
#     'n_estimators': [10,25,30,50,100,200]
# }

# grid_search = GridSearchCV(estimator=model, param_grid=params, cv=4,n_jobs=-1, verbose=1, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print(grid_search.best_score_)

# rf_best = grid_search.best_estimator_
# print(model.base_estimator_[5])

# plt.figure(figsize=(80,40))
# plot_tree(rf_best, filled=True)

###
# Add feature selection method
# feature_selector = SelectKBest(score_func=f_classif, k=30000)
# feature_selector = VarianceThreshold(threshold=(.8 * (1 - .2)))
# feature_selector = VarianceThreshold()
# Apply the feature selector to the dataset

# model.fit(d2_train_dataset, y_train)
# model.fit(X_train, y_train)

# # Train existing model
accuracies = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
    # nsamples, nx, ny = X_train.shape
    # X_train = X_train.reshape((nsamples,nx*ny))
    
    # nsamples, nx, ny = X_test.shape
    # X_test = X_test.reshape((nsamples,nx*ny))
    #############
    # X_train = feature_selector.fit_transform(X_train, y_train)
    model.fit(X_train, y_train)
    # X_test = feature_selector.transform(X_test)
    #############
    
    # model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)


print('Cross-validation score: ')
print(model.oob_score_)

# # Plotting evolvement of accuracy
plt.figure()
plt.plot(accuracies, 'b-')
plt.plot(np.poly1d(accuracies), 'r*')
plt.title('Model performance')
plt.ylabel('Accuracy')
plt.xlabel('Training iteration')
plt.grid(True)
    
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# # # Predicting unseen data
# # Load the unseen data from the .npy file
data2 = np.load('ssqpca58.npy', allow_pickle=True).item()

features2 = []
labels2 = []
for class_label2, class_data2 in data2.items():
    features2.extend(class_data2)
    labels2.extend([class_label2] * len(class_data2))

# Convert the lists to numpy arrays
features2 = np.array(features2)
labels2 = np.array(labels2)

# nsamples2, nx2, ny2 = features2.shape
# features2 = features2.reshape((nsamples2,nx2*ny2))

# features2 = feature_selector.transform(features2)

# Predict and print report and show confusion matrix
y_pred2 = model.predict(features2)
print(classification_report(labels2, y_pred2))
print(accuracy_score(labels2, y_pred2))
cm2 = confusion_matrix(labels2, y_pred2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp.plot()

# ############### Save Model ####################
joblib.dump(model, 'model.sav')

plt.show()
This is code developed under a master thesis project for the purpose of analysing EEG data and attempt to train a machine learning model to classify different emotional responses.
The thesis is a part of the master's degree in Robottechnology and signalprocessing with specialisation in health technology at University of Stavanger.

Simple user guide:
Use final.py-script for preprocessing of smaller sampled data, e.g. 300.000 samples.
Make sure to set the correct values for bad channel detection and bad epoch rejection when saving features.

Use final2.py for preprocessing smaller data using Zhang-fit algorithm for baselining.

Use the script in longer_data_preprocessing.py for preprocessing and extracting features for longer data sets, e.g. 2.250.000 samples

The file model.py is for the purpose of doing iterative model training and testing.

ml_LR.py, ml_NB.py, ml_RF.py, ml_SVM.py, ml_kNN.py and ml_nn.py are files for proposed methods for development of different machine learning models.

ssqpcaNN .npy-files are the features extracted using synchrosqueezed wavelet transformation with PCA of 12 components. First N refers to the respondent number and the second N refers to the total number of classes. Respondent l is named l for large, which indicates the features extracted from one of the larger data sets.

The following versions of some of the different packages in pip are used for this project:
BaselineRemoval==0.1.4
flake8==3.8.3
gitdb==4.0.5
GitPython==3.1.9
importlib-metadata==2.0.0
ipykernel==6.3.1
ipython==7.27.0
joblib==0.15.1
matplotlib==3.5.3
matplotlib-inline==0.1.2
numpy==1.21.3
pandas==1.0.5
py==1.8.2
pylint==2.2.2
python-adjust==1.0.3
python-dateutil==2.7.5
python-utils==3.5.2
PyWavelets==1.1.1
scikit-image==0.18.3
scikit-learn==0.23.1
scipy==1.4.1
seaborn==0.12.2
serial==0.0.97

This is code developed under a master thesis project for the purpose of analysing EEG data and attempt to train a machine learning model to classify different emotional responses.
The thesis is a part of the master's degree in Robottechnology and signalprocessing with specialisation in health technology at University of Stavanger.

Simple user guide:
Use final.py-file for preprocessing of smaller sampled data, e.g. 300.000 samples.
Make sure to set the correct values for bad channel detection and bad epoch rejection when saving features.

Use final2.py for preprocessing smaller data using Zhang-fit algorithm for baselining.

Use longer_data_preprocessing for preprocessing and extracting features for longer data sets, e.g. 2,25m samples

The file modeltraining is for the purpose of doing iterative model training and testing.

ml_LR.py, ml_NB.py, ml_RF.py, ml_SVM.py, ml_kNN.py and ml_nn.py are files for proposed methods for development of machine learning models.

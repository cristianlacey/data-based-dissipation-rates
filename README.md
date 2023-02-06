# Data-Based Instantaneous Conditional Dissipation Rate Profiles for Premixed Combustion

This repository stores the deep neural network (DNN) model trained to predict the normalized instantaneous dissipation rate profiles for premixed turbuent combustion. The model was trained using ```TensorFlow``` version 2.4.1. The Anaconda environment used to train and postprocess the DNN model is provided in ```tf-gpu.yml``` and can be imported with the command ```conda env create -n myenv -f tf-gpu.yml```.

An example script for loading the DNN model stored in ```lambda_profile_dnn/``` and generating model predictions for a test dataset stored as a CSV at ```test_data.csv``` is provided as follows:

```python
# Import required libraries
import pandas as pd
import tensorflow as tf

# Define directories and column names
test_data_dir = 'test_data.csv'
dnn_dir = 'lambda_profile_dnn'
training_feature_names = ['FPROG', \
                          'PROGV', \
                          'dFPROG', \
                          'FS', \
                          'DELFILT', \
                          'FDIFF', \
                          'FPROGSRC', \
                          'FRHO', \
                          'FCHIPP', \
                          'alpha', \
                          'beta', \
                          'gamma', \
                          'alpha_align', \
                          'beta_align', \
                          'gamma_align']
training_label_names =  ['f_vec_'+str(i) for i in range(32)]

# Load testing dataset (example for CSV data)
test_df = pd.read_csv(test_data_dir)
test_features = test_df[training_feature_names]

# Load DNN model
dnn_model = tf.keras.models.load_model(dnn_dir)

# Generate test predictions
dnn_predictions = pd.DataFrame(dnn_model.predict(test_features), columns=training_label_names)

```

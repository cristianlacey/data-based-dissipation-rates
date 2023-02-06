# Data-Based Instantaneous Conditional Dissipation Rate Profile Model for Premixed Turbulent Combustion

This repository contains the deep neural network (DNN) model trained to predict the normalized instantaneous dissipation rate profiles for manifold models of premixed turbulent combustion. The DNN was trained using ```TensorFlow``` version 2.4.1. The Anaconda environment used to train and postprocess the DNN model is provided in ```tf-gpu.yml``` and can be imported using the command ```conda env create -n myenv -f tf-gpu.yml```.

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

The selected training features and corresponding column names are summarized in the following table:

| Input Feature | Description | Column Name   |
| :---          |    :----:   |        ---:   |
|               | filtered progress variable                  |    ```FPROG```   |
|               | progress variable subfilter variance        |    ```PROGV```   |
|               | magnitude of the filtered progress variable gradient                 |    ```dFPROG```  |
|               | magnitude of the filtered strain rate        |    ```FS```      |
|               | filter size                 |    ```DELFILT``` |
|               | filtered molecular diffusivity        |    ```FDIFF```   |
|               | filtered progress variable source term                  |    ```FPROGSRC```|
|               | filtered progress variable dissipation rate       |    ```FCHIPP```  |
|               | principal rate of strain (smallest)                 |    ```alpha```   |
|               | principal rate of strain (intermediate)        |    ```beta```    |
|               | principal rate of strain (largest)                  |    ```gamma```   |
|               | alignment of (smallest) principal rate of strain with progress variable gradient        |    ```alpha_align```   |
|               | alignment of (intermediate) principal rate of strain with progress variable gradient                  |    ```beta_align```    |
|               | alignment of (largest) principal rate of strain with progress variable gradient        |    ```gamma_align```   |

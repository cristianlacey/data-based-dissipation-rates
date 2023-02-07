# Data-Based Instantaneous Conditional Dissipation Rate Profile Model for Premixed Turbulent Combustion

This repository contains the deep neural network (DNN) model trained to predict the normalized instantaneous dissipation rate profiles for manifold models of premixed turbulent combustion. The DNN was trained using ```TensorFlow``` version 2.4.1. The Anaconda environment used to train and postprocess the DNN model is provided in ```tf-gpu.yml``` and can be imported using the command ```conda env create -n myenv -f tf-gpu.yml```.

## Example Script

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

## Expected Training Features

The selected training features and corresponding column names are summarized in the following table:

| Input Feature | Description | Column Name   |
| :---:         |    :----   |        :---:   |
|   $\widetilde{\Lambda}$            | filtered progress variable                  |    ```FPROG```   |
|     $\Lambda_v$          | progress variable subfilter variance        |    ```PROGV```   |
|       $\lvert \nabla \widetilde \Lambda \rvert$        | magnitude of the filtered progress variable gradient                 |    ```dFPROG```  |
|        $\lvert \widetilde S \rvert$       | magnitude of the filtered strain rate        |    ```FS```      |
|        $\Delta_L \equiv V_{\rm stencil}^{1/3}$       | local filter size                 |    ```DELFILT``` |
|       $\widetilde{D}_{\Lambda}$        | filtered molecular diffusivity        |    ```FDIFF```   |
|       $\overline{\rho}$        | filtered density        |    ```FRHO```   |
|      $\overline{\dot{m}}_{\Lambda}$         | filtered progress variable source term                  |    ```FPROGSRC```|
|      $\widetilde \chi_{\Lambda \Lambda}$         | filtered progress variable dissipation rate       |    ```FCHIPP```  |
|      $\alpha$         | principal rate of strain (smallest)                 |    ```alpha```   |
|       $\beta$        | principal rate of strain (intermediate)        |    ```beta```    |
|      $\gamma$         | principal rate of strain (largest)                  |    ```gamma```   |
|        $e_{\alpha}\cdot \nabla \widetilde{\Lambda}/\lvert \nabla \widetilde{\Lambda}\rvert$       | alignment of (smallest) principal rate of strain with progress variable gradient        |    ```alpha_align```   |
|       $e_{\beta}\cdot \nabla \widetilde{\Lambda}/\lvert \nabla \widetilde{\Lambda}\rvert$        | alignment of (intermediate) principal rate of strain with progress variable gradient                  |    ```beta_align```    |
|       $e_{\gamma}\cdot \nabla \widetilde{\Lambda}/\lvert \nabla \widetilde{\Lambda}\rvert$        | alignment of (largest) principal rate of strain with progress variable gradient        |    ```gamma_align```   |


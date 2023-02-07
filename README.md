# Data-Based Instantaneous Conditional Dissipation Rate Profile Model for Premixed Turbulent Combustion

This repository contains a deep neural network (DNN) model for use with manifold models of premixed turbulent combustion. The DNN was trained using ```TensorFlow``` version 2.4.1. The Anaconda environment containing all dependencies used to train and postprocess the DNN model is provided in ```tf-gpu.yml```. All associated libraries will automatically be installed to a new virtual environment named ```myenv``` via the command ```conda env create -n myenv -f tf-gpu.yml```.


## DNN Model Outputs

The DNN predicts the normalized instantaneous dissipation rate profiles defined according to

$$ g(\Lambda;\Lambda_{\rm ref}) \equiv \frac{\chi_{\Lambda \Lambda}(\Lambda)}{\chi_{\Lambda \Lambda}(\Lambda_{\rm ref})},$$

where $\Lambda$ is the progress variable, $\Lambda_{\rm ref} = 0.5$ is the reference progress variable, and $\chi_{\Lambda \Lambda} \equiv 2 D_{\Lambda} \nabla \Lambda \cdot \nabla \Lambda$ is the progress variable dissipation rate. The DNN outputs comprise a normalized instantaneous dissipation rate profile prediction discretized on a uniform 32-point grid in progress variable space. Details of the neural network architecture and training procedure are outlined in the following publication:

- C. E. Lacey, S. Sundaresan, M. E. Mueller, Data-based instantaneous conditional progress variable dissipation rate modeling for turbulent premixed combustion, Combustion and Flame 250 (2023) submitted.

If you use the DNN model in any published work, we kindly ask you to cite this paper.


## DNN Model Input Features

The DNN expects input features with corresponding column names as summarized in the following table:

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


## Generating Model Predictions

An example script that loads the DNN model and generates model predictions for a test dataset is provided below. The script assumes the DNN model is stored in the directory ```lambda_profile_dnn/``` and the test dataset is stored in the working directory as a CSV file named ```test_data.csv```.

```python
# Import required libraries
import pandas as pd
import tensorflow as tf

# Define directories and column names
test_data_dir = 'test_data.csv'
dnn_dir = 'lambda_profile_dnn'
label_names =  ['g_' + str(i) for i in range(32)]
feature_names = ['FPROG', \
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

# Load testing dataset (example for CSV data)
test_df = pd.read_csv(test_data_dir)
test_features = test_df[feature_names]

# Load DNN model
dnn_model = tf.keras.models.load_model(dnn_dir)

# Generate test predictions
dnn_predictions = pd.DataFrame(dnn_model.predict(test_features), columns=label_names)

```

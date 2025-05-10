# SECOM – Notebook Guide

This guide provides an overview of the Jupyter notebooks associated with the SECOM dataset, which is used for detecting manufacturing defects through anomaly detection techniques.

## Structure

- **EDA_Preprocessing.ipynb**  
  Performs exploratory data analysis (EDA) and preprocessing on the raw SECOM dataset. This includes data inspection, handling missing values, feature scaling, and preparing the dataset for model training.

- **Autoencoder.ipynb**  
  Implements an autoencoder-based model for anomaly detection. It includes model-specific preprocessing, network architecture design, training with hyperparameter tuning, evaluation, and result saving.

- **ISForest.ipynb**  
  Trains an Isolation Forest model on the processed data. This notebook covers tailored preprocessing, model training, evaluation metrics, and saving results.

- **OneClassSVM.ipynb**  
  Applies a One-Class SVM model for detecting manufacturing defects. It includes model-specific preprocessing, training, hyperparameter tuning, evaluation, and result saving.

Each notebook is self-contained and includes detailed explanations for preprocessing, model training, tuning, evaluation, and interpretation of results.

## Usage

1. Set up the environment using the provided `requirements.txt` file.  
   Python 3.12.4 is recommended.
   ```bash
   pip install -r requirements.txt
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Run the notebooks in the following order:
   - `EDA_Preprocessing.ipynb`
   - One or more model notebooks:
     - `Autoencoder.ipynb`
     - `ISForest.ipynb`
     - `OneClassSVM.ipynb`

## Notes

- Ensure the SECOM dataset is placed in the current directory or the specified data path.
- Output files such as trained models, metrics, and visualizations will be saved locally for further analysis and comparison.
# NSL KDD – Notebook Guide

This guide provides an overview of the Jupyter notebooks related to the NSL-KDD dataset, a widely used benchmark dataset for network intrusion detection and anomaly detection tasks.

## Structure

- **EDA.ipynb**  
  Performs exploratory data analysis (EDA) on the raw dataset. It provides insights into the data, identifies patterns, handles missing values, and gives an overview of feature distributions and correlations.

- **Preprocessing.ipynb**  
  Focuses on preparing the dataset for modeling. This includes data cleaning, encoding categorical features, feature selection, scaling, and transforming the dataset into a format suitable for model training.

- **Autoencoder.ipynb**  
  Implements an autoencoder-based model for anomaly detection. The notebook covers model-specific preprocessing, architecture design, training with hyperparameter tuning, performance evaluation, and result saving.

- **ISForest.ipynb**  
  Trains an Isolation Forest model. Includes model-specific data preparation, model training, performance evaluation, and saving results for further analysis.

- **OneClassSVM.ipynb**  
  Applies a One-Class SVM model for intrusion detection. Covers data preprocessing, model training, hyperparameter tuning, evaluation, and saving the results.

Each notebook is self-contained and includes clear explanations of preprocessing, model training, tuning, evaluation, and result interpretation.

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

3. Run the notebooks in the following suggested order:
   - `EDA.ipynb`
   - `Preprocessing.ipynb`
   - One of the model notebooks: `Autoencoder.ipynb`, `ISForest.ipynb`, or `OneClassSVM.ipynb`

## Notes

- Ensure the NSL-KDD dataset files (e.g., `KDDTrain+.txt`, `KDDTest+.txt`) are placed in the current directory.
- Output files such as trained models and evaluation metrics will be saved locally for further review or integration.
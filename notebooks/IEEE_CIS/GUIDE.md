# IEEE_CIS – Notebook Guide

This folder contains Jupyter notebooks related to the IEEE-CIS fraud detection dataset, a financial dataset used for anomaly and fraud detection tasks.

## Structure

- **EDA_Preprocessing.ipynb**  
  This notebook performs exploratory data analysis (EDA) and preprocessing on the raw dataset. It provides insights into the data, handles missing values, encodes categorical features, and prepares the data for model training.

- **Autoencoder.ipynb**  
  Implements an autoencoder-based model for anomaly detection. The notebook includes preprocessing steps specific to the model, model architecture design, training with hyperparameter tuning, evaluation of performance, and saving the results.

- **ISForest.ipynb**  
  Trains an Isolation Forest model on the preprocessed data. This notebook includes model-specific preprocessing, training with relevant parameters, evaluation metrics, and result storage.

- **OneClassSVM.ipynb**  
  Applies a One-Class SVM model for fraud detection. It contains steps for preprocessing, training, parameter tuning, evaluation, and saving of results.

Each model notebook is self-contained and includes detailed explanations of each step: preprocessing, training, tuning, and evaluation.

## Usage

1. Set up the environment with required packages (Python 3.12.4 recommended). You can use a `requirements.txt` file provided.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Download the dataset from the [IEEE-CIS website](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place it in the `data/` directory.
3. Run `EDA_Preprocessing.ipynb` first to generate preprocessed data.
4. Then proceed with any of the model notebooks (Autoencoder, ISForest, or OneClassSVM) for training and evaluation.

## Notes

- Make sure the dataset is placed correctly in the expected data directory (e.g., `data/`).
- Each notebook saves its outputs (metrics, models, results) for further analysis or comparison.
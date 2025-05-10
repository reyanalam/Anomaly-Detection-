from django.db import models
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.decomposition import PCA

class ModelManager:
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models from the models directory"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            return
        
        for dataset_type in ['SECCOM', 'NSL_KDD', 'IEEE_CIS']:
            model_file = os.path.join(self.model_path, f'{dataset_type.lower()}_model.joblib')
            scaler_file = os.path.join(self.model_path, f'{dataset_type.lower()}_scaler.joblib')
            
            if os.path.exists(model_file):
                self.models[dataset_type] = joblib.load(model_file)
            if os.path.exists(scaler_file):
                self.scalers[dataset_type] = joblib.load(scaler_file)
    
    def save_model(self, dataset_type, model, scaler):
        """Save a model and its scaler for a specific dataset type"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Save the model
        model_file = os.path.join(self.model_path, f'{dataset_type.lower()}_model.joblib')
        joblib.dump(model, model_file)
        self.models[dataset_type] = model
        
        # Save the scaler
        scaler_file = os.path.join(self.model_path, f'{dataset_type.lower()}_scaler.joblib')
        joblib.dump(scaler, scaler_file)
        self.scalers[dataset_type] = scaler
        
        print(f"Model and scaler saved for {dataset_type}")
    
    def get_model(self, dataset_type):
        """Get the model for a specific dataset type"""
        if dataset_type not in self.models:
            raise ValueError(f"No model found for dataset type: {dataset_type}")
        return self.models[dataset_type]
    
    def get_scaler(self, dataset_type):
        """Get the scaler for a specific dataset type"""
        if dataset_type not in self.scalers:
            raise ValueError(f"No scaler found for dataset type: {dataset_type}")
        return self.scalers[dataset_type]

def preprocess_seccom(df):
    """Preprocess SECCOM dataset"""
    # Remove columns with more than 10% missing values
    missing_threshold = len(df) * 0.1
    df = df.dropna(thresh=len(df) - missing_threshold, axis=1)
    
    # Fill remaining missing values with column mean
    df = df.fillna(df.mean())
    
    # Remove any non-numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Remove highly correlated features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df = df.drop(columns=to_drop)
    
    # Feature selection using mutual information
    X = df.drop(columns=['label']) if 'label' in df.columns else df
    y = df['label'] if 'label' in df.columns else None
    
    if y is not None:
        mi_selected = SelectKBest(score_func=mutual_info_classif, k=20)
        X_selected = mi_selected.fit_transform(X, y)
        selected_features = X.columns[mi_selected.get_support()]
        df = pd.DataFrame(X_selected, columns=selected_features)
    
    # Normalize the data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_scaled, scaler

def preprocess_nsl_kdd(df):
    """Preprocess NSL-KDD dataset"""
    # Convert specific columns to numeric
    numeric_columns = ['is_host_login', 'is_guest_login', 'logged_in', 'land']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove columns with only one unique value
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop(columns=[col])
    
    # Convert categorical columns to numeric using Label Encoding
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Remove the 'num_outbound_cmds' column as it has only one unique value
    if 'num_outbound_cmds' in df.columns:
        df = df.drop(columns=['num_outbound_cmds'])
    
    # Feature selection using ANOVA F-value
    X = df.drop(columns=['label']) if 'label' in df.columns else df
    y = df['label'] if 'label' in df.columns else None
    
    if y is not None:
        selector = SelectKBest(score_func=f_classif, k=20)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        df = pd.DataFrame(X_selected, columns=selected_features)
    
    # Normalize the data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_scaled, scaler

def preprocess_ieee_cis(df):
    """Preprocess IEEE CIS dataset"""
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert categorical variables to numerical
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Remove highly correlated features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df = df.drop(columns=to_drop)
    
    # Feature selection using mutual information
    X = df.drop(columns=['isFraud']) if 'isFraud' in df.columns else df
    y = df['isFraud'] if 'isFraud' in df.columns else None
    
    if y is not None:
        mi_selected = SelectKBest(score_func=mutual_info_classif, k=20)
        X_selected = mi_selected.fit_transform(X, y)
        selected_features = X.columns[mi_selected.get_support()]
        df = pd.DataFrame(X_selected, columns=selected_features)
    
    # Normalize the data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_scaled, scaler

def get_preprocessor(dataset_type):
    """Return the appropriate preprocessing function based on dataset type"""
    preprocessors = {
        'SECCOM': preprocess_seccom,
        'NSL_KDD': preprocess_nsl_kdd,
        'IEEE_CIS': preprocess_ieee_cis
    }
    
    if dataset_type not in preprocessors:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return preprocessors[dataset_type]

# Example of how to train and save a model
def train_and_save_model(dataset_type, X, y):
    """Train a model and save it using the ModelManager"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Create and train the model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save the model and scaler
    model_manager.save_model(dataset_type, model, scaler)
    
    return model, scaler

# Initialize the model manager
model_manager = ModelManager()

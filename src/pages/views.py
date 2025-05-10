from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
from datetime import datetime
import sys
import importlib.util
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from .models import get_preprocessor, model_manager, train_and_save_model

@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def about(request):
    return render(request, 'about.html')

@csrf_exempt
def contact(request):
    return render(request, 'contact.html')

@csrf_exempt
def login(request):
    return render(request, 'login.html')

@csrf_exempt
def signup(request):
    return render(request, 'signup.html')

@csrf_exempt
def upload(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            # Get the uploaded file and dataset type
            uploaded_file = request.FILES['file']
            dataset_type = request.POST.get('datasetType')
            print("Uploaded succesfull")
            
            # Print available models and scalers
            print("Available models:", model_manager.models.keys())
            print("Available scalers:", model_manager.scalers.keys())
            
            # Validate dataset type
            valid_types = ['NSL_KDD', 'IEEE_CIS', 'SECCOM']
            if not dataset_type or dataset_type not in valid_types:
                return JsonResponse({'error': f'Please select a valid dataset type. Must be one of {valid_types}'}, status=400)
            
            # Read the file based on its extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                return JsonResponse({'error': 'Unsupported file format. Please upload CSV or Excel files.'}, status=400)
            print("File read successfully")
            
            # Get first 10 columns before preprocessing
            first_10_columns = df.columns[:10].tolist()
            print("First 10 columns extracted successfully")
            
            # Get the appropriate preprocessor for the dataset type
            try:
                preprocessor = get_preprocessor(dataset_type)
                print("Preprocessor loaded successfully")
            except ValueError as e:
                return JsonResponse({'error': str(e)}, status=400)
            
            # Preprocess the data
            #processed_data, _ = preprocessor(df)
            processed_data = df
            print("Data preprocessed successfully")
            print("Processed data shape:", processed_data.shape)
            print("Processed data columns:", processed_data.columns.tolist())
            
            # Check if model exists, if not train and save it
            if dataset_type not in model_manager.models:
                print(f"Training new model for {dataset_type}")
                # Assuming the last column is the target variable
                X = processed_data.iloc[:, :-1]
                y = processed_data.iloc[:, -1]
                model, scaler = train_and_save_model(dataset_type, X, y)
            else:
                # Get the existing model and scaler
                model = model_manager.get_model(dataset_type)
                scaler = model_manager.get_scaler(dataset_type)
                print(f"Using existing model for {dataset_type}")
            
            try:
                # Ensure we're using only the features that were used during training
                # Remove the target column if it exists
                if 'class' in processed_data.columns:
                    processed_data = processed_data.drop('class', axis=1)
                if 'label' in processed_data.columns:
                    processed_data = processed_data.drop('label', axis=1)
                
                # Ensure data is numeric
                processed_data = processed_data.select_dtypes(include=[np.number])
                print("Numeric columns selected:", processed_data.columns.tolist())
                
                # Scale the data using the saved scaler
                scaled_data = scaler.transform(processed_data)
                print("Data scaled successfully")
                print("Scaled data shape:", scaled_data.shape)
                
                # Make predictions
                predictions = model.predict(scaled_data)
                print("Predictions made successfully")
                print("Number of predictions:", len(predictions))
                
                # Create a new DataFrame with only the first 10 columns and predictions
                result_df = df[first_10_columns].copy()
                result_df['prediction'] = predictions
                
                # Convert DataFrame to list of dictionaries for template
                data = result_df.to_dict('records')
                
                return render(request, 'result.html', {
                    'columns': first_10_columns,
                    'data': data
                })
                
            except Exception as e:
                print(f"Error during scaling/prediction: {str(e)}")
                return JsonResponse({'error': f'Error during processing: {str(e)}'}, status=400)
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            error_data = {
                'columns': ['Error'],
                'data': [{'Error': str(e), 'prediction': 'Failed'}]
            }
            return render(request, 'result.html', error_data)
    
    return render(request, 'upload.html')

@csrf_exempt
def result(request):
    return render(request, 'result.html')



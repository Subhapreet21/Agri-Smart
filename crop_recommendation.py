import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import load_crop_data

def train_crop_recommendation_model():
    """
    Train a Random Forest model for crop recommendation based on soil parameters and environmental factors.
    """
    # Load the crop dataset
    df = load_crop_data()
    
    # Features and target
    X = df[['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
    y = df['Label']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'target_names': model.classes_.tolist()
    }
    
    with open('models/crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def load_crop_recommendation_model():
    """
    Load the trained crop recommendation model.
    """
    try:
        with open('models/crop_recommendation_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except (FileNotFoundError, pickle.UnpicklingError):
        # If model doesn't exist or has errors, train it
        return train_crop_recommendation_model()

def predict_crop(model_data, X):
    """
    Predict the most suitable crop based on input features.
    
    Args:
        model_data: Dictionary containing model, scaler, and related data
        X: Input features as a numpy array or DataFrame
    
    Returns:
        Predicted crop and probabilities for each crop
    """
    # Get the model and scaler
    model = model_data['model']
    scaler = model_data['scaler']
    target_names = model_data['target_names']
    
    # Standardize the input features
    X_scaled = scaler.transform(X)
    
    # Predict the crop
    prediction = model.predict(X_scaled)
    
    # Get probabilities for each crop
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Create a dictionary of crop probabilities
    crop_probabilities = {target_names[i]: probabilities[i] for i in range(len(target_names))}
    
    # Sort by probability (descending)
    crop_probabilities = dict(sorted(crop_probabilities.items(), key=lambda item: item[1], reverse=True))
    
    return prediction[0], crop_probabilities

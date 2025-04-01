import pandas as pd
import numpy as np
import os

def load_crop_data():
    """
    Load the crop dataset from CSV file.
    """
    df = pd.read_csv('crop_dataset.csv')
    return df

def get_crop_info(df, crop_name):
    """
    Get information about a specific crop from the dataset.
    
    Args:
        df: DataFrame containing crop data
        crop_name: Name of the crop
    
    Returns:
        Dictionary with crop information
    """
    # Filter the dataframe for the specific crop
    crop_data = df[df['Label'] == crop_name]
    
    if crop_data.empty:
        return None
    
    # Get the first row (or compute averages for numerical columns if needed)
    crop_info = crop_data.iloc[0].to_dict()
    
    return crop_info

def preprocess_features(input_features):
    """
    Preprocess the input features for prediction.
    
    Args:
        input_features: Dictionary of input features
    
    Returns:
        NumPy array of preprocessed features
    """
    # Get the feature values in the correct order
    feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    X = np.array([[input_features[feature] for feature in feature_names]])
    
    return X

def extract_crop_parameter_ranges(df):
    """
    Extract parameter ranges for each crop.
    
    Args:
        df: DataFrame containing crop data
    
    Returns:
        Dictionary with parameter ranges for each crop
    """
    crops = df['Label'].unique()
    parameters = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    
    crop_ranges = {}
    
    for crop in crops:
        crop_data = df[df['Label'] == crop]
        
        crop_ranges[crop] = {}
        
        for param in parameters:
            crop_ranges[crop][param] = {
                'min': crop_data[param].min(),
                'max': crop_data[param].max(),
                'mean': crop_data[param].mean(),
                'median': crop_data[param].median()
            }
    
    return crop_ranges

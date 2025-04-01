import pandas as pd
import numpy as np
import os

def load_crop_data():
    """
    Load the crop dataset from CSV file.
    """
    try:
        df = pd.read_csv('crop_dataset.csv')
        return df
    except Exception as e:
        print(f"Error loading crop dataset: {e}")
        # Create a simple dataset with a few rows for demo purposes
        return pd.DataFrame({
            'N': [83, 60, 40],
            'P': [45, 55, 30],
            'K': [60, 40, 35],
            'Temperature': [25, 23, 28],
            'Humidity': [75, 50, 60],
            'pH': [6.5, 7.0, 6.2],
            'Rainfall': [200, 150, 250],
            'Label': ['Rice', 'Maize', 'Wheat'],
            'Disease_Prone': ['No', 'Yes', 'No'],
            'Common_Disease(Fungal)': ['None', 'Rust', 'None'],
            'Common_Disease(Bacterial)': ['Leaf Blight', 'None', 'None'],
            'Common_Disease(Viral)': ['None', 'None', 'None'],
            'Salinity_dS_m': [2.0, 1.5, 1.8],
            'Water_Requirement': [450, 350, 400],
            'Disease_Resistance_Score': [7.0, 5.5, 6.0],
            'Nutrient_Deficiency': ['None', 'Nitrogen', 'Phosphorus']
        })

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
    # Base features that are always used
    base_features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    
    # Get the feature values in the correct order
    feature_values = []
    for feature in base_features:
        feature_values.append(input_features.get(feature, 0))
    
    # Create the features array
    X = np.array([feature_values])
    
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
    
    # Base parameters that are always present
    base_parameters = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    
    # Extended parameters that might be present
    extended_parameters = ['Salinity_dS_m', 'Water_Requirement', 'Disease_Resistance_Score']
    
    # Determine which parameters exist in the dataset
    parameters = base_parameters.copy()
    for param in extended_parameters:
        if param in df.columns:
            parameters.append(param)
    
    crop_ranges = {}
    
    for crop in crops:
        crop_data = df[df['Label'] == crop]
        
        crop_ranges[crop] = {}
        
        for param in parameters:
            if param in df.columns:
                try:
                    crop_ranges[crop][param] = {
                        'min': crop_data[param].min(),
                        'max': crop_data[param].max(),
                        'mean': crop_data[param].mean(),
                        'median': crop_data[param].median()
                    }
                except:
                    # If parameter values can't be calculated, use placeholders
                    crop_ranges[crop][param] = {
                        'min': 0,
                        'max': 0,
                        'mean': 0,
                        'median': 0
                    }
    
    return crop_ranges

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
    # Get the feature values in the correct order
    feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 
                    'Salinity_dS_m', 'Water_Requirement', 'Disease_Resistance_Score']
    
    # Set default values for new features if not provided in input
    default_values = {
        'Salinity_dS_m': 2.0,  # Default salinity
        'Water_Requirement': 400.0,  # Default water requirement
        'Disease_Resistance_Score': 5.0  # Default disease resistance score
    }
    
    # For each feature, use the provided value or the default
    feature_values = []
    for feature in feature_names:
        if feature in input_features:
            feature_values.append(input_features[feature])
        else:
            feature_values.append(default_values.get(feature, 0))
    
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
    parameters = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 
                 'Salinity_dS_m', 'Water_Requirement', 'Disease_Resistance_Score']
    
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

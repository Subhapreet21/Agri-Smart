import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    Preprocess the uploaded image for disease identification.
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed image as numpy array
    """
    # Resize the image
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values (if not done by the model internally)
    img_array = img_array / 255.0
    
    return img_array

def identify_disease(image):
    """
    Identify crop disease from an image.
    
    In a real implementation, this would use a trained deep learning model.
    For this example, we'll return simulated results.
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary with disease identification results
    """
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # In a real implementation, this would be where we'd use a model to predict
    # For this example, we'll simulate a random disease detection
    
    # Sample disease information (for demonstration)
    diseases = [
        {
            'is_disease_detected': True,
            'disease_name': 'Late Blight',
            'disease_description': 'Late blight is a plant disease caused by the oomycete Phytophthora infestans. It primarily affects potatoes and tomatoes, causing significant damage to leaves, stems, and fruits.',
            'treatments': [
                'Apply fungicides containing chlorothalonil, mancozeb, or copper',
                'Remove and destroy infected plant parts',
                'Increase plant spacing to improve air circulation'
            ],
            'prevention': [
                'Use resistant varieties when possible',
                'Provide good drainage in the field',
                'Avoid overhead irrigation',
                'Practice crop rotation'
            ]
        },
        {
            'is_disease_detected': True,
            'disease_name': 'Powdery Mildew',
            'disease_description': 'Powdery mildew is a fungal disease that affects a wide range of plants. It appears as a white to gray powdery growth on leaf surfaces, stems, and sometimes fruit.',
            'treatments': [
                'Apply fungicides containing sulfur or potassium bicarbonate',
                'Prune affected areas to increase air circulation',
                'Apply neem oil or other horticultural oils'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Ensure proper spacing between plants',
                'Avoid excess nitrogen fertilization',
                'Water at the base of plants, avoiding wetting the foliage'
            ]
        },
        {
            'is_disease_detected': True,
            'disease_name': 'Bacterial Leaf Spot',
            'disease_description': 'Bacterial leaf spot is caused by various species of bacteria. It typically appears as dark, water-soaked spots on leaves that may turn yellow, brown, or black.',
            'treatments': [
                'Apply copper-based bactericides',
                'Remove and destroy infected leaves',
                'Avoid working with plants when they are wet'
            ],
            'prevention': [
                'Use disease-free seeds and transplants',
                'Practice crop rotation',
                'Provide adequate spacing between plants',
                'Avoid overhead irrigation'
            ]
        },
        {
            'is_disease_detected': False,
            'disease_name': None,
            'disease_description': None,
            'treatments': [],
            'prevention': [
                'Regular monitoring for early disease detection',
                'Maintain proper plant nutrition',
                'Practice good sanitation in the garden or field',
                'Rotate crops to prevent disease buildup'
            ]
        }
    ]
    
    # For demonstration, randomly select a disease result
    # In a real implementation, this would be based on model prediction
    import random
    result = random.choice(diseases)
    
    return result

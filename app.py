import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os

from crop_recommendation import load_crop_recommendation_model, predict_crop
from disease_identification import identify_disease
from data_visualization import display_crop_distribution, display_feature_importance, display_parameter_ranges
from utils import load_crop_data, get_crop_info, preprocess_features

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Agri-Smart: Agricultural Advisory System",
    page_icon="🌱",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return load_crop_data()

# Load recommendation model
@st.cache_resource
def load_model():
    return load_crop_recommendation_model()

# Main function
def main():
    # Load data and model
    df = load_data()
    model = load_model()
    
    # Sidebar navigation
    st.sidebar.title("Agri-Smart 🌱")
    st.sidebar.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f33f.svg", width=100)
    
    # Add description
    st.sidebar.markdown("""
    **Agri-Smart** is your intelligent agricultural 
    advisory system that helps make data-driven 
    farming decisions.
    """)
    
    # Navigation with custom icons
    st.sidebar.markdown("### Navigation")
    
    navigation = st.sidebar.radio(
        "",
        ["🏠 Home", "🌾 Crop Recommendation", "🔍 Disease Identification", "📊 Data Insights"],
        format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
    )
    
    # Strip emoji from navigation for processing
    if "🏠" in navigation:
        navigation = "Home"
    elif "🌾" in navigation:
        navigation = "Crop Recommendation"  
    elif "🔍" in navigation:
        navigation = "Disease Identification"
    elif "📊" in navigation:
        navigation = "Data Insights"
    
    # Home Page
    if navigation == "Home":
        display_home()
    
    # Crop Recommendation Page
    elif navigation == "Crop Recommendation":
        display_crop_recommendation(df, model)
    
    # Disease Identification Page
    elif navigation == "Disease Identification":
        display_disease_identification()
    
    # Data Insights Page
    elif navigation == "Data Insights":
        display_data_insights(df)
    
    # Error handling for non-existent navigation
    else:
        st.error("Page not found. Please select a valid navigation option.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2023 Agri-Smart | Agricultural Advisory System")

# Home Page
def display_home():
    st.title("Welcome to Agri-Smart 🌱")
    
    st.markdown("""
    ### Your Agricultural Advisory Companion
    
    Agri-Smart helps farmers make informed decisions about crop selection and disease management.
    
    **Key Features:**
    - 🌾 **Crop Recommendation**: Get personalized crop suggestions based on soil parameters and environmental conditions
    - 🔍 **Disease Identification**: Identify crop diseases by uploading images
    - 📊 **Data Insights**: Explore agricultural data visualizations and analytics
    
    *Use the sidebar to navigate through different sections of the application.*
    """)
    
    # Display feature highlights in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌾 Crop Recommendation")
        st.markdown("""
        Input your soil parameters and environmental conditions to get personalized crop recommendations.
        
        - N, P, K values
        - pH level
        - Temperature
        - Humidity
        - Rainfall
        """)
        st.button("Try Crop Recommendation →", on_click=lambda: st.session_state.update({"navigation": "Crop Recommendation"}))
    
    with col2:
        st.markdown("### 🔍 Disease Identification")
        st.markdown("""
        Upload images of your crops to identify potential diseases.
        
        - Disease detection
        - Treatment suggestions
        - Preventive measures
        """)
        st.button("Try Disease Identification →", on_click=lambda: st.session_state.update({"navigation": "Disease Identification"}))
    
    with col3:
        st.markdown("### 📊 Data Insights")
        st.markdown("""
        Explore agricultural data visualizations and analytics.
        
        - Crop distribution
        - Parameter analysis
        - Feature importance
        """)
        st.button("Explore Data Insights →", on_click=lambda: st.session_state.update({"navigation": "Data Insights"}))
    
    # If button clicked, navigate to the corresponding page
    if "navigation" in st.session_state:
        navigation = st.session_state["navigation"]
        # Remove the navigation from session state to prevent infinite loop
        del st.session_state["navigation"]
        # Rerun the app with the new navigation
        st.session_state["navigation_radio"] = navigation
        st.rerun()

# Crop Recommendation Page
def display_crop_recommendation(df, model):
    st.title("Crop Recommendation 🌾")
    st.markdown("Enter soil parameters and environmental conditions to get personalized crop recommendations.")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    # Input form in first column
    with col1:
        st.subheader("Input Parameters")
        
        # Create three columns for inputs
        input_col1, input_col2, input_col3 = st.columns(3)
        
        with input_col1:
            n_value = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50)
            temp_value = st.number_input("Temperature (°C)", min_value=5.0, max_value=45.0, value=25.0, step=0.1)
            rainfall_value = st.number_input("Rainfall (mm)", min_value=20.0, max_value=500.0, value=100.0, step=0.1)
        
        with input_col2:
            p_value = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=50)
            humidity_value = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=50.0, step=0.1)
        
        with input_col3:
            k_value = st.number_input("Potassium (K)", min_value=0, max_value=150, value=50)
            ph_value = st.number_input("pH Value", min_value=3.0, max_value=10.0, value=6.5, step=0.1)
        
        # Create a dictionary of input features
        input_features = {
            'N': n_value,
            'P': p_value,
            'K': k_value,
            'Temperature': temp_value,
            'Humidity': humidity_value,
            'pH': ph_value,
            'Rainfall': rainfall_value
        }
        
        # Button to predict
        if st.button("Get Recommendation"):
            with st.spinner("Analyzing your inputs..."):
                # Preprocess features
                X = preprocess_features(input_features)
                
                # Predict crop
                predicted_crop, probabilities = predict_crop(model, X)
                
                # Get crop info
                crop_info = get_crop_info(df, predicted_crop)
                
                # Store prediction results in session state
                st.session_state['prediction_results'] = {
                    'predicted_crop': predicted_crop,
                    'probabilities': probabilities,
                    'crop_info': crop_info
                }
                
                time.sleep(1)  # Simulate processing time
    
    # Display results in second column
    with col2:
        st.subheader("Recommended Crop")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state['prediction_results']
            predicted_crop = results['predicted_crop']
            crop_info = results['crop_info']
            
            # Display the predicted crop with a large header
            st.markdown(f"### 🌱 {predicted_crop}")
            
            # Display crop information
            if crop_info:
                st.markdown("#### Crop Information")
                
                # Display disease information
                if crop_info.get('Disease_Prone') == 'Yes':
                    st.warning("⚠️ This crop is prone to diseases")
                else:
                    st.success("✅ This crop is generally disease-resistant")
                
                # Display possible diseases
                diseases = []
                for disease_type in ['Common_Disease(Fungal)', 'Common_Disease(Bacterial)', 'Common_Disease(Viral)']:
                    if crop_info.get(disease_type) != 'None' and not pd.isna(crop_info.get(disease_type)):
                        diseases.append(f"{disease_type.split('(')[1].split(')')[0]}: {crop_info.get(disease_type)}")
                
                if diseases:
                    st.markdown("#### Common Diseases")
                    for disease in diseases:
                        st.markdown(f"- {disease}")
                
                # Display water requirement
                if not pd.isna(crop_info.get('Water_Requirement')):
                    st.markdown(f"#### Water Requirement")
                    st.markdown(f"{crop_info.get('Water_Requirement', 'Unknown')} mm")
                
                # Display nutrient deficiency
                if crop_info.get('Nutrient_Deficiency') != 'None' and not pd.isna(crop_info.get('Nutrient_Deficiency')):
                    st.markdown("#### Nutrient Deficiency")
                    st.markdown(f"Common deficiency: {crop_info.get('Nutrient_Deficiency')}")
                
                # Display disease resistance score
                if not pd.isna(crop_info.get('Disease_Resistance_Score')):
                    st.markdown("#### Disease Resistance Score")
                    resistance_score = float(crop_info.get('Disease_Resistance_Score', 5.0))
                    st.progress(resistance_score / 10.0)
                    st.caption(f"Score: {resistance_score}/10")
            else:
                st.info("Detailed information about this crop is not available.")
        else:
            st.info("Enter soil parameters and click 'Get Recommendation' to see results.")

# Disease Identification Page
def display_disease_identification():
    st.title("Crop Disease Identification 🔍")
    st.markdown("Upload an image of your crop to identify potential diseases.")
    
    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded crop image", use_column_width=True)
        
        with col2:
            st.subheader("Identification Results")
            
            # Process the image when the button is clicked
            if st.button("Identify Disease"):
                with st.spinner("Analyzing image..."):
                    # Identify disease from image
                    disease_result = identify_disease(image)
                    
                    # Store results in session state
                    st.session_state['disease_results'] = disease_result
                    
                    time.sleep(2)  # Simulate processing time
            
            # Display results if available
            if 'disease_results' in st.session_state:
                result = st.session_state['disease_results']
                
                if result['is_disease_detected']:
                    st.error(f"Disease Detected: {result['disease_name']}")
                    
                    st.markdown("### Disease Information")
                    st.markdown(result['disease_description'])
                    
                    st.markdown("### Treatment")
                    for treatment in result['treatments']:
                        st.markdown(f"- {treatment}")
                    
                    st.markdown("### Prevention")
                    for prevention in result['prevention']:
                        st.markdown(f"- {prevention}")
                else:
                    st.success("No disease detected. Your crop appears healthy!")
                    st.markdown("### Recommendations")
                    st.markdown("Continue with your current crop management practices. Regular monitoring is always recommended.")
    else:
        # Display instructions when no image is uploaded
        st.info("Please upload an image to identify potential crop diseases.")
        
        # Display example information
        st.markdown("### Common Crop Diseases")
        
        # Create three columns for example diseases
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        
        with ex_col1:
            st.markdown("#### Leaf Spot")
            st.markdown("""
            - Small, brown or black spots on leaves
            - Can eventually lead to leaf drop
            - Common in many crops including tomato, potato, and cucumber
            """)
        
        with ex_col2:
            st.markdown("#### Powdery Mildew")
            st.markdown("""
            - White powdery substance on leaves
            - Reduces photosynthesis and yield
            - Common in grapes, strawberries, and cucurbits
            """)
        
        with ex_col3:
            st.markdown("#### Rust")
            st.markdown("""
            - Orange-brown pustules on leaves
            - Can cause severe defoliation
            - Common in wheat, beans, and sunflowers
            """)

# Data Insights Page
def display_data_insights(df):
    st.title("Agricultural Data Insights 📊")
    st.markdown("Explore visualizations and analytics about crops and their parameters.")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Crop Distribution", "Parameter Analysis", "Feature Importance"])
    
    # Tab 1: Crop Distribution
    with tab1:
        st.subheader("Crop Distribution Analysis")
        display_crop_distribution(df)
    
    # Tab 2: Parameter Analysis
    with tab2:
        st.subheader("Parameter Ranges by Crop")
        display_parameter_ranges(df)
    
    # Tab 3: Feature Importance
    with tab3:
        st.subheader("Feature Importance for Crop Selection")
        display_feature_importance()

if __name__ == "__main__":
    main()

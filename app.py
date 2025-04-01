import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
from streamlit_option_menu import option_menu

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
    
    # Apply custom CSS
    try:
        with open('assets/custom.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        pass  # Silently ignore if the file doesn't exist
    
    # Sidebar navigation with option_menu
    with st.sidebar:
        st.title("🌿 Agri-Smart")
        
        # Use option_menu for navigation
        navigation = option_menu(
            menu_title=None,
            options=["Dashboard", "Crop Recommendation", "Disease Detection", "Data Insights"],
            icons=["house-fill", "flower1", "bug-fill", "graph-up"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#a2ffa6"},
                "icon": {"color": "#099313", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#4EE556"},
                "nav-link-selected": {"background-color": "#4EE556", "color": "black", "font-weight": "normal"},
            }
        )
        
        # Add description below navigation
        st.markdown("""
        **Agri-Smart** is your intelligent agricultural 
        advisory system that helps make data-driven 
        farming decisions.
        """)
    
    # Map option_menu selections to our existing pages
    if navigation == "Dashboard":
        navigation = "Home"
    elif navigation == "Disease Detection":
        navigation = "Disease Identification"
    
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
    st.markdown("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: white; border-radius: 8px;">
            <p>Agri-Smart - Making farming smarter with data 🌱</p>
        </div>
    """, unsafe_allow_html=True)

# Dashboard/Home Page
def display_home():
    st.title("🌿 Agri-Smart Dashboard")
    
    # Create a banner with statistics
    st.markdown("""
    <div style="background-color: #4EE556; padding: 20px; border-radius: 8px; margin-bottom: 20px; color: black;">
        <h2 style="text-align: center; margin-bottom: 15px;">Your Smart Farming Assistant</h2>
        <p style="text-align: center;">
            Making data-driven decisions in agriculture simpler and more accessible
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three statistics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: #4CAF50; font-size: 40px;">35+</h1>
            <p>Supported Crops</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: #4CAF50; font-size: 40px;">90%</h1>
            <p>Recommendation Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: #4CAF50; font-size: 40px;">20+</h1>
            <p>Disease Patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("### Our Features")
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; height: 280px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #097809;">🌾 Crop Recommendation</h3>
            <p>Get personalized crop suggestions based on your soil parameters and environmental conditions.</p>
            <ul>
                <li>AI-powered recommendations</li>
                <li>Considers NPK, pH, climate</li>
                <li>Detailed crop information</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("Try Crop Recommendation →", key="dash_crop_rec", on_click=lambda: st.session_state.update({"navigation": "Crop Recommendation"}))
    
    with feature_col2:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; height: 280px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #097809;">🔍 Disease Detection</h3>
            <p>Upload crop images to identify diseases and get treatment recommendations.</p>
            <ul>
                <li>Visual disease recognition</li>
                <li>Customized treatment plans</li>
                <li>Preventive measures</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("Try Disease Detection →", key="dash_disease", on_click=lambda: st.session_state.update({"navigation": "Disease Identification"}))
    
    with feature_col3:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; height: 280px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #097809;">📊 Data Insights</h3>
            <p>Explore visualizations and analytics about crops and their growing conditions.</p>
            <ul>
                <li>Crop distribution data</li>
                <li>Parameter analysis</li>
                <li>Feature importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("Explore Data Insights →", key="dash_insights", on_click=lambda: st.session_state.update({"navigation": "Data Insights"}))
    
    # Supported crops section
    st.markdown("### Supported Crops")
    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <p>Our system provides recommendations and insights for a wide variety of crops including:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display crops in columns
    crop_groups = [
        ["Wheat", "Barley", "Oat", "Peas", "Potato", "Tomato", "Beet", "Cabbage"],
        ["Alfalfa", "Garlic", "Onion", "Cumin", "Coriander", "Fennel", "Linseed", "Sunflower"],
        ["Mustard", "Amaranth", "Cauliflower", "Paddy", "Maize", "Bajra", "Jowar", "Soybean"],
        ["Castor", "Cotton", "Sugarcane", "Turmeric", "Chilly", "Bitter Gourd", "Guar", "Okra"],
        ["Brinjal", "Turmeric", "Ragi"]
    ]
    
    for group in crop_groups:
        crop_cols = st.columns(len(group))
        for i, crop in enumerate(group):
            with crop_cols[i]:
                st.markdown(f"""
                <div style="background-color: #f0fff0; padding: 10px; border-radius: 5px; text-align: center; margin: 5px;">
                    <p style="margin: 0; color: #097809;"><strong>{crop}</strong></p>
                </div>
                """, unsafe_allow_html=True)
    
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

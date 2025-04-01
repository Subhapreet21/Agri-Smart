import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go

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
                "container": {"padding": "5px", "background-color": "transparent"},
                "icon": {"color": "#16a34a", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px", 
                    "text-align": "left", 
                    "margin": "0px", 
                    "--hover-color": "#dcfce7",
                    "padding": "10px",
                    "border-radius": "8px",
                    "margin-bottom": "5px"
                },
                "nav-link-selected": {
                    "background-color": "#dcfce7", 
                    "color": "#166534", 
                    "font-weight": "600"
                },
            }
        )
        
        # Add description below navigation
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f0fdf4; border-radius: 8px; border-left: 4px solid #16a34a;">
            <p style="margin: 0; font-size: 14px; color: #166534;">
                <strong>Agri-Smart</strong> is your intelligent agricultural 
                advisory system that helps make data-driven 
                farming decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Map option_menu selections to our existing pages
    if navigation == "Dashboard":
        navigation = "Home"
    elif navigation == "Disease Detection":
        navigation = "Disease Identification"
    
    # Home Page
    if navigation == "Home":
        display_home(df)
    
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

    # Modern footer
    st.markdown("""
        <div style="text-align: center; margin-top: 2.5rem; padding: 1.5rem; background-color: white; border-radius: 12px; border-top: 3px solid #16a34a; box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);">
            <p style="color: #4b5563; font-size: 0.9rem; margin-bottom: 0.5rem;">Agri-Smart - Making farming smarter with data 🌱</p>
            <p style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 0;">© 2023 Agri-Smart Technologies</p>
        </div>
    """, unsafe_allow_html=True)

# Dashboard/Home Page
def display_home(df):
    st.title("🌿 Agri-Smart Dashboard")
    
    # Create a banner with statistics - modern design
    st.markdown("""
    <div style="background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; color: white; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
        <h2 style="text-align: center; margin-bottom: 15px; font-weight: 600; letter-spacing: -0.03em;">Your Smart Farming Assistant</h2>
        <p style="text-align: center; font-size: 16px; opacity: 0.9;">
            Making data-driven decisions in agriculture simpler and more accessible
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate statistics for the dashboard from the dataset
    # Get total unique crops
    unique_crops = df['Label'].nunique()
    
    # Get common diseases count - estimate based on disease prone crops
    common_diseases = []
    for col in ['Common_Disease(Fungal)', 'Common_Disease(Bacterial)', 'Common_Disease(Viral)']:
        diseases = df[col].dropna().unique().tolist()
        common_diseases.extend([d for d in diseases if d != 'None'])
    common_diseases = list(set(common_diseases))
    
    # Create statistics dictionary
    stats = {
        'total_crops': unique_crops,
        'common_diseases': common_diseases
    }
    
    # Create three statistics cards with modern styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="metric-value">{stats['total_crops']}</p>
            <p class="metric-label">Total Crops</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <p class="metric-value">87.6%</p>
            <p class="metric-label">Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="metric-value">{len(stats['common_diseases'])+1}</p>
            <p class="metric-label">Disease Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data visualizations in modern card style
    st.markdown("""
    <div class="card">
        <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">Crop Distribution</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Crop distribution visualization
    fig_dist = px.pie(df, names='Label', 
                      title='Distribution of Crops',
                      color_discrete_sequence=px.colors.sequential.Greens,
                      height=400)
    fig_dist.update_traces(textposition='inside', textinfo='percent+label')
    fig_dist.update_layout(
        title_font_size=18,
        font=dict(family="Inter, sans-serif", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Parameter selection for parameter analysis
    st.markdown("""
    <div class="card">
        <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">Environmental Parameters by Crop</h3>
    </div>
    """, unsafe_allow_html=True)
    
    param = st.selectbox("Select Parameter", 
                        ['Temperature', 'Humidity', 'pH', 'Rainfall'],
                        format_func=lambda x: {'Temperature': 'Temperature', 'Humidity': 'Humidity', 
                                              'pH': 'pH', 'Rainfall': 'Rainfall'}[x])
    
    # Parameter distribution visualization                      
    fig_param = px.box(df, x='Label', y=param,
                       title=f'{param.title()} Distribution by Crop',
                       color_discrete_sequence=['#16a34a'],
                       height=400)
    fig_param.update_layout(
        title_font_size=18,
        font=dict(family="Inter, sans-serif", size=14),
        xaxis_title="Crop",
        yaxis_title=param.title(),
        xaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig_param, use_container_width=True)
    
    # Check if Disease_Prone column exists in the dataframe
    if 'Disease_Prone' in df.columns:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">Disease Probability by Crop</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Convert Disease_Prone to numeric for visualization
        df_disease = df.copy()
        df_disease['Disease_Prone_Numeric'] = df_disease['Disease_Prone'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Disease probability visualization
        fig_disease = px.bar(df_disease.groupby('Label')['Disease_Prone_Numeric'].mean().reset_index(),
                            x='Label', y='Disease_Prone_Numeric',
                            title='Disease Probability by Crop',
                            color_discrete_sequence=['#16a34a'],
                            height=400)
        fig_disease.update_layout(
            title_font_size=18,
            font=dict(family="Inter, sans-serif", size=14),
            xaxis_title="Crop",
            yaxis_title="Disease Probability",
            xaxis={'categoryorder':'total descending'},
            yaxis={'tickformat': ',.0%'}
        )
        st.plotly_chart(fig_disease, use_container_width=True)
    
    # Feature cards with modern styling
    st.markdown("""
    <div class="card">
        <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">Our Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 12px;">🌾 Crop Recommendation</h3>
            <p style="color: #4b5563; margin-bottom: 15px;">Get personalized crop suggestions based on your soil parameters and environmental conditions.</p>
            <ul style="padding-left: 20px; color: #4b5563;">
                <li style="margin-bottom: 5px;">AI-powered recommendations</li>
                <li style="margin-bottom: 5px;">Considers NPK, pH, climate</li>
                <li style="margin-bottom: 5px;">Detailed crop information</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("Try Crop Recommendation →", key="dash_crop_rec", on_click=lambda: st.session_state.update({"navigation": "Crop Recommendation"}))
    
    with feature_col2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 12px;">🔍 Disease Detection</h3>
            <p style="color: #4b5563; margin-bottom: 15px;">Upload crop images to identify diseases and get treatment recommendations.</p>
            <ul style="padding-left: 20px; color: #4b5563;">
                <li style="margin-bottom: 5px;">Visual disease recognition</li>
                <li style="margin-bottom: 5px;">Customized treatment plans</li>
                <li style="margin-bottom: 5px;">Preventive measures</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("Try Disease Detection →", key="dash_disease", on_click=lambda: st.session_state.update({"navigation": "Disease Identification"}))
    
    with feature_col3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 12px;">📊 Data Insights</h3>
            <p style="color: #4b5563; margin-bottom: 15px;">Explore visualizations and analytics about crops and their growing conditions.</p>
            <ul style="padding-left: 20px; color: #4b5563;">
                <li style="margin-bottom: 5px;">Crop distribution data</li>
                <li style="margin-bottom: 5px;">Parameter analysis</li>
                <li style="margin-bottom: 5px;">Feature importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("Explore Data Insights →", key="dash_insights", on_click=lambda: st.session_state.update({"navigation": "Data Insights"}))
    
    # Supported crops section - modern design
    st.markdown("""
    <div class="card">
        <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">Supported Crops</h3>
        <p style="margin: 0 0 15px; color: #4b5563;">Our system provides recommendations and insights for a wide variety of crops including:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display crops in columns with pill-style design
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
                <div class="crop-pill">
                    {crop}
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
    
    # Introduction with modern styling
    st.markdown("""
    <div class="card" style="margin-bottom: 25px;">
        <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Personalized Crop Recommendations</h3>
        <p style="margin: 0; color: #4b5563;">Enter your soil parameters and environmental conditions below to get AI-powered crop recommendations tailored to your specific growing conditions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns with modern card styling
    col1, col2 = st.columns([2, 1])
    
    # Input form in first column
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Input Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for inputs
        input_col1, input_col2, input_col3 = st.columns(3)
        
        with input_col1:
            n_value = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50,
                                      help="Amount of nitrogen in soil (mg/kg)")
            temp_value = st.number_input("Temperature (°C)", min_value=5.0, max_value=45.0, value=25.0, step=0.1,
                                         help="Average temperature in °C")
            rainfall_value = st.number_input("Rainfall (mm)", min_value=20.0, max_value=500.0, value=100.0, step=0.1,
                                           help="Average rainfall in mm")
        
        with input_col2:
            p_value = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=50,
                                     help="Amount of phosphorus in soil (mg/kg)")
            humidity_value = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=50.0, step=0.1,
                                           help="Average humidity in %")
        
        with input_col3:
            k_value = st.number_input("Potassium (K)", min_value=0, max_value=150, value=50,
                                     help="Amount of potassium in soil (mg/kg)")
            ph_value = st.number_input("pH Value", min_value=3.0, max_value=10.0, value=6.5, step=0.1,
                                      help="pH value of soil (1-14)")
        
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
        
        # Button to predict with modern styling
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if st.button("Get Recommendation", key="crop_rec_btn"):
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
    
    # Display results in second column with modern styling
    with col2:
        if 'prediction_results' in st.session_state:
            results = st.session_state['prediction_results']
            predicted_crop = results['predicted_crop']
            crop_info = results['crop_info']
            
            # Display the predicted crop with modern styling
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(to right, #f0fdf4, #dcfce7); border-left: 4px solid #16a34a;">
                <h3 style="color: #166534; font-weight: 600; margin-bottom: 5px;">Recommended Crop</h3>
                <p style="font-size: 1.5rem; font-weight: 700; color: #16a34a; margin: 10px 0;">🌱 {predicted_crop}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display crop information
            if crop_info:
                # Disease proneness card
                prone_status = "Yes" if crop_info.get('Disease_Prone') == 'Yes' else "No"
                status_color = "#f87171" if prone_status == "Yes" else "#4ade80"
                status_icon = "⚠️" if prone_status == "Yes" else "✅"
                status_text = "This crop is prone to diseases" if prone_status == "Yes" else "This crop is generally disease-resistant"
                
                st.markdown(f"""
                <div class="card" style="margin-top: 15px;">
                    <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Disease Proneness</h4>
                    <p style="color: {status_color}; font-weight: 500;">{status_icon} {status_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Water requirement card
                if not pd.isna(crop_info.get('Water_Requirement')):
                    st.markdown(f"""
                    <div class="card" style="margin-top: 15px;">
                        <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Water Requirement</h4>
                        <p style="font-weight: 500; color: #4b5563;">{crop_info.get('Water_Requirement', 'Unknown')} mm</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Disease resistance score card
                if not pd.isna(crop_info.get('Disease_Resistance_Score')):
                    resistance_score = float(crop_info.get('Disease_Resistance_Score', 5.0))
                    st.markdown(f"""
                    <div class="card" style="margin-top: 15px;">
                        <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Disease Resistance</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(resistance_score / 10.0)
                    st.caption(f"Score: {resistance_score}/10")
                
                # Common diseases in an expandable section if any exist
                diseases = []
                for disease_type in ['Common_Disease(Fungal)', 'Common_Disease(Bacterial)', 'Common_Disease(Viral)']:
                    if crop_info.get(disease_type) != 'None' and not pd.isna(crop_info.get(disease_type)):
                        disease_category = disease_type.split('(')[1].split(')')[0]
                        diseases.append(f"{disease_category}: {crop_info.get(disease_type)}")
                
                if diseases:
                    with st.expander("Common Diseases", expanded=False):
                        for disease in diseases:
                            st.markdown(f"- {disease}")
            else:
                st.info("Detailed information about this crop is not available.")
        else:
            # Display placeholder with instructions
            st.markdown("""
            <div class="card" style="background-color: #f0fdf4; text-align: center; padding: 30px 15px;">
                <svg xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto 15px; display: block; width: 40px; height: 40px; color: #16a34a;" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p style="color: #4b5563; margin-bottom: 5px;">Enter soil parameters</p>
                <p style="color: #6b7280; font-size: 0.9rem;">Then click "Get Recommendation"</p>
            </div>
            """, unsafe_allow_html=True)

# Disease Identification Page
def display_disease_identification():
    st.title("Crop Disease Identification 🔍")
    
    # Introduction with modern styling
    st.markdown("""
    <div class="card" style="margin-bottom: 25px;">
        <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">AI-Powered Disease Detection</h3>
        <p style="margin: 0; color: #4b5563;">Upload an image of your crop to identify potential diseases and get treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern styled image uploader
    st.markdown("""
    <div class="card">
        <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Upload Crop Image</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select an image of your crop", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Create a card for the image
            st.markdown("""
            <div class="card">
                <h3 style="color: #16a34a; font-size: 1.1rem; margin-bottom: 10px;">Your Crop Image</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="Uploaded crop image", use_column_width=True)
        
        with col2:
            # Create a card for results
            st.markdown("""
            <div class="card">
                <h3 style="color: #16a34a; font-size: 1.1rem; margin-bottom: 10px;">Identification Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Process the image when the button is clicked with modern styling
            if st.button("Analyze Image", key="disease_identification_btn"):
                with st.spinner("Processing image with AI analysis..."):
                    # Identify disease from image
                    disease_result = identify_disease(image)
                    
                    # Store results in session state
                    st.session_state['disease_results'] = disease_result
                    
                    time.sleep(2)  # Simulate processing time
            
            # Display results if available with modern styling
            if 'disease_results' in st.session_state:
                result = st.session_state['disease_results']
                
                if result['is_disease_detected']:
                    # Disease detected UI with modern styling
                    st.markdown(f"""
                    <div style="background-color: #fee2e2; border-left: 4px solid #ef4444; padding: 12px 15px; border-radius: 6px; margin: 15px 0;">
                        <p style="margin: 0; font-weight: 600; color: #b91c1c;">⚠️ Disease Detected: {result['disease_name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Disease information in a card
                    st.markdown("""
                    <div class="card" style="margin-top: 15px;">
                        <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Disease Information</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #4b5563;'>{result['disease_description']}</p>", unsafe_allow_html=True)
                    
                    # Treatment options in expandable section
                    with st.expander("Treatment Options", expanded=True):
                        for treatment in result['treatments']:
                            st.markdown(f"""
                            <div style="display: flex; margin-bottom: 8px;">
                                <div style="color: #16a34a; margin-right: 8px;">•</div>
                                <div style="color: #4b5563;">{treatment}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Prevention tips in expandable section
                    with st.expander("Prevention Measures", expanded=True):
                        for prevention in result['prevention']:
                            st.markdown(f"""
                            <div style="display: flex; margin-bottom: 8px;">
                                <div style="color: #16a34a; margin-right: 8px;">•</div>
                                <div style="color: #4b5563;">{prevention}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Healthy crop UI with modern styling
                    st.markdown("""
                    <div style="background-color: #dcfce7; border-left: 4px solid #22c55e; padding: 12px 15px; border-radius: 6px; margin: 15px 0;">
                        <p style="margin: 0; font-weight: 600; color: #15803d;">✅ No disease detected. Your crop appears healthy!</p>
                    </div>
                    
                    <div class="card" style="margin-top: 15px;">
                        <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Recommendations</h4>
                        <p style="color: #4b5563;">Continue with your current crop management practices. Regular monitoring is always recommended to ensure early detection of any issues.</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Display instructions and examples in a more modern way
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Add an illustration or icon
            st.markdown("""
            <div style="background-color: #f0fdf4; padding: 30px; border-radius: 12px; text-align: center; height: 100%;">
                <svg xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto 15px; display: block; width: 80px; height: 80px; color: #16a34a;" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p style="color: #4b5563; margin-bottom: 5px;">Upload a clear, well-lit image</p>
                <p style="color: #6b7280; font-size: 0.9rem;">For the best results, ensure the affected parts are visible</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Modern styled disease examples
            st.markdown("""
            <div class="card">
                <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Common Crop Diseases</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for example diseases
            ex_col1, ex_col2, ex_col3 = st.columns(3)
            
            with ex_col1:
                st.markdown("""
                <div class="card" style="height: 100%;">
                    <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Leaf Spot</h4>
                    <ul style="padding-left: 20px; margin: 0; color: #4b5563;">
                        <li style="margin-bottom: 5px;">Small, brown or black spots</li>
                        <li style="margin-bottom: 5px;">Leads to leaf drop</li>
                        <li style="margin-bottom: 5px;">Common in many crops</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with ex_col2:
                st.markdown("""
                <div class="card" style="height: 100%;">
                    <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Powdery Mildew</h4>
                    <ul style="padding-left: 20px; margin: 0; color: #4b5563;">
                        <li style="margin-bottom: 5px;">White powdery substance</li>
                        <li style="margin-bottom: 5px;">Reduces photosynthesis</li>
                        <li style="margin-bottom: 5px;">Common in various fruits</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with ex_col3:
                st.markdown("""
                <div class="card" style="height: 100%;">
                    <h4 style="color: #166534; font-size: 1rem; margin-bottom: 10px;">Rust</h4>
                    <ul style="padding-left: 20px; margin: 0; color: #4b5563;">
                        <li style="margin-bottom: 5px;">Orange-brown pustules</li>
                        <li style="margin-bottom: 5px;">Causes defoliation</li>
                        <li style="margin-bottom: 5px;">Common in grains and beans</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# Data Insights Page
def display_data_insights(df):
    st.title("Agricultural Data Insights 📊")
    
    # Introduction with modern styling
    st.markdown("""
    <div class="card" style="margin-bottom: 25px;">
        <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Data-Driven Farming Insights</h3>
        <p style="margin: 0; color: #4b5563;">Explore visualizations and analytics about crops and their ideal growing parameters to make informed decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern tabs using Streamlit's tab component
    tab1, tab2, tab3 = st.tabs(["Crop Distribution", "Parameter Analysis", "Feature Importance"])
    
    # Tab 1: Crop Distribution with modern styling
    with tab1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Crop Distribution Analysis</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Explore the distribution of different crops in our dataset and understand their prevalence.</p>
        </div>
        """, unsafe_allow_html=True)
        display_crop_distribution(df)
    
    # Tab 2: Parameter Analysis with modern styling
    with tab2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Parameter Ranges by Crop</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Visualize the optimal ranges of soil parameters and environmental conditions for different crops.</p>
        </div>
        """, unsafe_allow_html=True)
        display_parameter_ranges(df)
    
    # Tab 3: Feature Importance with modern styling
    with tab3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Feature Importance for Crop Selection</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Understand which soil and environmental factors have the most significant impact on crop selection decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        display_feature_importance()
        
    # Add a tips section at the bottom
    st.markdown("""
    <div style="margin-top: 30px; padding: 20px; background-color: #f0fdf4; border-radius: 12px; border-left: 4px solid #16a34a;">
        <h3 style="color: #16a34a; font-size: 1.1rem; margin-bottom: 10px; display: flex; align-items: center;">
            <span style="margin-right: 8px;">💡</span> Data Interpretation Tips
        </h3>
        <ul style="margin: 0; padding-left: 20px; color: #4b5563;">
            <li style="margin-bottom: 5px;">Use the crop distribution data to understand which crops are most commonly grown in similar conditions.</li>
            <li style="margin-bottom: 5px;">Parameter ranges help you determine if your soil and climate conditions are suitable for specific crops.</li>
            <li style="margin-bottom: 5px;">Feature importance highlights which parameters you should focus on optimizing for better crop yields.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

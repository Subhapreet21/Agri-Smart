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

# Custom color palettes for a consistent theme
GREEN_PALETTE = ["#166534", "#15803d", "#16a34a", "#22c55e", "#4ade80", "#86efac", "#bbf7d0"]
BLUE_PALETTE = ["#0c4a6e", "#0369a1", "#0284c7", "#0ea5e9", "#38bdf8", "#7dd3fc", "#bae6fd"]
SOIL_PALETTE = ["#854d0e", "#a16207", "#ca8a04", "#eab308", "#facc15", "#fde047"]
DISEASE_PALETTE = ["#991b1b", "#b91c1c", "#dc2626", "#ef4444", "#f87171", "#fca5a5"]
CROP_PALETTE = px.colors.qualitative.Prism

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
load_css('assets/custom.css')

# Load data
@st.cache_data
def load_data():
    df = load_crop_data()
    # Add Season column for Rabi/Kharif classification
    from utils import add_season_to_dataframe
    df = add_season_to_dataframe(df)
    return df

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
            options=["Dashboard", "Crop Recommendation", "Disease Detection", "Data Insights", "Rabi Crops", "Kharif Crops"],
            icons=["house-fill", "flower1", "bug-fill", "graph-up", "sun", "cloud-rain"],
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
    
    # Rabi Crops Page
    elif navigation == "Rabi Crops":
        display_rabi_crops(df)
    
    # Kharif Crops Page
    elif navigation == "Kharif Crops":
        display_kharif_crops(df)
    
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
                      color_discrete_sequence=px.colors.qualitative.Bold,
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
                       color_discrete_sequence=GREEN_PALETTE[1:3],
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
                            color_discrete_sequence=DISEASE_PALETTE,
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
        
        # Advanced parameters expander
        with st.expander("Advanced Parameters", expanded=False):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                salinity_value = st.number_input("Salinity (dS/m)", min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                                           help="Soil salinity in dS/m")
            
            with adv_col2:
                water_req_value = st.number_input("Water Requirement (mm)", min_value=50.0, max_value=800.0, value=400.0, step=10.0,
                                           help="Water requirement in mm")
            
            with adv_col3:
                disease_resistance = st.number_input("Disease Resistance Score", min_value=1.0, max_value=10.0, value=5.0, step=0.1,
                                           help="Disease resistance score (1-10)")
                
        # Create a dictionary of input features
        input_features = {
            'N': n_value,
            'P': p_value,
            'K': k_value,
            'Temperature': temp_value,
            'Humidity': humidity_value,
            'pH': ph_value,
            'Rainfall': rainfall_value,
            'Salinity_dS_m': salinity_value,
            'Water_Requirement': water_req_value,
            'Disease_Resistance_Score': disease_resistance
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
            
            # Display the top 3 recommended crops with modern styling
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(to right, #f0fdf4, #dcfce7); border-left: 4px solid #16a34a;">
                <h3 style="color: #166534; font-weight: 600; margin-bottom: 5px;">Top 3 Recommended Crops</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Get top 3 crops
            top_crops = list(results['probabilities'].items())[:3]
            
            # Create medal emoji for rankings
            medals = ["🥇", "🥈", "🥉"]
            
            # Display top 3 crops
            for i, (crop, prob) in enumerate(top_crops):
                st.markdown(f"""
                <div class="card" style="margin-top: 10px; background: {'#f0fdf4' if i == 0 else 'white'}; border-left: {4 if i == 0 else 2}px solid {'#16a34a' if i == 0 else '#84cc16'};">
                    <h4 style="color: #166534; margin-bottom: 5px; font-size: {'1.3rem' if i == 0 else '1.1rem'};">
                        {medals[i]} {crop}
                    </h4>
                    <p style="color: #4b5563; margin: 0;">Confidence: {prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Set the predicted crop to the top crop for other sections
            predicted_crop = top_crops[0][0]
            
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
    tab1, tab2, tab3, tab4 = st.tabs(["Crop Distribution", "Parameter Analysis", "Feature Importance", "Parameter Correlations"])
    
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
        
    # Tab 4: Parameter Correlations (based on reference code)
    with tab4:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Parameter Correlations & Relationships</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Explore how different soil and environmental parameters relate to each other and influence crop suitability.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter Correlation Matrix
        st.subheader("Parameter Correlations")
        
        # Select features for correlation analysis
        corr_features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        # Calculate correlation matrix
        corr_matrix = df[corr_features].corr().round(2)
        
        # Display correlation heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_features,
            y=corr_features,
            colorscale='Blues'
        ))
        
        fig_corr.update_layout(
            title='Parameter Correlation Matrix', 
            height=400,
            margin=dict(l=50, r=50, t=50, b=30)
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        <div style="margin-top: 10px; padding: 10px; background-color: #f0fdf4; border-radius: 5px;">
            <p style="margin: 0; color: #16a34a; font-weight: 500;">💡 Values closer to 1 (darker blue) indicate strong positive correlation, meaning these parameters tend to increase together.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter Relationships scatter plot
        st.subheader("Parameter Relationships")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_x_param = st.selectbox(
                "Select X-axis Parameter",
                corr_features
            )
        
        with col2:
            selected_y_param = st.selectbox(
                "Select Y-axis Parameter",
                corr_features,
                index=1
            )
        
        if selected_x_param and selected_y_param:
            fig_scatter = px.scatter(
                df,
                x=selected_x_param,
                y=selected_y_param,
                color='Label',
                title=f'{selected_x_param} vs {selected_y_param}',
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=400
            )
            
            fig_scatter.update_layout(
                xaxis_title=selected_x_param,
                yaxis_title=selected_y_param,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Disease Proneness Analysis
        st.subheader("Disease Proneness Analysis")
        
        # Only proceed if 'Disease_Prone' column exists
        if 'Disease_Prone' in df.columns:
            param = st.selectbox(
                "Select Environmental Parameter",
                ['Temperature', 'Humidity', 'pH', 'Rainfall']
            )
            
            fig_disease = px.histogram(
                df,
                x=param,
                color='Disease_Prone',
                title=f'Disease Proneness by {param}',
                color_discrete_sequence=['#81C784', '#FFC107'],
                barmode='group',
                height=400
            )
            
            fig_disease.update_layout(
                xaxis_title=param,
                yaxis_title='Count',
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            st.plotly_chart(fig_disease, use_container_width=True)
            
            st.markdown("""
            <div style="margin-top: 10px; padding: 10px; background-color: #f0fdf4; border-radius: 5px;">
                <p style="margin: 0; color: #16a34a; font-weight: 500;">💡 This chart shows how environmental factors may influence disease susceptibility across different crops.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Disease proneness data is not available in the current dataset.")
        
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

# Rabi Crops Page
def display_rabi_crops(df):
    st.title("Rabi Crops Analysis 🌾")
    
    # Introduction with modern styling
    st.markdown("""
    <div class="card" style="margin-bottom: 25px;">
        <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Rabi Season Crop Insights</h3>
        <p style="margin: 0; color: #4b5563;">Rabi crops are sown in winter and harvested in spring. This page provides specific analysis and insights for Rabi crops.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter dataframe for Rabi crops only
    rabi_df = df[df['Season'] == 'Rabi']
    
    # Display error if no Rabi crops found
    if len(rabi_df) == 0:
        st.error("No Rabi crops found in the dataset.")
        return
    
    # Create two columns for statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Total Rabi crops count
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="metric-value">{rabi_df['Label'].nunique()}</p>
            <p class="metric-label">Rabi Crop Varieties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Disease prone percentage
        if 'Disease_Prone' in rabi_df.columns:
            disease_prone_pct = rabi_df[rabi_df['Disease_Prone'] == 'Yes'].shape[0] / rabi_df.shape[0] * 100
            st.markdown(f"""
            <div class="dashboard-card">
                <p class="metric-value">{disease_prone_pct:.1f}%</p>
                <p class="metric-label">Disease Prone</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Rabi crop distribution
    st.markdown("""
    <div class="card">
        <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">Rabi Crop Distribution</h3>
    </div>
    """, unsafe_allow_html=True)
    
    fig_dist = px.pie(rabi_df, names='Label', 
                     title='Distribution of Rabi Crops',
                     color_discrete_sequence=px.colors.qualitative.Vivid,
                     height=400)
    fig_dist.update_traces(textposition='inside', textinfo='percent+label')
    fig_dist.update_layout(
        title_font_size=18,
        font=dict(family="Inter, sans-serif", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Modern tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Parameter Analysis", "Disease Analysis", "Recommendations"])
    
    # Tab 1: Parameter Analysis
    with tab1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Parameter Requirements for Rabi Crops</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Compare soil and environmental parameter requirements across different Rabi crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter selection
        param = st.selectbox("Select Parameter", 
                           ['Temperature', 'Humidity', 'pH', 'Rainfall', 'N', 'P', 'K'],
                           key="rabi_param",
                           format_func=lambda x: {'Temperature': 'Temperature (°C)', 
                                                'Humidity': 'Humidity (%)', 
                                                'pH': 'pH Level', 
                                                'Rainfall': 'Rainfall (mm)',
                                                'N': 'Nitrogen (N)',
                                                'P': 'Phosphorus (P)',
                                                'K': 'Potassium (K)'}[x])
        
        # Parameter comparison chart
        fig_param = px.box(rabi_df, x='Label', y=param,
                          title=f'{param} Requirements for Rabi Crops',
                          color='Label',
                          height=450)
        fig_param.update_layout(
            xaxis_title="Crop",
            yaxis_title=param,
            showlegend=False
        )
        st.plotly_chart(fig_param, use_container_width=True)
        
        # Parameter summary table
        st.subheader("Parameter Summary")
        
        # Calculate statistics for each crop and parameter
        param_summary = rabi_df.groupby('Label')[param].agg(['min', 'max', 'mean', 'median']).reset_index()
        param_summary.columns = ['Crop', 'Minimum', 'Maximum', 'Average', 'Median']
        param_summary = param_summary.round(2)
        
        # Display summary table with styling
        st.dataframe(param_summary, use_container_width=True)
    
    # Tab 2: Disease Analysis
    with tab2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Disease Analysis for Rabi Crops</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Understand disease patterns and susceptibility among Rabi crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if disease columns exist
        disease_cols = [col for col in rabi_df.columns if 'Disease' in col]
        
        if len(disease_cols) > 0:
            # Disease prone distribution
            if 'Disease_Prone' in disease_cols:
                # Count occurrences
                disease_prone_counts = rabi_df['Disease_Prone'].value_counts().reset_index()
                disease_prone_counts.columns = ['Status', 'Count']
                
                # Create pie chart
                fig_disease = px.pie(disease_prone_counts, names='Status', values='Count',
                                    title='Disease Proneness of Rabi Crops',
                                    color_discrete_sequence=['#10b981', '#f43f5e'],
                                    height=350)
                fig_disease.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_disease, use_container_width=True)
            
            # Common diseases
            disease_types = ['Common_Disease(Fungal)', 'Common_Disease(Bacterial)', 'Common_Disease(Viral)']
            available_types = [col for col in disease_types if col in rabi_df.columns]
            
            if available_types:
                st.subheader("Common Diseases in Rabi Crops")
                
                for disease_type in available_types:
                    # Skip if column doesn't exist
                    if disease_type not in rabi_df.columns:
                        continue
                    
                    # Get non-None diseases
                    disease_data = rabi_df[rabi_df[disease_type] != 'None']
                    
                    if len(disease_data) > 0:
                        # Format disease type for display
                        display_type = disease_type.replace('Common_Disease(', '').replace(')', '')
                        
                        st.markdown(f"##### {display_type} Diseases")
                        
                        # Count occurrences of each disease
                        disease_counts = disease_data[disease_type].value_counts().reset_index()
                        disease_counts.columns = ['Disease', 'Count']
                        
                        # Create horizontal bar chart
                        fig_disease_count = px.bar(disease_counts, 
                                                 y='Disease', 
                                                 x='Count',
                                                 title=f'Common {display_type} Diseases in Rabi Crops',
                                                 color_discrete_sequence=['#16a34a'],
                                                 height=max(250, len(disease_counts) * 50))
                        fig_disease_count.update_layout(xaxis_title="Number of Crops Affected")
                        st.plotly_chart(fig_disease_count, use_container_width=True)
                        
                    else:
                        display_type = disease_type.replace('Common_Disease(', '').replace(')', '')
                        st.info(f"No common {display_type.lower()} diseases found for Rabi crops in the dataset.")
        else:
            st.info("No disease information available in the dataset.")
    
    # Tab 3: Recommendations
    with tab3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Recommendations for Rabi Crops</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Best practices and recommendations for growing Rabi crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # General recommendations for Rabi crops
        st.markdown("""
        <div class="card">
            <h4 style="color: #16a34a; font-size: 1.1rem; margin-bottom: 12px;">General Best Practices</h4>
            <ul style="padding-left: 20px; color: #4b5563;">
                <li style="margin-bottom: 8px;">Prepare the land in September-October, before the soil moisture depletes.</li>
                <li style="margin-bottom: 8px;">Apply adequate organic matter to the soil during land preparation.</li>
                <li style="margin-bottom: 8px;">Ensure proper seed treatment before sowing to prevent diseases.</li>
                <li style="margin-bottom: 8px;">Maintain optimum soil moisture during the early growth stages.</li>
                <li style="margin-bottom: 8px;">Apply fertilizers as per crop requirements and soil test recommendations.</li>
                <li style="margin-bottom: 8px;">Monitor for pest and disease outbreaks regularly during the growing season.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Crop-specific recommendations
        st.markdown("""
        <div class="card" style="margin-top: 20px;">
            <h4 style="color: #16a34a; font-size: 1.1rem; margin-bottom: 12px;">Select a Crop for Specific Recommendations</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Dropdown to select a crop
        selected_rabi_crop = st.selectbox(
            "Choose a Rabi crop",
            options=rabi_df['Label'].unique(),
            key="rabi_crop_select"
        )
        
        # Display crop-specific info
        if selected_rabi_crop:
            crop_data = rabi_df[rabi_df['Label'] == selected_rabi_crop].iloc[0]
            
            # Header for requirements
            st.subheader(f"{selected_rabi_crop} Requirements")
            
            # First row of metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div style="background-color: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">Temperature</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #16a34a;">{crop_data['Temperature']}°C</p>
                </div>""", unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div style="background-color: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">Humidity</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #16a34a;">{crop_data['Humidity']}%</p>
                </div>""", unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div style="background-color: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">pH</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #16a34a;">{crop_data['pH']}</p>
                </div>""", unsafe_allow_html=True)
            
            # Second row of metrics
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div style="background-color: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">N</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #16a34a;">{crop_data['N']} mg/kg</p>
                </div>""", unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div style="background-color: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">P</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #16a34a;">{crop_data['P']} mg/kg</p>
                </div>""", unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div style="background-color: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">K</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #16a34a;">{crop_data['K']} mg/kg</p>
                </div>""", unsafe_allow_html=True)
            
            # Show additional parameters if they exist
            additional_params = ['Salinity_dS_m', 'Water_Requirement', 'Disease_Resistance_Score']
            available_params = [param for param in additional_params if param in crop_data]
            
            if available_params:
                st.subheader("Additional Parameters")
                
                cols = st.columns(min(3, len(available_params)))
                
                for i, param in enumerate(available_params):
                    display_name = param.replace('_', ' ')
                    
                    # Add units based on parameter
                    unit = ""
                    if param == 'Salinity_dS_m':
                        unit = "dS/m"
                        display_name = "Salinity"
                    elif param == 'Water_Requirement':
                        unit = "mm"
                    elif param == 'Disease_Resistance_Score':
                        unit = "/10"
                        display_name = "Disease Resistance"
                    
                    value = crop_data[param]
                    
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                            <div style="background-color: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                                <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">{display_name}</p>
                                <p style="margin: 0; font-weight: 600; color: #16a34a;">{value} {unit}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Display disease information if available
            disease_cols = [col for col in crop_data.index if 'Disease' in col and col != 'Disease_Resistance_Score']
            
            if disease_cols:
                diseases = []
                
                for col in disease_cols:
                    if crop_data[col] != 'None' and str(crop_data[col]) != 'nan':
                        disease_type = col.replace('Common_Disease(', '').replace(')', '')
                        diseases.append(f"{crop_data[col]} ({disease_type})")
                
                if diseases:
                    st.subheader("Disease Information")
                    
                    st.markdown(f"""
                    <div style="margin-top: 10px; background-color: #fff8f1; border-left: 4px solid #f97316; padding: 15px; border-radius: 4px;">
                        <p style="margin: 0 0 10px; color: #4b5563;">Common diseases that affect {selected_rabi_crop}:</p>
                    """, unsafe_allow_html=True)
                    
                    # Display diseases as bullet points
                    for disease in diseases:
                        st.markdown(f"• {disease}", unsafe_allow_html=False)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.subheader("Disease Information")
                    st.success(f"✓ No common diseases recorded for {selected_rabi_crop} in our database.")

# Kharif Crops Page
def display_kharif_crops(df):
    st.title("Kharif Crops Analysis ☔")
    
    # Introduction with modern styling
    st.markdown("""
    <div class="card" style="margin-bottom: 25px;">
        <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Kharif Season Crop Insights</h3>
        <p style="margin: 0; color: #4b5563;">Kharif crops are sown at the beginning of the monsoon season and harvested at the end of the monsoon season. This page provides specific analysis and insights for Kharif crops.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter dataframe for Kharif crops only
    kharif_df = df[df['Season'] == 'Kharif']
    
    # Display error if no Kharif crops found
    if len(kharif_df) == 0:
        st.error("No Kharif crops found in the dataset.")
        return
    
    # Create two columns for statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Total Kharif crops count
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="metric-value">{kharif_df['Label'].nunique()}</p>
            <p class="metric-label">Kharif Crop Varieties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Disease prone percentage
        if 'Disease_Prone' in kharif_df.columns:
            disease_prone_pct = kharif_df[kharif_df['Disease_Prone'] == 'Yes'].shape[0] / kharif_df.shape[0] * 100
            st.markdown(f"""
            <div class="dashboard-card">
                <p class="metric-value">{disease_prone_pct:.1f}%</p>
                <p class="metric-label">Disease Prone</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Kharif crop distribution
    st.markdown("""
    <div class="card">
        <h3 style="color: #16a34a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">Kharif Crop Distribution</h3>
    </div>
    """, unsafe_allow_html=True)
    
    fig_dist = px.pie(kharif_df, names='Label', 
                     title='Distribution of Kharif Crops',
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     height=400)
    fig_dist.update_traces(textposition='inside', textinfo='percent+label')
    fig_dist.update_layout(
        title_font_size=18,
        font=dict(family="Inter, sans-serif", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Modern tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Parameter Analysis", "Disease Analysis", "Recommendations"])
    
    # Tab 1: Parameter Analysis
    with tab1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Parameter Requirements for Kharif Crops</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Compare soil and environmental parameter requirements across different Kharif crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter selection
        param = st.selectbox("Select Parameter", 
                           ['Temperature', 'Humidity', 'pH', 'Rainfall', 'N', 'P', 'K'],
                           key="kharif_param",
                           format_func=lambda x: {'Temperature': 'Temperature (°C)', 
                                                'Humidity': 'Humidity (%)', 
                                                'pH': 'pH Level', 
                                                'Rainfall': 'Rainfall (mm)',
                                                'N': 'Nitrogen (N)',
                                                'P': 'Phosphorus (P)',
                                                'K': 'Potassium (K)'}[x])
        
        # Parameter comparison chart
        fig_param = px.box(kharif_df, x='Label', y=param,
                          title=f'{param} Requirements for Kharif Crops',
                          color='Label',
                          height=450)
        fig_param.update_layout(
            xaxis_title="Crop",
            yaxis_title=param,
            showlegend=False
        )
        st.plotly_chart(fig_param, use_container_width=True)
        
        # Parameter summary table
        st.subheader("Parameter Summary")
        
        # Calculate statistics for each crop and parameter
        param_summary = kharif_df.groupby('Label')[param].agg(['min', 'max', 'mean', 'median']).reset_index()
        param_summary.columns = ['Crop', 'Minimum', 'Maximum', 'Average', 'Median']
        param_summary = param_summary.round(2)
        
        # Display summary table with styling
        st.dataframe(param_summary, use_container_width=True)
    
    # Tab 2: Disease Analysis
    with tab2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Disease Analysis for Kharif Crops</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Understand disease patterns and susceptibility among Kharif crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if disease columns exist
        disease_cols = [col for col in kharif_df.columns if 'Disease' in col]
        
        if len(disease_cols) > 0:
            # Disease prone distribution
            if 'Disease_Prone' in disease_cols:
                # Count occurrences
                disease_prone_counts = kharif_df['Disease_Prone'].value_counts().reset_index()
                disease_prone_counts.columns = ['Status', 'Count']
                
                # Create pie chart
                fig_disease = px.pie(disease_prone_counts, names='Status', values='Count',
                                    title='Disease Proneness of Kharif Crops',
                                    color_discrete_sequence=['#10b981', '#f43f5e'],
                                    height=350)
                fig_disease.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_disease, use_container_width=True)
            
            # Common diseases
            disease_types = ['Common_Disease(Fungal)', 'Common_Disease(Bacterial)', 'Common_Disease(Viral)']
            available_types = [col for col in disease_types if col in kharif_df.columns]
            
            if available_types:
                st.subheader("Common Diseases in Kharif Crops")
                
                for disease_type in available_types:
                    # Skip if column doesn't exist
                    if disease_type not in kharif_df.columns:
                        continue
                    
                    # Get non-None diseases
                    disease_data = kharif_df[kharif_df[disease_type] != 'None']
                    
                    if len(disease_data) > 0:
                        # Format disease type for display
                        display_type = disease_type.replace('Common_Disease(', '').replace(')', '')
                        
                        st.markdown(f"##### {display_type} Diseases")
                        
                        # Count occurrences of each disease
                        disease_counts = disease_data[disease_type].value_counts().reset_index()
                        disease_counts.columns = ['Disease', 'Count']
                        
                        # Create horizontal bar chart
                        fig_disease_count = px.bar(disease_counts, 
                                                 y='Disease', 
                                                 x='Count',
                                                 title=f'Common {display_type} Diseases in Kharif Crops',
                                                 color_discrete_sequence=['#0284c7'],
                                                 height=max(250, len(disease_counts) * 50))
                        fig_disease_count.update_layout(xaxis_title="Number of Crops Affected")
                        st.plotly_chart(fig_disease_count, use_container_width=True)
                        
                    else:
                        display_type = disease_type.replace('Common_Disease(', '').replace(')', '')
                        st.info(f"No common {display_type.lower()} diseases found for Kharif crops in the dataset.")
        else:
            st.info("No disease information available in the dataset.")
    
    # Tab 3: Recommendations
    with tab3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Recommendations for Kharif Crops</h3>
            <p style="margin: 0 0 15px; color: #4b5563;">Best practices and recommendations for growing Kharif crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # General recommendations for Kharif crops
        st.markdown("""
        <div class="card">
            <h4 style="color: #16a34a; font-size: 1.1rem; margin-bottom: 12px;">General Best Practices</h4>
            <ul style="padding-left: 20px; color: #4b5563;">
                <li style="margin-bottom: 8px;">Prepare land well before the onset of monsoon (May-June).</li>
                <li style="margin-bottom: 8px;">Use drought-resistant varieties in regions with uncertain rainfall.</li>
                <li style="margin-bottom: 8px;">Implement proper drainage systems to prevent waterlogging.</li>
                <li style="margin-bottom: 8px;">Monitor for increased pest activity due to high humidity.</li>
                <li style="margin-bottom: 8px;">Apply balanced fertilizers based on crop stage and soil conditions.</li>
                <li style="margin-bottom: 8px;">Implement integrated pest management practices to reduce chemical usage.</li>
                <li style="margin-bottom: 8px;">Provide supplementary irrigation if monsoon is delayed or insufficient.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Crop-specific recommendations
        st.markdown("""
        <div class="card" style="margin-top: 20px;">
            <h4 style="color: #16a34a; font-size: 1.1rem; margin-bottom: 12px;">Select a Crop for Specific Recommendations</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Dropdown to select a crop
        selected_kharif_crop = st.selectbox(
            "Choose a Kharif crop",
            options=kharif_df['Label'].unique(),
            key="kharif_crop_select"
        )
        
        # Display crop-specific info
        if selected_kharif_crop:
            crop_data = kharif_df[kharif_df['Label'] == selected_kharif_crop].iloc[0]
            
            # Header for requirements
            st.subheader(f"{selected_kharif_crop} Requirements")
            
            # First row of metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">Temperature</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #0284c7;">{crop_data['Temperature']}°C</p>
                </div>""", unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">Humidity</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #0284c7;">{crop_data['Humidity']}%</p>
                </div>""", unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">pH</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #0284c7;">{crop_data['pH']}</p>
                </div>""", unsafe_allow_html=True)
            
            # Second row of metrics
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">N</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #0284c7;">{crop_data['N']} mg/kg</p>
                </div>""", unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">P</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #0284c7;">{crop_data['P']} mg/kg</p>
                </div>""", unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">K</p>
                """, unsafe_allow_html=True)
                st.markdown(f"""<p style="margin: 0; font-weight: 600; color: #0284c7;">{crop_data['K']} mg/kg</p>
                </div>""", unsafe_allow_html=True)
            
            # Show additional parameters if they exist
            additional_params = ['Salinity_dS_m', 'Water_Requirement', 'Disease_Resistance_Score']
            available_params = [param for param in additional_params if param in crop_data]
            
            if available_params:
                st.subheader("Additional Parameters")
                
                cols = st.columns(min(3, len(available_params)))
                
                for i, param in enumerate(available_params):
                    display_name = param.replace('_', ' ')
                    
                    # Add units based on parameter
                    unit = ""
                    if param == 'Salinity_dS_m':
                        unit = "dS/m"
                        display_name = "Salinity"
                    elif param == 'Water_Requirement':
                        unit = "mm"
                    elif param == 'Disease_Resistance_Score':
                        unit = "/10"
                        display_name = "Disease Resistance"
                    
                    value = crop_data[param]
                    
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                            <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; text-align: center;">
                                <p style="margin: 0 0 5px; font-size: 0.9rem; color: #4b5563;">{display_name}</p>
                                <p style="margin: 0; font-weight: 600; color: #0284c7;">{value} {unit}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Display disease information if available
            disease_cols = [col for col in crop_data.index if 'Disease' in col and col != 'Disease_Resistance_Score']
            
            if disease_cols:
                diseases = []
                
                for col in disease_cols:
                    if crop_data[col] != 'None' and str(crop_data[col]) != 'nan':
                        disease_type = col.replace('Common_Disease(', '').replace(')', '')
                        diseases.append(f"{crop_data[col]} ({disease_type})")
                
                if diseases:
                    st.subheader("Disease Information")
                    
                    st.markdown(f"""
                    <div style="margin-top: 10px; background-color: #fff8f1; border-left: 4px solid #f97316; padding: 15px; border-radius: 4px;">
                        <p style="margin: 0 0 10px; color: #4b5563;">Common diseases that affect {selected_kharif_crop}:</p>
                    """, unsafe_allow_html=True)
                    
                    # Display diseases as bullet points
                    for disease in diseases:
                        st.markdown(f"• {disease}", unsafe_allow_html=False)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.subheader("Disease Information")
                    st.success(f"✓ No common diseases recorded for {selected_kharif_crop} in our database.")

if __name__ == "__main__":
    main()

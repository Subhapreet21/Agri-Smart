import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def display_crop_distribution(df):
    """
    Display the distribution of crops in the dataset.
    """
    # Count the occurrences of each crop
    crop_counts = df['Label'].value_counts().reset_index()
    crop_counts.columns = ['Crop', 'Count']
    
    # Identify the top crops (for better visualization)
    top_crops = crop_counts.head(15)
    
    # Create a bar chart using Plotly
    fig = px.bar(
        top_crops,
        x='Crop',
        y='Count',
        color='Count',
        color_continuous_scale='Viridis',
        title='Distribution of Top Crops in Dataset'
    )
    
    fig.update_layout(
        xaxis_title='Crop',
        yaxis_title='Number of Instances',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a pie chart for disease-prone crops
    disease_prone = df['Disease_Prone'].value_counts().reset_index()
    disease_prone.columns = ['Disease Prone', 'Count']
    
    fig2 = px.pie(
        disease_prone,
        values='Count',
        names='Disease Prone',
        title='Distribution of Disease-Prone Crops',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Display crops by common diseases
    st.subheader("Crops by Common Diseases")
    
    # Collect disease data
    fungal_diseases = df[df['Common_Disease(Fungal)'] != 'None']['Common_Disease(Fungal)'].value_counts().head(5)
    bacterial_diseases = df[df['Common_Disease(Bacterial)'] != 'None']['Common_Disease(Bacterial)'].value_counts().head(5)
    viral_diseases = df[df['Common_Disease(Viral)'] != 'None']['Common_Disease(Viral)'].value_counts().head(5)
    
    # Create columns for each disease type
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Fungal Diseases")
        if not fungal_diseases.empty:
            fig_fungal = px.pie(
                values=fungal_diseases.values,
                names=fungal_diseases.index,
                title='Top Fungal Diseases',
                color_discrete_sequence=px.colors.sequential.Greens
            )
            st.plotly_chart(fig_fungal, use_container_width=True)
        else:
            st.info("No fungal disease data available")
    
    with col2:
        st.markdown("### Bacterial Diseases")
        if not bacterial_diseases.empty:
            fig_bacterial = px.pie(
                values=bacterial_diseases.values,
                names=bacterial_diseases.index,
                title='Top Bacterial Diseases',
                color_discrete_sequence=px.colors.sequential.Blues
            )
            st.plotly_chart(fig_bacterial, use_container_width=True)
        else:
            st.info("No bacterial disease data available")
    
    with col3:
        st.markdown("### Viral Diseases")
        if not viral_diseases.empty:
            fig_viral = px.pie(
                values=viral_diseases.values,
                names=viral_diseases.index,
                title='Top Viral Diseases',
                color_discrete_sequence=px.colors.sequential.Reds
            )
            st.plotly_chart(fig_viral, use_container_width=True)
        else:
            st.info("No viral disease data available")

def display_parameter_ranges(df):
    """
    Display the parameter ranges for different crops.
    """
    # Create selectbox for crop selection
    crops = sorted(df['Label'].unique())
    selected_crop = st.selectbox("Select a crop to view parameter ranges:", crops)
    
    # Filter data for the selected crop
    crop_data = df[df['Label'] == selected_crop]
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a radar chart for parameter ranges
        parameters = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        # Calculate mean values for each parameter
        mean_values = [crop_data[param].mean() for param in parameters]
        
        # Calculate min and max values for each parameter from the entire dataset
        min_values = [df[param].min() for param in parameters]
        max_values = [df[param].max() for param in parameters]
        
        # Normalize the mean values (0-1 scale)
        normalized_means = [(mean_values[i] - min_values[i]) / (max_values[i] - min_values[i]) 
                            if max_values[i] > min_values[i] else 0.5 
                            for i in range(len(parameters))]
        
        # Create radar chart using Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_means,
            theta=parameters,
            fill='toself',
            name=selected_crop,
            line_color='green'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Parameter Profile for {selected_crop}",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"Statistics for {selected_crop}")
        
        # Display statistics for each parameter
        for param in parameters:
            st.markdown(f"### {param}")
            
            # Calculate statistics
            min_val = crop_data[param].min()
            max_val = crop_data[param].max()
            mean_val = crop_data[param].mean()
            
            # Display the statistics
            st.metric(label="Average", value=f"{mean_val:.2f}")
            st.caption(f"Range: {min_val:.2f} - {max_val:.2f}")
        
        # Display additional crop information if available
        st.markdown("### Additional Information")
        
        # Water requirement
        if 'Water_Requirement' in crop_data.columns:
            water_req = crop_data['Water_Requirement'].mean()
            st.metric(label="Water Requirement (mm)", value=f"{water_req:.1f}")
        
        # Disease resistance score
        if 'Disease_Resistance_Score' in crop_data.columns:
            disease_score = crop_data['Disease_Resistance_Score'].mean()
            st.metric(label="Disease Resistance Score", value=f"{disease_score:.1f}/10")
            st.progress(float(disease_score) / 10.0)
        
        # Check if the crop is prone to diseases
        if 'Disease_Prone' in crop_data.columns:
            disease_prone = crop_data['Disease_Prone'].mode()[0]
            if disease_prone == 'Yes':
                st.warning("⚠️ This crop is prone to diseases")
            else:
                st.success("✅ This crop is generally disease-resistant")

def display_feature_importance():
    """
    Display the feature importance for crop recommendation.
    """
    # For demonstration, we use static feature importance
    # In a real implementation, this would come from the trained model
    
    features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    importance = [0.18, 0.15, 0.17, 0.21, 0.12, 0.08, 0.09]  # Example values
    
    # Create bar chart using Plotly
    fig = px.bar(
        x=features,
        y=importance,
        color=importance,
        color_continuous_scale='Viridis',
        labels={'x': 'Feature', 'y': 'Importance'},
        title='Feature Importance for Crop Recommendation'
    )
    
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Importance Score',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation between parameters
    st.subheader("Parameter Correlations")
    
    # Create a static correlation matrix for demonstration
    # In a real implementation, this would be calculated from the dataset
    correlation_data = pd.DataFrame([
        [1.00, 0.25, 0.18, 0.05, -0.12, 0.08, 0.02],
        [0.25, 1.00, 0.31, 0.10, -0.05, 0.15, 0.08],
        [0.18, 0.31, 1.00, 0.22, 0.07, 0.11, 0.15],
        [0.05, 0.10, 0.22, 1.00, 0.35, -0.09, 0.28],
        [-0.12, -0.05, 0.07, 0.35, 1.00, -0.22, 0.41],
        [0.08, 0.15, 0.11, -0.09, -0.22, 1.00, -0.14],
        [0.02, 0.08, 0.15, 0.28, 0.41, -0.14, 1.00]
    ], columns=features, index=features)
    
    # Create heatmap
    fig2 = px.imshow(
        correlation_data,
        color_continuous_scale='RdBu_r',
        labels=dict(x="Parameter", y="Parameter", color="Correlation"),
        title="Correlation Between Parameters"
    )
    
    fig2.update_layout(height=500)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### Understanding Feature Importance
    
    Feature importance helps us understand which soil and environmental parameters have the most significant impact on crop selection:
    
    - **Temperature**: Has the highest importance, indicating that different crops have specific temperature requirements.
    - **Nitrogen (N)**: The second most important feature, essential for plant growth and development.
    - **Potassium (K)**: Important for overall plant health and disease resistance.
    - **Phosphorus (P)**: Essential for root development and flowering.
    - **Humidity**: Affects plant transpiration and disease susceptibility.
    - **Rainfall**: Determines water availability for crops.
    - **pH**: Affects nutrient availability in the soil.
    
    The correlation heatmap shows how different parameters relate to each other. Strong positive correlations (blue) indicate parameters that tend to increase together, while negative correlations (red) show inverse relationships.
    """)

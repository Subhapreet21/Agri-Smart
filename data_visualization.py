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
    
    # Create a treemap for crop distribution
    st.subheader("Crop Distribution Treemap")
    fig_treemap = px.treemap(
        crop_counts,
        path=['Crop'],
        values='Count',
        color='Count',
        color_continuous_scale='Viridis',
        title='Distribution of Crops as Treemap'
    )
    
    fig_treemap.update_layout(height=500)
    st.plotly_chart(fig_treemap, use_container_width=True)
    
    # Create columns for more visualizations
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        # Create a histogram for water requirements
        if 'Water_Requirement' in df.columns:
            fig_water = px.histogram(
                df, 
                x='Water_Requirement',
                color='Disease_Prone',
                nbins=20,
                title='Water Requirements Distribution',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_water.update_layout(bargap=0.1)
            st.plotly_chart(fig_water, use_container_width=True)
        else:
            st.info("Water requirement data not available")
    
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
                color_discrete_sequence=px.colors.sequential.Greens,
                hole=0.4
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
                color_discrete_sequence=px.colors.sequential.Blues,
                hole=0.4
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
                color_discrete_sequence=px.colors.sequential.Reds,
                hole=0.4
            )
            st.plotly_chart(fig_viral, use_container_width=True)
        else:
            st.info("No viral disease data available")
            
    # Add a nutrient deficiency distribution chart
    st.subheader("Nutrient Deficiency Distribution")
    
    if 'Nutrient_Deficiency' in df.columns:
        nutrient_def = df[df['Nutrient_Deficiency'] != 'None']
        if not nutrient_def.empty:
            nutrient_counts = nutrient_def['Nutrient_Deficiency'].value_counts().reset_index()
            nutrient_counts.columns = ['Nutrient', 'Count']
            
            fig_nutrient = px.bar(
                nutrient_counts,
                x='Nutrient',
                y='Count',
                color='Count',
                title='Common Nutrient Deficiencies in Crops',
                color_continuous_scale='YlOrRd'
            )
            
            fig_nutrient.update_layout(height=400)
            st.plotly_chart(fig_nutrient, use_container_width=True)
        else:
            st.info("No nutrient deficiency data available")
    else:
        st.info("Nutrient deficiency data not available")

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
        
        # Add parameter distribution comparison
        st.subheader("Parameter Distribution Comparison")
        
        # Create tabs for each parameter
        param_tabs = st.tabs(parameters)
        
        for i, param in enumerate(parameters):
            with param_tabs[i]:
                # Create histogram comparing this crop's parameter with overall distribution
                fig_hist = go.Figure()
                
                # Add histogram for all crops (background)
                fig_hist.add_trace(go.Histogram(
                    x=df[param],
                    name='All Crops',
                    opacity=0.6,
                    nbinsx=20,
                    marker_color='lightgray'
                ))
                
                # Add histogram for selected crop
                fig_hist.add_trace(go.Histogram(
                    x=crop_data[param],
                    name=selected_crop,
                    opacity=0.8,
                    nbinsx=20,
                    marker_color='green'
                ))
                
                fig_hist.update_layout(
                    barmode='overlay',
                    title=f"{param} Distribution: {selected_crop} vs All Crops",
                    xaxis_title=param,
                    yaxis_title="Count",
                    height=300
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
    
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
                
        # Salinity tolerance if available
        if 'Salinity_dS_m' in crop_data.columns:
            salinity = crop_data['Salinity_dS_m'].mean()
            st.metric(label="Salinity Tolerance (dS/m)", value=f"{salinity:.2f}")
            
            # Interpret the salinity value
            if salinity < 1.0:
                st.caption("Low salt tolerance")
            elif salinity < 2.0:
                st.caption("Moderate salt tolerance")
            else:
                st.caption("High salt tolerance")

def display_feature_importance():
    """
    Display the feature importance for crop recommendation.
    """
    # For demonstration, we use static feature importance
    # In a real implementation, this would come from the trained model
    
    features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    importance = [0.18, 0.15, 0.17, 0.21, 0.12, 0.08, 0.09]  # Example values
    
    # Create tabs for different visualizations
    importance_tabs = st.tabs(["Bar Chart", "Sunburst", "3D Visualization"])
    
    with importance_tabs[0]:
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
    
    with importance_tabs[1]:
        # Create sunburst chart
        df_sunburst = pd.DataFrame({
            'Features': features,
            'Importance': importance,
            'Category': ['Nutrient', 'Nutrient', 'Nutrient', 'Environmental', 'Environmental', 'Soil', 'Environmental']
        })
        
        fig_sunburst = px.sunburst(
            df_sunburst,
            path=['Category', 'Features'],
            values='Importance',
            color='Importance',
            color_continuous_scale='Viridis',
            title='Feature Importance by Category'
        )
        
        fig_sunburst.update_layout(height=500)
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with importance_tabs[2]:
        # Create a 3D visualization (scatter plot in 3D space)
        # We'll visualize N, P, K in 3D space with Temperature as the color
        df_3d = pd.DataFrame({
            'N': [20, 40, 60, 80, 100, 120],
            'P': [15, 30, 45, 60, 75, 90],
            'K': [30, 40, 50, 60, 70, 80],
            'Temperature': [18, 22, 25, 28, 32, 35],
            'Crop': ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Potato']
        })
        
        fig_3d = px.scatter_3d(
            df_3d,
            x='N',
            y='P',
            z='K',
            color='Temperature',
            size=[importance[0]]*6,  # Use N importance for size
            text='Crop',
            title='3D Visualization of Key Parameters',
            color_continuous_scale='Viridis'
        )
        
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
    
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
    
    # Create correlation visualization tabs
    corr_tabs = st.tabs(["Heatmap", "Network Graph"])
    
    with corr_tabs[0]:
        # Create heatmap
        fig2 = px.imshow(
            correlation_data,
            color_continuous_scale='RdBu_r',
            labels=dict(x="Parameter", y="Parameter", color="Correlation"),
            title="Correlation Between Parameters"
        )
        
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    
    with corr_tabs[1]:
        # Create a network graph of correlations
        # First, we need to transform the correlation matrix into edge data
        edges = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if abs(correlation_data.iloc[i, j]) > 0.1:  # Only include correlations above threshold
                    edges.append({
                        'source': features[i],
                        'target': features[j],
                        'value': abs(correlation_data.iloc[i, j]),
                        'color': 'blue' if correlation_data.iloc[i, j] > 0 else 'red'
                    })
        
        # Create network graph
        edge_x = []
        edge_y = []
        edge_colors = []
        
        # Simple layout for nodes (in a circle)
        import math
        node_x = [math.cos(2*math.pi*i/len(features)) for i in range(len(features))]
        node_y = [math.sin(2*math.pi*i/len(features)) for i in range(len(features))]
        
        # Add edges
        for edge in edges:
            i = features.index(edge['source'])
            j = features.index(edge['target'])
            
            edge_x.extend([node_x[i], node_x[j], None])
            edge_y.extend([node_y[i], node_y[j], None])
            edge_colors.append(edge['color'])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=features,
            marker=dict(
                showscale=True,
                color=[importance[i]*5 for i in range(len(features))],
                size=15,
                colorscale='Viridis',
                line_width=2
            )
        )
        
        # Create figure
        fig_network = go.Figure(data=[edge_trace, node_trace],
                              layout=go.Layout(
                                  title='Parameter Correlation Network',
                                  showlegend=False,
                                  hovermode='closest',
                                  margin=dict(b=20, l=5, r=5, t=40),
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  height=500
                              ))
        
        st.plotly_chart(fig_network, use_container_width=True)
    
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
    
    # Add crop comparison functionality
    st.subheader("Crop Comparison Tool")
    st.write("Compare optimal growing conditions for different crops")
    
    # Select crops to compare (multi-select)
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample crop data for demonstration
        crop_data = {
            'Rice': {'N': 80, 'P': 40, 'K': 40, 'Temperature': 25, 'pH': 6.5, 'Water': 'High'},
            'Wheat': {'N': 60, 'P': 30, 'K': 30, 'Temperature': 20, 'pH': 7.0, 'Water': 'Medium'},
            'Maize': {'N': 70, 'P': 35, 'K': 40, 'Temperature': 23, 'pH': 6.8, 'Water': 'Medium'},
            'Potato': {'N': 90, 'P': 45, 'K': 60, 'Temperature': 18, 'pH': 6.0, 'Water': 'Medium-High'},
            'Cotton': {'N': 50, 'P': 25, 'K': 25, 'Temperature': 28, 'pH': 6.2, 'Water': 'Low-Medium'},
            'Sugarcane': {'N': 100, 'P': 50, 'K': 50, 'Temperature': 27, 'pH': 6.5, 'Water': 'High'}
        }
        
        selected_crops = st.multiselect(
            "Select crops to compare",
            options=list(crop_data.keys()),
            default=list(crop_data.keys())[:3]
        )
    
    with col2:
        # Select parameters to compare
        selected_params = st.multiselect(
            "Select parameters to compare",
            options=['N', 'P', 'K', 'Temperature', 'pH'],
            default=['N', 'P', 'K']
        )
    
    if selected_crops and selected_params:
        # Create comparison data
        comparison_data = []
        
        for crop in selected_crops:
            for param in selected_params:
                comparison_data.append({
                    'Crop': crop,
                    'Parameter': param,
                    'Value': crop_data[crop][param]
                })
        
        # Convert to DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        fig_comparison = px.bar(
            df_comparison,
            x='Crop',
            y='Value',
            color='Parameter',
            barmode='group',
            title='Crop Parameter Comparison',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig_comparison.update_layout(height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Create radar chart for comparison
        fig_radar = go.Figure()
        
        for crop in selected_crops:
            values = [crop_data[crop][param] for param in selected_params]
            # Normalize values for better visualization
            max_values = [max([crop_data[c][param] for c in selected_crops]) for param in selected_params]
            normalized_values = [values[i]/max_values[i] for i in range(len(values))]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=selected_params,
                fill='toself',
                name=crop
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Comparative Crop Requirements (Normalized)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

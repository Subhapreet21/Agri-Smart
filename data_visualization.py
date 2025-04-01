import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def display_crop_distribution(df):
    """
    Display the distribution of crops in the dataset using simple visualizations.
    With segregation of Rabi and Kharif crops.
    """
    # Add crop season classification
    rabi_crops = ['Wheat', 'Barley', 'Chickpea', 'Peas', 'Mustard', 'Linseed', 'Fennel', 'Cumin']
    kharif_crops = ['Rice', 'Maize', 'Bajra', 'Jowar', 'Soybean', 'Cotton', 'Sugarcane', 'Turmeric', 'Chilly', 'Okra', 'Brinjal', 'Paddy']
    
    # Create a copy to avoid modifying the original
    df_viz = df.copy()
    
    # Add season column
    df_viz['Season'] = 'Other'
    for crop in rabi_crops:
        df_viz.loc[df_viz['Label'] == crop, 'Season'] = 'Rabi'
    for crop in kharif_crops:
        df_viz.loc[df_viz['Label'] == crop, 'Season'] = 'Kharif'
    
    # Create tabs for different views
    dist_tab1, dist_tab2 = st.tabs(["All Crops", "Seasonal View"])
    
    with dist_tab1:
        # Count the occurrences of each crop
        crop_counts = df['Label'].value_counts().reset_index()
        crop_counts.columns = ['Crop', 'Count']
        
        # Identify the top 10 crops (for better visualization)
        top_crops = crop_counts.head(10)
        
        # Explanation for users with modern styling
        st.markdown("""
        <div class="card" style="margin-bottom: 20px;">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Common Crops in the Dataset</h3>
            <p style="margin: 0; color: #4b5563;">This chart shows the <strong>most common crops</strong> in our dataset. 
            Taller bars represent crops that appear more frequently.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a modern bar chart using Plotly
        fig = px.bar(
            top_crops,
            x='Crop',
            y='Count',
            color='Crop',  # Each crop gets its own color
            color_discrete_sequence=px.colors.qualitative.Bold,
            title='Top 10 Most Common Crops',
        )
        
        fig.update_layout(
            xaxis_title='Crop Type',
            yaxis_title='Number of Crops',
            height=400,
            showlegend=False,  # Hide legend as colors are just for visual distinction
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_tab2:
        # Create a season-based visualization
        st.markdown("""
        <div class="card" style="margin-bottom: 20px;">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Seasonal Crop Distribution</h3>
            <p style="margin: 0; color: #4b5563;">This view categorizes crops by their growing seasons:
            <strong>Rabi</strong> (winter crops) and <strong>Kharif</strong> (monsoon crops).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Season counts
        season_counts = df_viz['Season'].value_counts().reset_index()
        season_counts.columns = ['Season', 'Count']
        
        # Create a pie chart for seasons
        fig_season = px.pie(
            season_counts, 
            values='Count', 
            names='Season',
            color='Season',
            color_discrete_map={'Rabi': '#16a34a', 'Kharif': '#ca8a04', 'Other': '#64748b'},
            title='Distribution by Growing Season',
            hole=0.4
        )
        
        fig_season.update_layout(
            height=300
        )
        
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Create a grouped bar chart for seasonal crops
        seasonal_crops = df_viz.groupby(['Season', 'Label']).size().reset_index(name='Count')
        seasonal_crops = seasonal_crops[seasonal_crops['Season'] != 'Other'].sort_values('Count', ascending=False).head(10)
        
        fig_seasonal = px.bar(
            seasonal_crops,
            x='Label',
            y='Count',
            color='Season',
            color_discrete_map={'Rabi': '#16a34a', 'Kharif': '#ca8a04'},
            title='Top Crops by Season',
            barmode='group'
        )
        
        fig_seasonal.update_layout(
            xaxis_title='Crop',
            yaxis_title='Count',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Create columns for more visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a simple pie chart for disease-prone crops
        disease_prone = df['Disease_Prone'].value_counts().reset_index()
        disease_prone.columns = ['Disease Prone', 'Count']
        
        # Explanation for users
        st.markdown("""
        ### Crops and Disease Risk
        This chart shows what percentage of crops are prone to diseases.
        """)
        
        fig2 = px.pie(
            disease_prone,
            values='Count',
            names='Disease Prone',
            color_discrete_sequence=['#4CAF50', '#F44336'],  # Green for No, Red for Yes
            hole=0.3
        )
        
        fig2.update_layout(
            height=350,
            legend_title="Disease Prone"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Create a simple bar chart for water requirements by crop category
        if 'Water_Requirement' in df.columns:
            # Explanation for users
            st.markdown("""
            ### Water Needs by Disease Risk
            This chart compares water requirements for disease-prone vs. disease-resistant crops.
            """)
            
            # Group by disease prone and get average water requirement
            water_by_disease = df.groupby('Disease_Prone')['Water_Requirement'].mean().reset_index()
            
            fig_water = px.bar(
                water_by_disease,
                x='Disease_Prone',
                y='Water_Requirement',
                color='Disease_Prone',
                color_discrete_sequence=['#4CAF50', '#F44336'],  # Green for No, Red for Yes
                title='Average Water Requirements',
                labels={'Disease_Prone': 'Disease Prone', 'Water_Requirement': 'Water Needed (mm)'}
            )
            
            fig_water.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_water, use_container_width=True)
        else:
            st.info("Water requirement data not available")
    
    # Display a simple overview of common diseases
    st.subheader("Common Crop Diseases")
    st.markdown("""
    This section shows the most common diseases that affect crops. Understanding 
    these diseases can help with better crop management and disease prevention.
    """)
    
    # Collect disease data
    disease_types = {
        'Fungal': {'color': '#8BC34A', 'description': 'Caused by fungi - often appear as spots, rot, or mildew'},
        'Bacterial': {'color': '#2196F3', 'description': 'Caused by bacteria - often appear as spots, wilting, or rot'},
        'Viral': {'color': '#F44336', 'description': 'Caused by viruses - often cause stunting, yellowing, or mosaic patterns'}
    }
    
    # Create a simple table with disease types and descriptions
    for disease_type, info in disease_types.items():
        col = f'Common_Disease({disease_type})'
        if col in df.columns:
            # Get diseases that aren't "None"
            diseases = df[df[col] != 'None'][col].value_counts().head(3)
            
            if not diseases.empty:
                st.markdown(f"### {disease_type} Diseases")
                st.markdown(f"*{info['description']}*")
                
                # Create a simple horizontal bar chart
                diseases_df = diseases.reset_index()
                diseases_df.columns = ['Disease', 'Count']
                
                fig_disease = px.bar(
                    diseases_df,
                    y='Disease',
                    x='Count',
                    orientation='h',
                    color_discrete_sequence=[info['color']]
                )
                
                fig_disease.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Number of Affected Crops",
                    yaxis_title=""
                )
                
                st.plotly_chart(fig_disease, use_container_width=True)
            else:
                st.info(f"No {disease_type.lower()} disease data available")
    
    # Add a simple nutrient deficiency section
    if 'Nutrient_Deficiency' in df.columns:
        nutrient_def = df[df['Nutrient_Deficiency'] != 'None']
        if not nutrient_def.empty:
            st.subheader("Common Nutrient Deficiencies")
            st.markdown("""
            This chart shows which nutrient deficiencies are most common in crops.
            Addressing these deficiencies can help improve crop health and yield.
            """)
            
            nutrient_counts = nutrient_def['Nutrient_Deficiency'].value_counts().reset_index()
            nutrient_counts.columns = ['Nutrient', 'Count']
            
            fig_nutrient = px.pie(
                nutrient_counts,
                values='Count',
                names='Nutrient',
                color_discrete_sequence=px.colors.qualitative.Safe,  # Simple, distinct colors
                title='Common Nutrient Deficiencies'
            )
            
            fig_nutrient.update_traces(textposition='inside', textinfo='percent+label')
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

def display_parameter_analytics(df):
    """
    Display advanced analytics about crop parameters including correlations
    and parameter relationships for agricultural data insights.
    """
    # Define crop season classifications
    rabi_crops = ['Wheat', 'Barley', 'Chickpea', 'Peas', 'Mustard', 'Linseed', 'Fennel', 'Cumin']
    kharif_crops = ['Rice', 'Maize', 'Bajra', 'Jowar', 'Soybean', 'Cotton', 'Sugarcane', 'Turmeric', 'Chilly', 'Okra', 'Brinjal', 'Paddy']
    
    # Add tabs for different analytics views
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["Parameter Correlations", "Parameter Relationships", "Success Rate Analysis"])
    
    # Dictionary to convert column names to correct case
    column_mapping = {
        'n': 'N', 'p': 'P', 'k': 'K', 
        'temperature': 'Temperature', 'humidity': 'Humidity', 
        'ph': 'pH', 'rainfall': 'Rainfall'
    }
    
    # Create a copy of dataframe with lowercase columns for easier matching
    df_lower = df.copy()
    
    # Tab 1: Parameter Correlations
    with analytics_tab1:
        st.markdown("""
        <div class="card" style="margin-bottom: 20px;">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Parameter Correlation Analysis</h3>
            <p style="margin: 0; color: #4b5563;">This heatmap shows how different soil and environmental parameters are related to each other.
            Stronger colors indicate stronger relationships between parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Define the features to analyze
        corr_features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        # Calculate correlation matrix
        corr_matrix = df[corr_features].corr().round(2)
        
        # Create correlation heatmap with modern styling
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_features,
            y=corr_features,
            colorscale='Blues',
            zmin=-1, zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text}',
            textfont={"size": 12},
        ))
        
        fig_corr.update_layout(
            title='Parameter Correlation Matrix',
            height=450,
            width=650,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Add correlation explanation
        st.markdown("""
        <div style="background-color: #f0fdf4; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <h4 style="color: #16a34a; margin-top: 0;">How to Read This Chart</h4>
            <ul style="margin-bottom: 0; padding-left: 20px; color: #4b5563;">
                <li>Values close to <strong>1</strong> (dark blue) indicate <strong>strong positive correlation</strong> - when one parameter increases, the other also increases</li>
                <li>Values close to <strong>-1</strong> (dark red) indicate <strong>strong negative correlation</strong> - when one parameter increases, the other decreases</li>
                <li>Values close to <strong>0</strong> (light colors) indicate <strong>little or no correlation</strong> - parameters change independently</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Add seasonal correlation analysis
        st.subheader("Seasonal Parameter Correlations")
        
        # Add crop season classification if not already added
        if 'Season' not in df.columns:
            rabi_crops = ['Wheat', 'Barley', 'Chickpea', 'Peas', 'Mustard', 'Linseed', 'Fennel', 'Cumin']
            kharif_crops = ['Rice', 'Maize', 'Bajra', 'Jowar', 'Soybean', 'Cotton', 'Sugarcane', 'Turmeric', 'Chilly', 'Okra', 'Brinjal', 'Paddy']
            
            df_lower['Season'] = 'Other'
            for crop in rabi_crops:
                df_lower.loc[df_lower['Label'] == crop, 'Season'] = 'Rabi'
            for crop in kharif_crops:
                df_lower.loc[df_lower['Label'] == crop, 'Season'] = 'Kharif'
        
        # Create seasonal correlation tabs
        season_tab1, season_tab2 = st.tabs(["Rabi (Winter) Crops", "Kharif (Monsoon) Crops"])
        
        with season_tab1:
            rabi_df = df_lower[df_lower['Season'] == 'Rabi']
            if not rabi_df.empty:
                rabi_corr = rabi_df[corr_features].corr().round(2)
                
                fig_rabi = go.Figure(data=go.Heatmap(
                    z=rabi_corr,
                    x=corr_features,
                    y=corr_features,
                    colorscale='Blues',
                    zmin=-1, zmax=1,
                    text=rabi_corr.values,
                    texttemplate='%{text}',
                    textfont={"size": 12},
                ))
                
                fig_rabi.update_layout(
                    title='Rabi Crop Parameter Correlations',
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig_rabi, use_container_width=True)
            else:
                st.info("No Rabi crop data available for correlation analysis")
        
        with season_tab2:
            kharif_df = df_lower[df_lower['Season'] == 'Kharif']
            if not kharif_df.empty:
                kharif_corr = kharif_df[corr_features].corr().round(2)
                
                fig_kharif = go.Figure(data=go.Heatmap(
                    z=kharif_corr,
                    x=corr_features,
                    y=corr_features,
                    colorscale='Blues',
                    zmin=-1, zmax=1,
                    text=kharif_corr.values,
                    texttemplate='%{text}',
                    textfont={"size": 12},
                ))
                
                fig_kharif.update_layout(
                    title='Kharif Crop Parameter Correlations',
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig_kharif, use_container_width=True)
            else:
                st.info("No Kharif crop data available for correlation analysis")
    
    # Tab 2: Parameter Relationships
    with analytics_tab2:
        st.markdown("""
        <div class="card" style="margin-bottom: 20px;">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Parameter Relationship Analysis</h3>
            <p style="margin: 0; color: #4b5563;">Explore relationships between different parameters and how they affect various crops.
            Select parameters for X and Y axes to visualize their relationships.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for parameter selection
        col1, col2 = st.columns(2)
        
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
            # Create scatter plot with crop label coloring
            fig_scatter = px.scatter(
                df,
                x=selected_x_param,
                y=selected_y_param,
                color='Label',
                hover_data=['Label', 'Temperature', 'Humidity', 'pH'],
                title=f'{selected_x_param} vs {selected_y_param} by Crop Type',
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=500
            )
            
            # Add seasonal grouping
            if 'Season' in df_lower.columns:
                # Create a pattern to show Rabi/Kharif differences
                season_patterns = []
                for label in fig_scatter.data:
                    crop_name = label.name
                    if crop_name in rabi_crops:
                        label.marker.symbol = 'circle'
                        label.marker.size = 10
                    elif crop_name in kharif_crops:
                        label.marker.symbol = 'diamond'
                        label.marker.size = 10
            
            # Add trend line
            fig_scatter.update_layout(
                xaxis_title=selected_x_param,
                yaxis_title=selected_y_param,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend_title="Crop Type",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Add seasonal filtering 
            show_seasonal = st.checkbox("Filter by growing season")
            if show_seasonal and 'Season' in df_lower.columns:
                season_filter = st.selectbox("Select growing season", ["All", "Rabi", "Kharif"])
                
                if season_filter != "All":
                    filtered_df = df_lower[df_lower['Season'] == season_filter]
                    
                    fig_seasonal = px.scatter(
                        filtered_df,
                        x=selected_x_param,
                        y=selected_y_param,
                        color='Label',
                        hover_data=['Label', 'Temperature', 'Humidity', 'pH'],
                        title=f'{selected_x_param} vs {selected_y_param} for {season_filter} Crops',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        height=500
                    )
                    
                    fig_seasonal.update_layout(
                        xaxis_title=selected_x_param,
                        yaxis_title=selected_y_param,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig_seasonal, use_container_width=True)
        
    # Tab 3: Success Rate Analysis
    with analytics_tab3:
        st.markdown("""
        <div class="card" style="margin-bottom: 20px;">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">Success Rate Analysis</h3>
            <p style="margin: 0; color: #4b5563;">Analyze how different environmental parameters affect disease proneness and success rates of crops.
            This can help identify optimal growing conditions for better yields.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter selection for analysis
        param = st.selectbox(
            "Select Environmental Parameter",
            ['Temperature', 'Humidity', 'pH', 'Rainfall']
        )
        
        if 'Disease_Prone' in df.columns:
            # Create histogram showing disease proneness by parameter
            fig_success = px.histogram(
                df,
                x=param,
                color='Disease_Prone',
                barmode='group',
                title=f'Disease Proneness by {param}',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#F44336'},
                labels={'Disease_Prone': 'Disease Prone'},
                height=400
            )
            
            fig_success.update_layout(
                xaxis_title=param,
                yaxis_title='Number of Crops',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig_success, use_container_width=True)
            
            # Add seasonal analysis
            if 'Season' in df_lower.columns:
                st.subheader("Seasonal Success Rate Analysis")
                
                # Create cross-tab of season vs disease prone
                season_disease = pd.crosstab(df_lower['Season'], df_lower['Disease_Prone'])
                
                # Calculate percentage
                season_disease_pct = season_disease.div(season_disease.sum(axis=1), axis=0) * 100
                
                # Create a stacked bar chart
                fig_season = px.bar(
                    season_disease_pct.reset_index(),
                    x='Season',
                    y=['No', 'Yes'],
                    title='Disease Proneness by Growing Season',
                    color_discrete_map={'No': '#4CAF50', 'Yes': '#F44336'},
                    labels={'value': 'Percentage', 'variable': 'Disease Prone'},
                    height=400
                )
                
                fig_season.update_layout(
                    xaxis_title='Growing Season',
                    yaxis_title='Percentage of Crops',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    barmode='stack'
                )
                
                st.plotly_chart(fig_season, use_container_width=True)
                
                # Add interpretation
                st.markdown("""
                <div style="background-color: #f0fdf4; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="color: #16a34a; margin-top: 0;">Interpreting Seasonal Differences</h4>
                    <p style="color: #4b5563;">This chart shows the percentage of crops in each growing season that are prone to diseases vs. those that are disease-resistant. 
                    A higher green portion indicates more disease-resistant crops in that season.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Disease proneness data not available for success rate analysis")
        
        # Add optimal parameters analysis
        st.subheader("Optimal Parameter Ranges")
        
        # Create box plots for each parameter by disease proneness
        if 'Disease_Prone' in df.columns:
            param_list = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            
            for i in range(0, len(param_list), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(param_list):
                        with cols[j]:
                            cur_param = param_list[i + j]
                            
                            fig_box = px.box(
                                df,
                                x='Disease_Prone',
                                y=cur_param,
                                color='Disease_Prone',
                                title=f'{cur_param} Range by Disease Proneness',
                                color_discrete_map={'No': '#4CAF50', 'Yes': '#F44336'},
                                height=300
                            )
                            
                            fig_box.update_layout(
                                xaxis_title='Disease Prone',
                                yaxis_title=cur_param,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_box, use_container_width=True)
                            
                            # Calculate and display the optimal range for disease-resistant crops
                            resistant_crops = df[df['Disease_Prone'] == 'No']
                            if not resistant_crops.empty:
                                q1 = resistant_crops[cur_param].quantile(0.25)
                                q3 = resistant_crops[cur_param].quantile(0.75)
                                
                                st.markdown(f"""
                                <div style="background-color: #f0fdf4; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                                    <p style="margin: 0; color: #16a34a; font-weight: 500;">Optimal range: {q1:.1f} - {q3:.1f}</p>
                                </div>
                                """, unsafe_allow_html=True)

def display_feature_importance():
    """
    Display the feature importance for crop recommendation in a simple, understandable way.
    """
    # For demonstration, we use static feature importance values
    # In a real implementation, these would come from the trained model
    
    # Extended features including new ones
    features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 
                'Disease_Resistance_Score', 'Water_Requirement', 'Salinity_dS_m']
    importance = [0.16, 0.14, 0.15, 0.18, 0.10, 0.07, 0.08, 0.05, 0.04, 0.03]  # Example values
    
    # Convert feature abbreviations to full names for better understanding
    feature_names = {
        'N': 'Nitrogen',
        'P': 'Phosphorus',
        'K': 'Potassium',
        'Temperature': 'Temperature',
        'Humidity': 'Humidity',
        'pH': 'Soil pH',
        'Rainfall': 'Rainfall',
        'Disease_Resistance_Score': 'Disease Resistance',
        'Water_Requirement': 'Water Requirement',
        'Salinity_dS_m': 'Soil Salinity'
    }
    
    # Add explanations for each feature
    feature_explanations = {
        'N': 'Essential for leaf growth and overall plant development',
        'P': 'Important for root development and flowering',
        'K': 'Helps with disease resistance and overall plant health',
        'Temperature': 'Different crops need different temperature ranges',
        'Humidity': 'Affects transpiration and disease susceptibility',
        'pH': 'Determines nutrient availability in soil',
        'Rainfall': 'Water availability for plant growth',
        'Disease_Resistance_Score': 'Natural ability to resist common diseases',
        'Water_Requirement': 'Amount of water needed for optimal growth',
        'Salinity_dS_m': 'Tolerance to salt content in soil'
    }
    
    # Create a simple header with modern styling
    st.markdown("""
    <div class="card" style="margin-bottom: 25px;">
        <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 10px;">What Makes a Crop Grow Well?</h3>
        <p style="margin: 0; color: #4b5563;">Different factors affect crop growth in different ways. Some are more important than others.
        This visualization shows how important each factor is for deciding which crop to plant.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    imp_tab1, imp_tab2 = st.tabs(["Interactive Chart", "Detailed Explanation"])
    
    with imp_tab1:
        # Create feature importance data
        importance_data = pd.DataFrame({
            'Feature': [feature_names[f] for f in features],
            'Importance': importance,
            'Description': [feature_explanations[f] for f in features]
        })
        
        # Sort by importance for better visualization
        importance_data = importance_data.sort_values('Importance', ascending=False)
        
        # Create a more modern interactive chart 
        fig = px.bar(
            importance_data,
            y='Feature',
            x='Importance',
            color='Importance',
            color_continuous_scale='Viridis',
            labels={'Feature': 'Growing Factor', 'Importance': 'Importance Score'},
            title='Factor Importance for Crop Selection',
            orientation='h',  # horizontal bars for better readability with many features
            hover_data=['Description']
        )
        
        fig.update_layout(
            yaxis=dict(title=''),
            xaxis=dict(title='Importance Score'),
            height=500,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="margin-top: 10px; padding: 10px; background-color: #f0fdf4; border-radius: 5px;">
            <p style="margin: 0; color: #16a34a; font-weight: 500;">💡 Hover over bars to see descriptions</p>
        </div>
        """, unsafe_allow_html=True)
        
    with imp_tab2:
        # Create a more detailed explanation of factors with modern styling
        st.markdown("""
        <div class="card">
            <h3 style="color: #16a34a; font-size: 1.2rem; margin-bottom: 15px;">Understanding Growing Factors</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for the factors
        col1, col2 = st.columns(2)
        
        # Distribute the factors between the columns
        sorted_features = sorted(features, key=lambda x: importance[features.index(x)], reverse=True)
        half = len(sorted_features) // 2
        
        with col1:
            for f in sorted_features[:half]:
                idx = features.index(f)
                st.markdown(f"""
                <div class="card" style="margin-bottom: 15px;">
                    <h4 style="color: #166534; font-size: 1rem; margin-bottom: 5px;">{feature_names[f]}</h4>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="flex-grow: 1; background-color: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background-color: #16a34a; width: {importance[idx]*100}%; height: 100%;"></div>
                        </div>
                        <div style="margin-left: 10px; font-weight: 500; color: #16a34a;">{importance[idx]:.2f}</div>
                    </div>
                    <p style="margin: 5px 0 0; color: #4b5563; font-size: 0.9rem;">{feature_explanations[f]}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for f in sorted_features[half:]:
                idx = features.index(f)
                st.markdown(f"""
                <div class="card" style="margin-bottom: 15px;">
                    <h4 style="color: #166534; font-size: 1rem; margin-bottom: 5px;">{feature_names[f]}</h4>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="flex-grow: 1; background-color: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background-color: #16a34a; width: {importance[idx]*100}%; height: 100%;"></div>
                        </div>
                        <div style="margin-left: 10px; font-weight: 500; color: #16a34a;">{importance[idx]:.2f}</div>
                    </div>
                    <p style="margin: 5px 0 0; color: #4b5563; font-size: 0.9rem;">{feature_explanations[f]}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Add a simple explanation of what these values mean
    st.markdown("""
    ### What This Means
    
    The taller the bar, the more important that factor is when choosing the right crop for your land.
    """)
    
    # Create a simple table explaining each factor
    st.subheader("Understanding Each Factor")
    
    for feature in features:
        with st.expander(f"{feature_names[feature]}", expanded=False):
            col1, col2 = st.columns([1, 3])
            with col1:
                # Show importance as a simple gauge
                importance_value = importance[features.index(feature)]
                st.metric("Importance", f"{importance_value:.2f}")
                st.progress(importance_value)
            with col2:
                st.markdown(f"**{feature_explanations[feature]}**")
                
                if feature == 'N':
                    st.markdown("- Helps plants grow leafy and green")
                    st.markdown("- Too little: yellow leaves, stunted growth")
                    st.markdown("- Too much: lots of leaves but weak stems")
                elif feature == 'P':
                    st.markdown("- Helps develop strong roots")
                    st.markdown("- Important for flowering and seed production")
                    st.markdown("- Too little: poor growth and few flowers")
                elif feature == 'K':
                    st.markdown("- Strengthens plants against disease")
                    st.markdown("- Helps regulate water in plants")
                    st.markdown("- Too little: weak plants, poor crop quality")
                elif feature == 'Temperature':
                    st.markdown("- Each crop has an ideal temperature range")
                    st.markdown("- Too cold: slow growth or damage")
                    st.markdown("- Too hot: stress, wilting, or damage")
                elif feature == 'Humidity':
                    st.markdown("- Affects how plants lose water")
                    st.markdown("- Too low: plants dry out quickly")
                    st.markdown("- Too high: can encourage fungal diseases")
                elif feature == 'pH':
                    st.markdown("- Affects whether plants can access nutrients")
                    st.markdown("- Most crops prefer slightly acidic to neutral soil (pH 6-7)")
                    st.markdown("- Wrong pH: nutrients are locked in soil, unavailable to plants")
                elif feature == 'Rainfall':
                    st.markdown("- Water is essential for all plant growth")
                    st.markdown("- Too little: drought stress, wilting")
                    st.markdown("- Too much: root rot, nutrient leaching")
    
    # Show a simple relationship between parameters
    st.subheader("How These Factors Work Together")
    
    st.markdown("""
    Some growing factors work together, while others counteract each other. 
    This simplified chart shows how they relate.
    """)
    
    # Create a simplified correlation explanation with colored boxes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Factors that work together")
        st.markdown("When one increases, the other often increases too:")
        
        st.markdown("""
        - Temperature 🔼 and Humidity 🔼
        - Rainfall 🔼 and Humidity 🔼
        - Nitrogen 🔼 and plant growth 🔼
        """)
        
    with col2:
        st.markdown("### Factors that counteract each other")
        st.markdown("When one increases, the other often decreases:")
        
        st.markdown("""
        - pH (high/alkaline) 🔼 and nutrient availability 🔽
        - Too much rain 🔼 and fertilizer effectiveness 🔽
        - High temperature 🔼 and water retention 🔽 
        """)
    
    # Add a simple diagram showing relationship between factors
    st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f331.svg", width=50)
    st.markdown("""
    *The right balance of all these factors leads to healthy crops!*
    """)
    
    # Show a recommendation message
    st.markdown("""
    ### How to Use This Information
    
    When planning what to plant:
    
    1. **Focus on the most important factors first** - especially Temperature and NPK levels
    2. **Measure these factors in your soil** - soil testing kits are available at garden centers
    3. **Choose crops that match your conditions** - or adjust your soil to match crop needs
    
    Remember: It's easier to choose the right crop for your conditions than to change your conditions for a crop!
    """)
    
    # Add crop comparison functionality
    st.subheader("Crop Comparison Tool")
    st.write("Compare optimal growing conditions for different crops")
    
    # Select crops to compare (multi-select)
    col1, col2 = st.columns(2)
    
    with col1:
        # Comprehensive crop data for visualization
        crop_data = {
            'Rice': {'N': 80, 'P': 40, 'K': 40, 'Temperature': 25, 'pH': 6.5, 'Humidity': 80, 'Rainfall': 200},
            'Wheat': {'N': 60, 'P': 30, 'K': 30, 'Temperature': 20, 'pH': 7.0, 'Humidity': 65, 'Rainfall': 100},
            'Maize': {'N': 70, 'P': 35, 'K': 40, 'Temperature': 23, 'pH': 6.8, 'Humidity': 70, 'Rainfall': 150},
            'Potato': {'N': 90, 'P': 45, 'K': 60, 'Temperature': 18, 'pH': 6.0, 'Humidity': 75, 'Rainfall': 120},
            'Cotton': {'N': 50, 'P': 25, 'K': 25, 'Temperature': 28, 'pH': 6.2, 'Humidity': 60, 'Rainfall': 80},
            'Sugarcane': {'N': 100, 'P': 50, 'K': 50, 'Temperature': 27, 'pH': 6.5, 'Humidity': 85, 'Rainfall': 220},
            'Tomato': {'N': 85, 'P': 45, 'K': 55, 'Temperature': 24, 'pH': 6.3, 'Humidity': 70, 'Rainfall': 90},
            'Barley': {'N': 55, 'P': 25, 'K': 35, 'Temperature': 19, 'pH': 6.8, 'Humidity': 60, 'Rainfall': 110},
            'Soybean': {'N': 65, 'P': 40, 'K': 40, 'Temperature': 22, 'pH': 6.5, 'Humidity': 70, 'Rainfall': 130},
            'Cabbage': {'N': 75, 'P': 35, 'K': 45, 'Temperature': 17, 'pH': 6.2, 'Humidity': 75, 'Rainfall': 100},
            'Onion': {'N': 70, 'P': 40, 'K': 45, 'Temperature': 20, 'pH': 6.5, 'Humidity': 65, 'Rainfall': 90},
            'Turmeric': {'N': 60, 'P': 30, 'K': 35, 'Temperature': 25, 'pH': 6.3, 'Humidity': 80, 'Rainfall': 180},
            'Chilly': {'N': 75, 'P': 35, 'K': 40, 'Temperature': 26, 'pH': 6.5, 'Humidity': 70, 'Rainfall': 100},
            'Okra': {'N': 65, 'P': 35, 'K': 40, 'Temperature': 25, 'pH': 6.7, 'Humidity': 75, 'Rainfall': 120},
            'Sunflower': {'N': 55, 'P': 35, 'K': 30, 'Temperature': 24, 'pH': 6.8, 'Humidity': 60, 'Rainfall': 90}
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
            options=['N', 'P', 'K', 'Temperature', 'pH', 'Humidity', 'Rainfall'],
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
        
        st.divider()
        st.subheader("Crop Parameter Parallel Coordinates")
        st.write("This parallel coordinates plot allows you to see how each crop performs across multiple parameters simultaneously. It helps visualize the relationships between different crop requirements.")
        
        # Create a DataFrame suitable for parallel coordinates plot
        parallel_data = []
        for crop in selected_crops:
            crop_row = {'Crop': crop}
            for param in selected_params:
                crop_row[param] = crop_data[crop][param]
            parallel_data.append(crop_row)
        
        df_parallel = pd.DataFrame(parallel_data)
        
        # Create parallel coordinates plot
        fig_parallel = px.parallel_coordinates(
            df_parallel,
            color="Crop",
            labels={col: col for col in df_parallel.columns},
            color_continuous_scale=px.colors.diverging.Tealrose,
            color_continuous_midpoint=2,
            title="Crop Parameters Comparison - Parallel Coordinates",
        )
        
        fig_parallel.update_layout(
            font=dict(size=12),
            height=550,
            margin=dict(l=80, r=80, t=80, b=50),
        )
        
        st.plotly_chart(fig_parallel, use_container_width=True)
        
        # Add interactive heatmap for correlation between parameters
        st.divider()
        st.subheader("Parameter Correlation Heatmap")
        st.write("This heatmap shows the correlation between different parameters across the selected crops.")
        
        # Prepare correlation data
        if len(selected_params) > 1:
            corr_data = df_parallel.drop('Crop', axis=1).corr().round(2)
            
            # Create heatmap
            fig_heatmap = px.imshow(
                corr_data,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Parameter Correlation Heatmap"
            )
            
            fig_heatmap.update_layout(
                height=400,
                margin=dict(l=60, r=60, t=80, b=50)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.info("💡 **How to read this:** Values closer to 1 mean strong positive correlation (parameters increase together), values closer to -1 mean strong negative correlation (as one increases, the other decreases), and values close to 0 mean little to no correlation.")

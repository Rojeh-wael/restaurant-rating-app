import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======== PAGE CONFIGURATION ========
st.set_page_config(
    page_title="Restaurant Rating Analysis & Prediction",
    page_icon="🍽️",
    layout="wide"
)

# ======== LOAD MODELS & PREPROCESSORS ========
@st.cache_resource
def load_ml_model():
    return joblib.load("best_model.pkl")

@st.cache_resource
def load_nn_model():
    return tf.keras.models.load_model("my_model.keras")

@st.cache_resource
def load_encoder():
    return joblib.load('encoder.pkl')

@st.cache_resource
def load_ord_enc():
    return joblib.load('ord_enc.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

@st.cache_data
def load_dataset():
    df = pd.read_csv("raw.csv")  # replace with your dataset path
    # Normalize column names: lowercase & strip spaces
    df.columns = df.columns.str.strip().str.lower()
    return df

# ======== LOAD DATA AND MODELS ========
try:
    df = load_dataset()
    best_model = load_ml_model()
    nn_model = load_nn_model()
    encoder = load_encoder()
    scaler = load_scaler()
    ord = load_ord_enc()
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ======== SIDEBAR NAVIGATION ========
st.sidebar.title("🍽️ Restaurant Analytics")
page = st.sidebar.selectbox("Choose a page", ["📊 Analysis Dashboard", "🔮 Prediction Tool"])

# ======== ANALYSIS PAGE ========
if page == "📊 Analysis Dashboard":
    st.title("📊 Restaurant Data Analysis Dashboard")
    st.markdown("---")
    
    # Dataset Overview
    st.header("🔍 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Restaurants", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Unique Locations", df['location'].nunique())
    
    # Show first few rows
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # ======== VISUALIZATIONS ========
    st.header("📈 Data Visualizations")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution Analysis", "Categorical Analysis", "Correlation Analysis", "Location Analysis"])
    
    with tab1:
        st.subheader("Distribution of Numerical Features")
        
        # Rating distribution (if you have a rating column)
        if 'rate' in df.columns:
            fig_rating = px.histogram(df, x='rate', title='Distribution of Restaurant Ratings',
                                    nbins=20, color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig_rating, use_container_width=True)
        
        # Cost distribution
        if 'cost2plates' in df.columns:
            fig_cost = px.histogram(df, x='cost2plates', title='Distribution of Cost for 2 Plates',
                                  nbins=30, color_discrete_sequence=['#4ECDC4'])
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Votes distribution
        if 'votes' in df.columns:
            fig_votes = px.histogram(df, x='votes', title='Distribution of Votes',
                                   nbins=25, color_discrete_sequence=['#45B7D1'])
            st.plotly_chart(fig_votes, use_container_width=True)
    
    with tab2:
        st.subheader("Categorical Feature Analysis")
        
        # Online Order Analysis
        if 'online_order' in df.columns:
            fig_online = px.pie(df, names='online_order', title='Online Order Distribution',
                              color_discrete_sequence=['#FF9999', '#66B2FF'])
            st.plotly_chart(fig_online, use_container_width=True)
        
        # Restaurant Type Analysis - Your specific analysis
        if 'rest_type' in df.columns:
            rest_type_count = df['rest_type'].value_counts().head(10).reset_index()
            rest_type_count.columns = ['rest_type', 'count']  # Ensure proper column names
            
            fig_rest_type = px.bar(rest_type_count,
                                 x='count',
                                 y='rest_type',
                                 title='Most common type restaurants',
                                 orientation='h',
                                 color='count',
                                 color_continuous_scale='viridis')
            fig_rest_type.update_layout(
                xaxis_title='Count of type',
                yaxis_title='Restaurant type',
                title_font_size=20,
                font_size=12,
                height=600
            )
            fig_rest_type.update_yaxes(categoryorder='total ascending')  # Order by count
            st.plotly_chart(fig_rest_type, use_container_width=True)
        
        # Table Booking Analysis
        if 'book_table' in df.columns:
            fig_table = px.pie(df, names='book_table', title='Table Booking Distribution',
                             color_discrete_sequence=['#FFB366', '#FF6B9D'])
            st.plotly_chart(fig_table, use_container_width=True)
        
        # Restaurant Service Type Analysis - Your specific pie chart
        if 'type' in df.columns:
            common_type = df['type'].value_counts().head(5)
            
            # Create explode effect for the most common type
            explode_values = [0.1 if val == common_type.max() else 0 for val in common_type]
            
            # Create pie chart using plotly
            fig_service_type = go.Figure(data=[go.Pie(
                labels=common_type.index,
                values=common_type.values,
                hole=0,  # Set to 0 for full pie chart
                textinfo='label+percent',
                textposition='auto',
                marker=dict(
                    colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'],  # Set2 color palette
                ),
                pull=explode_values  # This creates the explode effect
            )])
            
            fig_service_type.update_layout(
                title='Common Types of Restaurant Service',
                title_font_size=16,
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig_service_type, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Select only numerical columns for correlation
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            
            # Create heatmap using plotly
            fig_corr = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="Correlation Matrix of Numerical Features",
                               color_continuous_scale='RdBu')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Cost vs Rating scatter plot (if both columns exist)
        if 'cost2plates' in df.columns and 'rate' in df.columns:
            fig_scatter = px.scatter(df, x='cost2plates', y='rate', 
                                   title='Cost vs Rating',
                                   opacity=0.6,
                                   color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.subheader("Location-based Analysis")
        
        if 'location' in df.columns:
            # Top locations by count - Your specific analysis
            locations = df['location'].value_counts().head(10).reset_index()
            locations.columns = ['location', 'count']  # Ensure proper column names
            
            fig_location = px.bar(locations, 
                                x='count', 
                                y='location',
                                title='No of restaurants in top 10 locations',
                                orientation='h',
                                color='count',
                                color_continuous_scale='viridis')
            fig_location.update_layout(
                xaxis_title='Restaurants count',
                yaxis_title='Locations',
                title_font_size=20,
                font_size=12,
                height=600
            )
            fig_location.update_yaxes(categoryorder='total ascending')  # Order by count
            st.plotly_chart(fig_location, use_container_width=True)
            
            # Online Order Availability by Location - Your specific analysis
            if 'online_order' in df.columns:
                online_order_locations = pd.crosstab(df['location'], df['online_order'])
                
                # Convert crosstab to format suitable for plotly
                online_order_locations_reset = online_order_locations.reset_index()
                
                # Create grouped bar chart
                fig_online_locations = px.bar(
                    online_order_locations_reset.melt(id_vars='location', 
                                                    var_name='online_order', 
                                                    value_name='count'),
                    x='location',
                    y='count',
                    color='online_order',
                    title='Online order availability',
                    barmode='group',
                    color_discrete_map={'Yes': '#2E8B57', 'No': '#DC143C'}
                )
                
                fig_online_locations.update_layout(
                    xaxis_title='Location',
                    yaxis_title='Count',
                    title_font_size=16,
                    font_size=10,
                    height=600,
                    legend_title_text='Online Order Available'
                )
                fig_online_locations.update_xaxes(tickangle=45)
                st.plotly_chart(fig_online_locations, use_container_width=True)
            
            # Table Booking Availability by Location - Your specific analysis
            if 'book_table' in df.columns:
                table_book_avail = pd.crosstab(df['location'], df['book_table'])
                
                # Convert crosstab to format suitable for plotly
                table_book_avail_reset = table_book_avail.reset_index()
                
                # Create grouped bar chart
                fig_table_locations = px.bar(
                    table_book_avail_reset.melt(id_vars='location', 
                                              var_name='book_table', 
                                              value_name='count'),
                    x='location',
                    y='count',
                    color='book_table',
                    title='Book table availability in different locations',
                    barmode='group',
                    color_discrete_map={'Yes': '#4169E1', 'No': '#FF6347'}
                )
                
                fig_table_locations.update_layout(
                    xaxis_title='Location',
                    yaxis_title='Count',
                    title_font_size=16,
                    font_size=10,
                    height=600,
                    legend_title_text='Table Booking Available'
                )
                fig_table_locations.update_xaxes(tickangle=45)
                st.plotly_chart(fig_table_locations, use_container_width=True)
            
            # Distribution of Different Types of Places in Different Locations - Your specific analysis
            if 'location' in df.columns and 'type' in df.columns:
                # Get top 10 locations for better visualization
                top_10_locations = df['location'].value_counts().head(10).index
                df_filtered = df[df['location'].isin(top_10_locations)]
                
                # Create count plot equivalent using plotly
                fig_location_type = px.histogram(
                    df_filtered,
                    x='location',
                    color='type',
                    title='Distribution of Different Types of Places in Different Locations',
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig_location_type.update_layout(
                    xaxis_title='Location',
                    yaxis_title='Count',
                    title_font_size=16,
                    font_size=10,
                    height=700,
                    legend_title_text='Type of Place'
                )
                fig_location_type.update_xaxes(tickangle=80)
                st.plotly_chart(fig_location_type, use_container_width=True)
            
            # Average cost by location (if cost column exists)
            if 'cost2plates' in df.columns:
                avg_cost_location = df.groupby('location')['cost2plates'].mean().sort_values(ascending=False).head(10)
                fig_cost_loc = px.bar(x=avg_cost_location.index, y=avg_cost_location.values,
                                    title='Average Cost by Location (Top 10)',
                                    color=avg_cost_location.values,
                                    color_continuous_scale='viridis')
                fig_cost_loc.update_layout(xaxis_title='Location', yaxis_title='Average Cost for 2 Plates')
                fig_cost_loc.update_xaxes(tickangle=45)
                st.plotly_chart(fig_cost_loc, use_container_width=True)
    
    # ======== KEY INSIGHTS ========
    st.header("💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.info("**Dataset Insights:**")
        st.write(f"• Total restaurants analyzed: {len(df):,}")
        if 'location' in df.columns:
            st.write(f"• Number of unique locations: {df['location'].nunique()}")
        if 'cost2plates' in df.columns:
            st.write(f"• Average cost for 2 plates: $ {df['cost2plates'].mean():.2f}")
        if 'votes' in df.columns:
            st.write(f"• Average votes per restaurant: {df['votes'].mean():.1f}")
    
    with insights_col2:
        st.success("**Business Insights:**")
        if 'online_order' in df.columns:
            online_pct = (df['online_order'].value_counts(normalize=True).get('Yes', 0) * 100)
            st.write(f"• {online_pct:.1f}% restaurants offer online ordering")
        if 'book_table' in df.columns:
            booking_pct = (df['book_table'].value_counts(normalize=True).get('Yes', 0) * 100)
            st.write(f"• {booking_pct:.1f}% restaurants allow table booking")
        if 'rest_type' in df.columns:
            top_type = df['rest_type'].value_counts().index[0]
            st.write(f"• Most common restaurant type: {top_type}")

# ======== PREDICTION PAGE ========
elif page == "🔮 Prediction Tool":
    st.title("🔮 Restaurant Rating Prediction")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Restaurant Rating Predictor! 
    Fill in the details below to predict the rating category of a restaurant.
    """)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Restaurant Details")
        
        with st.form("prediction_form"):
            online_order = st.selectbox("Online Order Available?", 
                                      sorted(df["online_order"].dropna().unique()),
                                      help="Does the restaurant offer online ordering?")
            
            book_table = st.selectbox("Table Booking Available?", 
                                    sorted(df["book_table"].dropna().unique()),
                                    help="Can customers book tables in advance?")
            
            votes = st.number_input("Number of Votes", 
                                  min_value=0, 
                                  value=100,
                                  help="Total number of customer votes/reviews")
            
            location = st.selectbox("Location", 
                                  sorted(df["location"].dropna().unique()),
                                  help="Restaurant location/area")
            
            rest_type = st.selectbox("Restaurant Type", 
                                   sorted(df["rest_type"].dropna().unique()),
                                   help="Type/category of the restaurant")
            
            cuisines = st.selectbox("Cuisines", 
                                  sorted(df["cuisines"].dropna().unique()),
                                  help="Type of cuisine served")
            
            cost2plates = st.number_input("Cost for 2 Plates ($)", 
                                        min_value=50, 
                                        max_value=10000, 
                                        value=500,
                                        help="Approximate cost for two people")
            
            restaurant_type = st.selectbox("Service Type", 
                                         sorted(df["type"].dropna().unique()),
                                         help="Dining type (Delivery, Dine-out, etc.)")
            
            submitted = st.form_submit_button("🎯 Predict Rating", use_container_width=True)
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        # Feature definitions
        num_feat = ['votes', 'cost2plates']
        bin_feat = ['online_order', 'book_table']
        cat_feat = ['location', 'rest_type', 'cuisines', 'type']
        
        # Rating mapping
        rate_map = {
            0: "Poor",
            1: "Good", 
            2: "Excellent"
        }
        
        if submitted:
            try:
                # Create DataFrame for single row
                input_df = pd.DataFrame([{
                    "online_order": online_order,
                    "book_table": book_table,
                    "votes": votes,
                    "location": location,
                    "rest_type": rest_type,
                    "cuisines": cuisines,
                    "cost2plates": cost2plates,
                    "type": restaurant_type
                }])
                
                # Transform features
                ohe_test = encoder.transform(input_df[cat_feat])
                ohe_test_df = pd.DataFrame(ohe_test, columns=encoder.get_feature_names_out(cat_feat), index=input_df.index)
                
                ord_test = ord.transform(input_df[bin_feat])
                ord_test_df = pd.DataFrame(ord_test, columns=bin_feat, index=input_df.index)
                
                num_test = scaler.transform(input_df[num_feat])
                num_test_df = pd.DataFrame(num_test, columns=num_feat, index=input_df.index)
                
                # Combine all features
                final_input = pd.concat([ohe_test_df, ord_test_df, num_test_df], axis=1)
                
                # Make predictions
                ml_pred = best_model.predict(final_input)
                nn_pred = nn_model.predict(final_input)
                nn_pred_class = np.argmax(nn_pred, axis=1)
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # ML Model Result
                st.markdown("### 🤖 Machine Learning Model")
                ml_result = rate_map[int(ml_pred[0])]
                if ml_result == "Poor":
                    st.error(f"**Rating: {ml_result}** 😞")
                elif ml_result == "Good":
                    st.warning(f"**Rating: {ml_result}** 😊")
                else:
                    st.success(f"**Rating: {ml_result}** 🌟")
                
                # Neural Network Result
                st.markdown("### 🧠 Neural Network Model")
                nn_result = rate_map[int(nn_pred_class[0])]
                if nn_result == "Poor":
                    st.error(f"**Rating: {nn_result}** 😞")
                elif nn_result == "Good":
                    st.warning(f"**Rating: {nn_result}** 😊")
                else:
                    st.success(f"**Rating: {nn_result}** 🌟")
                
                # Confidence scores for NN
                st.markdown("### 📈 Neural Network Confidence")
                confidence_df = pd.DataFrame({
                    'Rating Category': ['Poor', 'Good', 'Excellent'],
                    'Confidence': nn_pred[0] * 100
                })
                
                fig_confidence = px.bar(confidence_df, x='Rating Category', y='Confidence',
                                      title='Prediction Confidence Scores',
                                      color='Confidence',
                                      color_continuous_scale='viridis')
                fig_confidence.update_layout(yaxis_title='Confidence (%)')
                st.plotly_chart(fig_confidence, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please check that all model files are available and properly trained.")
        
        else:
            st.info("👆 Fill in the form and click 'Predict Rating' to see results")
            
            # Show some example statistics while waiting
            if not df.empty:
                st.markdown("### 📊 Dataset Statistics")
                if 'cost2plates' in df.columns:
                    st.metric("Average Cost ($)", f"${df['cost2plates'].mean():.0f}")
                if 'votes' in df.columns:
                    st.metric("Average Votes", f"{df['votes'].mean():.0f}")
                st.metric("Total Restaurants", len(df))

# ======== FOOTER ========
st.markdown("---")

st.markdown("Built By Rojeh_Wael using Streamlit | Restaurant Analytics Dashboard")

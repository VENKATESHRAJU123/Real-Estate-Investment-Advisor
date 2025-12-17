"""
Real Estate Investment Advisor - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        classification_model = joblib.load('../models/classification_model.pkl')
        regression_model = joblib.load('../models/regression_model.pkl')
        return classification_model, regression_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please train the models first by running the training notebook.")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Real Estate Investment Advisor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict Property Profitability & Future Value")
    st.markdown("---")
    
    # Load models
    class_model, reg_model = load_models()
    
    if class_model is None or reg_model is None:
        st.stop()
    
    # Sidebar inputs
    st.sidebar.header("🏡 Property Details")
    st.sidebar.markdown("Fill in the property information below:")
    
    # Property characteristics
    st.sidebar.subheader("Basic Information")
    bhk = st.sidebar.slider("BHK (Bedrooms)", 1, 5, 2)
    size_sqft = st.sidebar.number_input("Size (Sq Ft)", 500, 5000, 1000, step=100)
    price_lakhs = st.sidebar.number_input("Current Price (Lakhs)", 10, 500, 50, step=5)
    year_built = st.sidebar.slider("Year Built", 2000, 2024, 2015)
    
    # Location and structure
    st.sidebar.subheader("Location & Structure")
    floor_no = st.sidebar.number_input("Floor Number", 0, 50, 5)
    total_floors = st.sidebar.number_input("Total Floors", 1, 50, 10)
    
    # Amenities
    st.sidebar.subheader("Amenities & Infrastructure")
    nearby_schools = st.sidebar.slider("Nearby Schools", 0, 10, 3)
    nearby_hospitals = st.sidebar.slider("Nearby Hospitals", 0, 10, 2)
    transport_access = st.sidebar.slider("Public Transport Access (1-10)", 1, 10, 7)
    parking = st.sidebar.slider("Parking Spaces", 0, 5, 1)
    
    # Calculate derived features
    price_per_sqft = (price_lakhs * 100000) / size_sqft
    age_of_property = 2025 - year_built
    infrastructure_score = (nearby_schools * 0.3 + nearby_hospitals * 0.3 + transport_access * 0.4)
    school_density_score = nearby_schools / (age_of_property + 1)
    floor_ratio = floor_no / (total_floors + 1)
    amenities_count = 3  # Placeholder
    
    # Prepare input data
    input_data = pd.DataFrame({
        'BHK': [bhk],
        'Size_in_SqFt': [size_sqft],
        'Price_in_Lakhs': [price_lakhs],
        'Price_per_SqFt': [price_per_sqft],
        'Year_Built': [year_built],
        'Floor_No': [floor_no],
        'Total_Floors': [total_floors],
        'Age_of_Property': [age_of_property],
        'Nearby_Schools': [nearby_schools],
        'Nearby_Hospitals': [nearby_hospitals],
        'Public_Transport_Accessibility': [transport_access],
        'Parking_Space': [parking],
        'Infrastructure_Score': [infrastructure_score],
        'Amenities_Count': [amenities_count],
        'School_Density_Score': [school_density_score],
        'Floor_Ratio': [floor_ratio]
    })
    
    # Analyze button
    if st.sidebar.button("🔍 Analyze Investment", type="primary"):
        
        # Main content area
        col1, col2 = st.columns(2)
        
        # Classification prediction
        with col1:
            st.subheader("📊 Investment Classification")
            
            try:
                good_investment = class_model.predict(input_data)[0]
                confidence = class_model.predict_proba(input_data)[0]
                
                if good_investment == 1:
                    st.success("✅ **Good Investment**")
                    st.metric("Confidence Score", f"{confidence[1]*100:.1f}%")
                else:
                    st.error("❌ **Not Recommended**")
                    st.metric("Risk Score", f"{confidence[0]*100:.1f}%")
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence[1]*100,
                    title={'text': "Investment Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen" if good_investment == 1 else "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightcoral"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Classification error: {e}")
        
        # Regression prediction
        with col2:
            st.subheader("💰 Price Prediction")
            
            try:
                future_price = reg_model.predict(input_data)[0]
                growth_rate = ((future_price / price_lakhs) - 1) * 100
                
                st.metric(
                    "Estimated Price (5 Years)",
                    f"₹{future_price:.2f} Lakhs",
                    delta=f"+{growth_rate:.1f}%"
                )
                
                # Price trend chart
                years = list(range(0, 6))
                prices = [price_lakhs * ((1 + (growth_rate/100/5)) ** year) for year in years]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years,
                    y=prices,
                    mode='lines+markers',
                    name='Projected Price',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="5-Year Price Projection",
                    xaxis_title="Years from Now",
                    yaxis_title="Price (Lakhs)",
                    hovermode='x unified',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Regression error: {e}")
        
        # Investment insights
        st.markdown("---")
        st.subheader("📈 Investment Insights")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Key factors
            st.markdown("#### Key Investment Factors")
            
            factors = {
                'Infrastructure Score': infrastructure_score,
                'Property Age Factor': max(0, 10 - age_of_property),
                'Size Factor': min(10, size_sqft / 500),
                'Location Factor': transport_access
            }
            
            fig = px.bar(
                x=list(factors.values()),
                y=list(factors.keys()),
                orientation='h',
                title="Property Score Factors",
                labels={'x': 'Score', 'y': 'Factor'}
            )
            fig.update_traces(marker_color='skyblue')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Recommendations
            st.markdown("#### Recommendations")
            
            recommendations = []
            
            if age_of_property < 5:
                recommendations.append("✅ Modern construction - lower maintenance costs")
            if infrastructure_score > 5:
                recommendations.append("✅ Excellent infrastructure connectivity")
            if price_per_sqft < 5000:
                recommendations.append("✅ Competitive price per square foot")
            if bhk >= 2:
                recommendations.append("✅ Family-friendly configuration")
            if parking >= 1:
                recommendations.append("✅ Adequate parking available")
            
            if not recommendations:
                recommendations.append("⚠️ Consider reviewing property features")
            
            for rec in recommendations:
                st.markdown(rec)
        
        # Detailed analysis
        st.markdown("---")
        with st.expander("📋 Detailed Property Analysis"):
            st.markdown("### Property Summary")
            
            summary_df = pd.DataFrame({
                'Feature': [
                    'BHK', 'Size (Sq Ft)', 'Current Price (Lakhs)',
                    'Price per Sq Ft', 'Age (Years)', 'Floor',
                    'Infrastructure Score', 'School Density'
                ],
                'Value': [
                    bhk, size_sqft, price_lakhs, f"₹{price_per_sqft:.2f}",
                    age_of_property, f"{floor_no}/{total_floors}",
                    f"{infrastructure_score:.2f}", f"{school_density_score:.2f}"
                ]
            })
            
            st.table(summary_df)

if __name__ == "__main__":
    main()

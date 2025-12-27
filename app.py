import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = "Motor_Vehicle_Collisions_-_Crashes.csv"

st.title("NYC Vehicle Collisions Dashboard")
st.markdown("**Simple Working Dashboard**")

@st.cache_data(persist=True)
def load_data(nrows):
    # Simply read the CSV and clean column names
    data = pd.read_csv(DATA_URL, nrows=nrows)
    
    # Debug: Show original columns
    st.sidebar.write("Original columns:", list(data.columns)[:15])
    
    # CLEAN ALL COLUMN NAMES - Remove spaces, make lowercase
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    
    st.sidebar.write("Cleaned columns:", list(data.columns)[:15])
    
    # Try to find date and time columns by pattern
    date_col = None
    time_col = None
    for col in data.columns:
        if 'date' in col and not date_col:
            date_col = col
        if 'time' in col and not time_col:
            time_col = col
    
    # Create datetime if possible
    if date_col and time_col:
        try:
            data['datetime'] = pd.to_datetime(data[date_col].astype(str) + ' ' + data[time_col].astype(str), errors='coerce')
        except:
            data['datetime'] = pd.to_datetime(data[date_col], errors='coerce')
    elif date_col:
        data['datetime'] = pd.to_datetime(data[date_col], errors='coerce')
    
    # Find latitude/longitude columns
    lat_col = None
    lon_col = None
    for col in data.columns:
        if 'lat' in col and not lat_col:
            lat_col = col
        if 'lon' in col and not lon_col:
            lon_col = col
    
    # Rename for consistency
    if lat_col:
        data = data.rename(columns={lat_col: 'latitude'})
    if lon_col:
        data = data.rename(columns={lon_col: 'longitude'})
    
    # Find injury column
    injury_col = None
    for col in data.columns:
        if 'injured' in col and 'person' in col:
            injury_col = col
            break
        if 'injured' in col:
            injury_col = col
    
    if injury_col:
        data = data.rename(columns={injury_col: 'injured_persons'})
        # Convert to numeric, fill NaN with 0
        data['injured_persons'] = pd.to_numeric(data['injured_persons'], errors='coerce').fillna(0)
    
    # Standardize factor column names - use only VEHICLE_1 and VEHICLE_2
    factor_columns = {}
    for col in data.columns:
        if 'contributing_factor' in col:
            # Find which vehicle number this is
            for i in range(1, 6):  # Check for numbers 1-5
                if f'vehicle_{i}' in col or f'vehicle_{i}_' in col:
                    # Only keep vehicle 1 and 2
                    if i in [1, 2]:
                        # Standardize to contributing_factor_vehicle_1 or contributing_factor_vehicle_2
                        new_name = f'contributing_factor_vehicle_{i}'
                        factor_columns[col] = new_name
                    break
    
    # Rename the factor columns
    if factor_columns:
        data = data.rename(columns=factor_columns)
        st.sidebar.write(f"Renamed factor columns: {list(factor_columns.values())}")
    
    # Drop rows without coordinates if we have them
    if 'latitude' in data.columns and 'longitude' in data.columns:
        data = data.dropna(subset=['latitude', 'longitude'])
        # Convert to numeric
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    
    st.sidebar.success(f"Loaded {len(data)} valid rows")
    return data

# Load the data
with st.spinner('Loading data...'):
    data = load_data(50000)

# Show what columns we actually have
st.sidebar.subheader("Available Columns")
st.sidebar.write([col for col in data.columns if not col.startswith('unnamed')])

# ===== VISUALIZATION 1: Simple Map with Streamlit's built-in map =====
st.header("1. Injury Locations Map")

if 'injured_persons' in data.columns and 'latitude' in data.columns and 'longitude' in data.columns:
    # Create a simple filter
    min_injuries = st.slider("Minimum injuries per incident", 0, int(data['injured_persons'].max()), 1)
    
    # Filter data
    filtered = data[data['injured_persons'] >= min_injuries][['latitude', 'longitude']].dropna()
    
    if not filtered.empty:
        # Use Streamlit's simple map (more reliable than pydeck)
        st.map(filtered.rename(columns={'latitude': 'lat', 'longitude': 'lon'}))
        st.write(f"Showing {len(filtered)} locations with {min_injuries}+ injuries")
    else:
        st.info("No data matches the filter. Try a lower injury count.")
else:
    missing = []
    if 'injured_persons' not in data.columns:
        missing.append("'injured_persons'")
    if 'latitude' not in data.columns:
        missing.append("'latitude'")
    if 'longitude' not in data.columns:
        missing.append("'longitude'")
    st.warning(f"Cannot show map. Missing columns: {', '.join(missing)}")

# ===== VISUALIZATION 2: Temporal Chart - Collisions Over Time =====
st.header("2. Collisions Over Time")

if 'datetime' in data.columns:
    # Extract date
    data['date'] = data['datetime'].dt.date
    
    # Count collisions per day
    daily_counts = data['date'].value_counts().sort_index().reset_index()
    daily_counts.columns = ['date', 'collisions']
    
    # Line chart
    fig = px.line(daily_counts, x='date', y='collisions', 
                  title='Daily Vehicle Collisions in NYC',
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add hourly analysis
    st.subheader("Collisions by Hour of Day")
    hour = st.slider("Select hour to analyze", 0, 23, 17)
    
    hour_data = data[data['datetime'].dt.hour == hour]
    st.write(f"**{len(hour_data)} collisions occurred between {hour}:00 and {hour+1}:00**")
    
    # Show hour distribution
    if 'datetime' in data.columns:
        data['hour'] = data['datetime'].dt.hour
        hourly_counts = data['hour'].value_counts().sort_index().reset_index()
        hourly_counts.columns = ['hour', 'collisions']
        
        fig2 = px.bar(hourly_counts, x='hour', y='collisions',
                     title='Collisions by Hour of Day (All Data)',
                     labels={'hour': 'Hour of Day (0-23)'})
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Cannot show temporal charts: No datetime column found.")

# ===== VISUALIZATION 3: Contributing Factor Analysis =====
st.header("3. Contributing Factor Analysis")

# Check if we have the factor columns
factor_cols = [col for col in data.columns if 'contributing_factor_vehicle_' in col]

if factor_cols:
    # Only use vehicle 1 and 2
    available_cols = []
    for i in [1, 2]:
        col_name = f'contributing_factor_vehicle_{i}'
        if col_name in data.columns:
            available_cols.append(col_name)
    
    if available_cols:
        # Let user select which factor column to analyze
        selected_factor = st.selectbox("Select factor column to analyze:", available_cols)
        
        if selected_factor in data.columns:
            # Clean up factor data
            factor_data = data[selected_factor].fillna('Unknown').replace('', 'Unspecified')
            
            # Get top factors
            top_factors = factor_data.value_counts().head(15).reset_index()
            top_factors.columns = ['factor', 'count']
            
            # Filter out "Unspecified" and "Unknown" if they dominate
            top_factors = top_factors[~top_factors['factor'].isin(['Unspecified', 'Unknown', 'unspecified', 'unknown'])]
            
            if not top_factors.empty:
                fig3 = px.bar(top_factors, x='factor', y='count',
                             title=f'Top Contributing Factors ({selected_factor})',
                             color='count',
                             labels={'factor': 'Contributing Factor', 'count': 'Number of Collisions'})
                fig3.update_xaxes(tickangle=45)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Show comparison between vehicle 1 and 2 factors if both available
                if len(available_cols) >= 2:
                    st.subheader("Comparison: Vehicle 1 vs Vehicle 2 Factors")
                    
                    # Get top factors for each
                    top_factors_dict = {}
                    for col in available_cols[:2]:  # Only first 2
                        factors = data[col].fillna('Unknown').replace('', 'Unspecified')
                        top_5 = factors.value_counts().head(5).reset_index()
                        top_5.columns = ['factor', 'count']
                        top_factors_dict[col] = top_5
                    
                    # Display side by side
                    cols = st.columns(len(top_factors_dict))
                    for idx, (col_name, factor_df) in enumerate(top_factors_dict.items()):
                        with cols[idx]:
                            st.write(f"**{col_name}**")
                            st.dataframe(factor_df)
            else:
                st.info(f"No significant factor data found for {selected_factor} (mostly 'Unspecified' or 'Unknown')")
    else:
        st.info("No contributing factor columns (vehicle 1 or 2) found in the data")
else:
    st.info("No contributing factor columns available for analysis")

# ===== VISUALIZATION 4: Heatmap =====
st.header("4. Collisions Heatmap")

if 'datetime' in data.columns and 'latitude' in data.columns and 'longitude' in data.columns:
    # Simple density heatmap
    fig4 = px.density_mapbox(data, lat='latitude', lon='longitude', 
                            radius=10,
                            zoom=10,
                            mapbox_style="carto-positron",
                            title='Collision Density Map')
    st.plotly_chart(fig4, use_container_width=True)

# ===== VISUALIZATION 5: Factor and Time Relationship =====
st.header("5. Factors by Time of Day")

if 'datetime' in data.columns and any('contributing_factor_vehicle_' in col for col in data.columns):
    # Find available factor columns
    factor_cols = [col for col in data.columns if 'contributing_factor_vehicle_1' in col or 
                   'contributing_factor_vehicle_2' in col]
    
    if factor_cols:
        selected_factor_col = st.selectbox("Select factor for time analysis:", factor_cols, key='time_factor')
        
        # Add hour column if not exists
        if 'hour' not in data.columns and 'datetime' in data.columns:
            data['hour'] = data['datetime'].dt.hour
        
        if 'hour' in data.columns:
            # Get top 3 factors for this column
            top_factors = data[selected_factor_col].fillna('Unknown').value_counts().head(3).index.tolist()
            
            if top_factors:
                # Filter data for top factors
                filtered_data = data[data[selected_factor_col].isin(top_factors)]
                
                if not filtered_data.empty:
                    # Group by hour and factor
                    hourly_factor = filtered_data.groupby(['hour', selected_factor_col]).size().reset_index(name='count')
                    
                    fig5 = px.line(hourly_factor, x='hour', y='count', color=selected_factor_col,
                                  title=f'Hourly Pattern for Top Factors ({selected_factor_col})',
                                  markers=True,
                                  labels={'hour': 'Hour of Day', 'count': 'Number of Collisions'})
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.info(f"No data available for the top factors in {selected_factor_col}")
            else:
                st.info(f"No significant factors found in {selected_factor_col}")

# ===== RAW DATA OPTION =====
if st.checkbox("Show Sample Data"):
    st.subheader("Sample of Loaded Data")
    
    # Create a sample that includes factor columns if they exist
    sample_cols = ['datetime', 'latitude', 'longitude', 'injured_persons']
    factor_cols = [col for col in data.columns if 'contributing_factor_vehicle_' in col]
    sample_cols.extend(factor_cols[:2])  # Add first 2 factor columns
    
    # Filter to only columns that exist
    sample_cols = [col for col in sample_cols if col in data.columns]
    
    if sample_cols:
        st.dataframe(data[sample_cols].head(100))
    else:
        st.dataframe(data.head(100))

# ===== DEBUG INFO =====
with st.expander("Debug Information"):
    st.write("Data shape:", data.shape)
    st.write("Column dtypes:", data.dtypes.to_dict())
    if 'datetime' in data.columns:
        st.write("Date range:", data['datetime'].min(), "to", data['datetime'].max())
    
    # Show column statistics
    st.write("### Column Summary")
    for col in data.columns:
        if col not in ['latitude', 'longitude', 'datetime']:
            non_null = data[col].notna().sum()
            st.write(f"**{col}**: {data[col].dtype}, {non_null} non-null values")
            
            # Show unique values for factor columns
            if 'contributing_factor_vehicle_' in col:
                unique_vals = data[col].dropna().unique()[:10]
                st.write(f"  Sample values: {list(unique_vals)[:5]}")

st.markdown("---")
st.markdown("**Dashboard created for NYC Vehicle Collisions Analysis**")
st.markdown("*Analyzing contributing factors from vehicles 1 & 2*")
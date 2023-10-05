# to run: cd into oauth2-proxy-v7.5.1.darwin-amd64 
# cmd line: ./oauth2-proxy --config proxy.cfg
import streamlit as st

# from streamlit.scriptrunner import get_script_run_ctx 
# from streamlit.script_run_context import get_script_run_ctx
# github issue #5457 streamlit
from streamlit.web.server.websocket_headers import _get_websocket_headers
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import calendar
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz
from fuzzywuzzy import process




# Streamlit app
st.set_page_config(
    page_title="Concept Hustleboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

###############################################################################################################################################################################################
# def get_session_id():
#     # ctx = st.report_thread.get_report_ctx()
#     ctx = get_script_run_ctx()
#     if ctx:
#         return ctx.session_id

# def get_session_info():
#     session_id = get_session_id()
#     if get_session_id():
#         print('session id:', session_id)
#         return st.server.server.Server.get_current()._get_session_info(session_id)
#     raise RuntimeError('No streamlit session')

# def get_request_headers():
#     try:
#         print('session info', get_session_info())
#         return get_session_info().ws.request.headers
#     except AttributeError as e:
#         print(e)
#         return {}


def get_loggedin_user():
    """return logged in user email adress or None if no headers were found"""
    headers = _get_websocket_headers()  # your way to access request headers
    return headers.get('X-Forwarded-Email', None)  

headers = _get_websocket_headers()  # your way to access request headers
for header in headers:
    if header == "Cookie":
        continue
    print(f"{header} : {headers[header]}")  
print('user', get_loggedin_user())
  
###############################################################################################################################################################################################
 
    
# Function to load data with caching
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# Function to preprocess df
@st.cache_data
def preprocess_df(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Commission'] = pd.to_numeric(df['Commission'], errors='coerce')
    df['Rent'] = pd.to_numeric(df['Rent'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year.astype('Int64')
    df['Zip'] = df['Zip'].astype(str)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# Load and preprocess data
df = preprocess_df(load_data('rentals.csv'))

# Create a unified sidebar for filtering
st.sidebar.markdown("## Data at Your Fingertips")

# Agent Name filter in the sidebar
all_names_df = sorted(df['Agent'].astype(str).unique())
agent_name = st.sidebar.selectbox('Name', ['', *all_names_df])

# Year and Month selection in the sidebar
years_selected = st.sidebar.multiselect('Select Year(s)', options=sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
start_month, end_month = st.sidebar.slider('Select Month Range', 1, 12, (1, 12))

# Agent's City filter in the sidebar
all_cities = sorted(df['City'].unique())
agent_city = st.sidebar.selectbox('City', ['', *all_cities])

# Agent's Zip filter in the sidebar
all_zips = sorted(df['Zip'].astype(str).unique())
agent_zip = st.sidebar.selectbox('Zip', ['', *all_zips])

# Status filter in the sidebar
status_options = ['All', 'Converted', 'Lost']
status_selected = st.sidebar.selectbox('Status', status_options)

# Initialize min_rent and max_rent with default values
min_rent = int(df['Rent'].min())
max_rent = int(df['Rent'].max())

# Rent filter with dropdowns for min-max
min_rent_options = list(range(int(df['Rent'].min()), int(df['Rent'].max()) + 1, 100))  # Adjust the step as needed
max_rent_options = list(range(int(df['Rent'].min()), int(df['Rent'].max()) + 1, 100))  # Adjust the step as needed

min_rent = st.sidebar.selectbox('Minimum Rent', [min_rent] + min_rent_options)
max_rent = st.sidebar.selectbox('Maximum Rent', [max_rent] + max_rent_options)

# Initialize min_commission and max_commission with default values
min_commission = int(df['Commission'].min())
max_commission = int(df['Commission'].max())

# Commission filter with dropdowns for min-max
min_commission_options = list(range(int(df['Commission'].min()), int(df['Commission'].max()) + 1, 10))  # Adjust the step as needed
max_commission_options = list(range(int(df['Commission'].min()), int(df['Commission'].max()) + 1, 10))  # Adjust the step as needed

min_commission = st.sidebar.selectbox('Minimum Commission', [min_commission] + min_commission_options)
max_commission = st.sidebar.selectbox('Maximum Commission', [max_commission] + max_commission_options)




# Add checkbox to show commission
show_commission = st.sidebar.checkbox('Show Commission')


# Filter data for df
filtered_data_df = df[
    (df['Year'].isin(years_selected)) &
    (df['Month'].between(start_month, end_month)) &
    (df['Agent'].str.contains(agent_name, na=False) | (agent_name == '')) &
    (df['City'].str.contains(agent_city, na=False) | (agent_city == '')) &
    ((df['Zip'].astype(str).str.contains(agent_zip, na=False, regex=False)) | (agent_zip == '')) &
    (df['Rent'].between(min_rent, max_rent)) &
    (df['Commission'].between(min_commission, max_commission)) &
    ((df['Status'] == status_selected) | (status_selected == 'All'))  
]


#############################################################################################################################################################################################################################
            
# Group the data by 'Agent' and 'Month' and calculate the sum of 'Rent' per month
heatmap_data = filtered_data_df.groupby(['Agent', 'Month'])['Rent'].sum().reset_index()

# Create a DataFrame suitable for a heatmap
heatmap_df = heatmap_data.pivot(index='Agent', columns='Month', values='Rent').fillna(0)

# Create a rectangular heatmap with custom color scale
fig = px.imshow(heatmap_df,
                labels=dict(x="Month", y="Agent", color="Rent"),
                x=heatmap_df.columns,
                y=heatmap_df.index,
                title='Rent Heatmap by Month',
                color_continuous_scale='YlOrRd')

# Initialize the figures
fig_total_rent = go.Figure()
fig_total_transactions = go.Figure()

# Loop through each unique year in the filtered data
for year in sorted(filtered_data_df['Year'].unique()):
    # Filter the data for the specific year
    filtered_year_data = filtered_data_df[filtered_data_df['Year'] == year]
    
    # Total Rent by Month for the specific year
    total_rent_by_month_year = filtered_year_data.groupby('Month')['Rent'].sum().reset_index()
    fig_total_rent.add_trace(
        go.Bar(name=f'Rent {year}', x=total_rent_by_month_year['Month'], y=total_rent_by_month_year['Rent'])
    )
    
    # Total Transactions by Month for the specific year
    total_transactions_by_month_year = filtered_year_data.groupby('Month').size().reset_index(name='Transactions')
    fig_total_transactions.add_trace(
        go.Bar(name=f'Transactions {year}', x=total_transactions_by_month_year['Month'], y=total_transactions_by_month_year['Transactions'])
    )

# Update layout for the rent figure
fig_total_rent.update_layout(title='Total Rent by Month for Each Year')

# Update layout for the transactions figure
fig_total_transactions.update_layout(title='Total Transactions by Month for Each Year')

# Create Streamlit columns
col1, col2, col3 = st.columns([1, 1, 3])

# Display the total rent bar chart in column 1
with col1:
    st.plotly_chart(fig_total_rent, use_container_width=True)

# Display the total transactions bar chart in column 2
with col2:
    st.plotly_chart(fig_total_transactions, use_container_width=True)

# Place the rectangular heatmap in column 3
with col3:
    st.plotly_chart(fig, use_container_width=True)


###############################################################################################################################################################################################
# Map


# Define a color dictionary for different years, add more colors for additional years
color_dict = {2021: 'red', 2022: 'blue', 2023: 'green'}  # Modify this based on your data years

# Create a selectbox for the map type
map_type = st.selectbox('Select View Type', ['Heatmap', 'Marker Map', 'List View'])

# Boston, MA coordinates
boston_coords = [42.3601, -71.0589]


if map_type == 'Heatmap':
        # Generate a list of heat data with commission as weight
    heat_data = [[row['Latitude'], row['Longitude'], row['Commission']] for index, row in filtered_data_df.iterrows()]
            
        # Create a Folium Map centered around Boston, MA
    m2 = folium.Map(location=boston_coords, zoom_start=13)
            
        # Add a HeatMap to the map
    HeatMap(heat_data).add_to(m2)
            
        # Display the map in Streamlit
    folium_static(m2, height=600, width=1000)

elif map_type == 'Marker Map':
        # Create a Folium Map centered around Boston, MA
    m1 = folium.Map(location=boston_coords, zoom_start=13)
            
        # Add a MarkerCluster to the map
    marker_cluster1 = MarkerCluster().add_to(m1)
            
        # Add a marker for each row in the filtered data
    for idx, row in filtered_data_df.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], popup=row['Address'], icon=folium.Icon(color=color_dict[row['Year']])).add_to(marker_cluster1)
            
        # Display the map in Streamlit
    folium_static(m1, height=600, width=1600)
        
elif map_type == 'List View':
    st.write('### List View:')
        
        # Subset the filtered_data to show only the selected columns
    display_columns = ['City', 'Zip', 'Status', 'Month', 'Year', 'Commission', 'Address', 'Landlord' ]
    subset_data = filtered_data_df[display_columns]
        
        # Sort the data by the Month column
    subset_data = subset_data.sort_values(by='Month')
        
        # Convert month numbers to actual month names
        # Assuming the 'Month' column has month numbers (1-12). If they are already named, you can skip this.
    subset_data['Month'] = subset_data['Month'].apply(lambda x: calendar.month_name[int(x)])

        # Calculate number of pages
    num_pages = int(len(subset_data)/10) + (1 if len(subset_data) % 10 else 0)
        
        # Use a selectbox for pagination. If only one page, no selectbox will appear.
    if num_pages > 1:
        pages = [f"Page {i} of {num_pages}" for i in range(1, num_pages + 1)]
        page = st.selectbox('', pages)
        current_page = int(page.split()[1])  # Extracting the current page number
        start_idx = (current_page - 1) * 10
        end_idx = start_idx + 10
        st.table(subset_data.iloc[start_idx:end_idx])
    else:
        st.table(subset_data)

elif map_type == 'Marker Map':
    fig = px.scatter_mapbox(filtered_data_df, lat="Latitude", lon="Longitude", color="Year",
                            color_discrete_map=color_dict,
                            size_max=15, zoom=10,
                            mapbox_style="open-street-map")
    fig.update_layout(width=1000, height=600)
    st.plotly_chart(fig)

elif map_type == 'City Marker Map':
    fig_map = px.scatter_mapbox(filtered_data_df, lat='Latitude', lon='Longitude', color='City', title='Property Locations')
    fig_map.update_layout(mapbox_style="open-street-map", width=1000, height=600)
    st.plotly_chart(fig_map)

elif map_type == 'List View':
    st.write('### List View:')
    
    # Subset the filtered_data to show only the selected columns
    display_columns = [ 'City', 'Zip', 'Status', 'Month', 'Year','Rent', 'Commission', 'Landlord', 'Address']
    subset_data = filtered_data_df[display_columns]
    
    # Sort the data by the Month column
    subset_data = subset_data.sort_values(by='Month')
    
    # Convert month numbers to actual month names
    subset_data['Month'] = subset_data['Month'].apply(lambda x: calendar.month_name[int(x)])
    
    # Calculate number of pages
    num_pages = int(len(subset_data)/10) + (1 if len(subset_data) % 10 else 0)
    
    # Use a selectbox for pagination. If only one page, no selectbox will appear.
    if num_pages > 1:
        pages = [f"Page {i} of {num_pages}" for i in range(1, num_pages + 1)]
        page = st.selectbox('', pages)
        current_page = int(page.split()[1])  # Extracting the current page number
        start_idx = (current_page - 1) * 10
        end_idx = start_idx + 10
        st.table(subset_data.iloc[start_idx:end_idx])
    else:
        st.table(subset_data)

#########################################################################################################################################################################################


# Create Streamlit columns
col1, col2, col3 = st.columns([1, 1, 1])

# Filter data for 'Converted' and 'Lost' statuses
converted_data = filtered_data_df[filtered_data_df['Status'] == 'Converted'].groupby('Month').size().reset_index(name='Converted')
lost_data = filtered_data_df[filtered_data_df['Status'] == 'Lost'].groupby('Month').size().reset_index(name='Lost')

# Create a figure for 'Converted vs Lost'
fig_status = go.Figure()
fig_status.add_trace(go.Bar(name='Converted', x=converted_data['Month'], y=converted_data['Converted']))
fig_status.add_trace(go.Bar(name='Lost', x=lost_data['Month'], y=lost_data['Lost']))
fig_status.update_layout(title='Converted vs Lost by Month')

# Display the 'Converted vs Lost' bar chart in column 1
with col1:
    st.plotly_chart(fig_status, use_container_width=True)

# Total Rent by Month
total_rent_by_month = filtered_data_df.groupby('Month')['Rent'].sum().reset_index()
fig_total_rent = go.Figure(data=[
    go.Bar(name='Rent', x=total_rent_by_month['Month'], y=total_rent_by_month['Rent'])
])
fig_total_rent.update_layout(title='Total Rent by Month')

# Display the total rent bar chart in column 2
with col2:
    st.plotly_chart(fig_total_rent, use_container_width=True)


# Top 10 Landlords based on the number of transactions
top_landlords = filtered_data_df.groupby('Landlord').size().sort_values(ascending=False).reset_index(name='Count').head(10)


# Create a figure for 'Top 10 Landlords'
fig_landlords = go.Figure(data=[go.Bar(name='Top 10 Landlords', x=top_landlords['Landlord'], y=top_landlords['Count'])])
fig_landlords.update_layout(title='Top 10 Landlords')

# Display the 'Top 10 Landlords' bar chart in column 3
with col3:
    st.plotly_chart(fig_landlords, use_container_width=True)

############################################################################################################################################################################################
cols = st.columns([1, 1, 1, 1, 1])

# First Column: Display bar chart for Average Commission per Year
with cols[2]:
    AvComm = filtered_data_df.groupby(['Year', 'Agent'])['Commission'].mean().reset_index(name='AvComm')
    fig = px.bar(AvComm, x='Agent', y='AvComm', color='Year', title='Average Commission per Year', barmode='group')
    cols[2].plotly_chart(fig)

# Second Column: Display bar chart for Revenue per Year
with cols[3]:
    revenue_per_year = filtered_data_df.groupby(['Year', 'Agent'])['Commission'].sum().reset_index(name='Revenue')
    fig = px.bar(revenue_per_year, x='Agent', y='Revenue', color='Year', title='Revenue per Year', barmode='group')
    cols[3].plotly_chart(fig)

# Third Column: Display bar chart for Count per Year
with cols[1]:
    count_per_year = filtered_data_df.groupby(['Year', 'Agent'])['Commission'].count().reset_index(name='Count')
    fig = px.bar(count_per_year, x='Agent', y='Count', color='Year', title='Count per Year', barmode='group')
    cols[1].plotly_chart(fig)

# Fourth Column: Display bar chart for Revenue Change (%)
with cols[4]:
    revenue_per_year = filtered_data_df.groupby(['Year', 'Agent'])['Commission'].sum().reset_index(name='Revenue')
    revenue_per_year['Revenue Change (%)'] = revenue_per_year.groupby('Agent')['Revenue'].pct_change() * 100
    fig = px.bar(revenue_per_year, x='Agent', y='Revenue Change (%)', color='Year', title='Revenue Change (%)', barmode='group')
    cols[4].plotly_chart(fig)

# Fifth Column: Display bar chart for Sales Change (%)
with cols[0]:
    count_per_year = filtered_data_df.groupby(['Year', 'Agent'])['Commission'].count().reset_index(name='Count')
    count_per_year['Sales Change (%)'] = count_per_year.groupby('Agent')['Count'].pct_change() * 100
    fig = px.bar(count_per_year, x='Agent', y='Sales Change (%)', color='Year', title='Sales Change (%)', barmode='group')
    cols[0].plotly_chart(fig)


            
# Sort the data by 'Year' in ascending order
filtered_data_df_sorted = filtered_data_df.sort_values(by='Year')

# Calculate the difference between Rent and Commission
filtered_data_df_sorted['Rent_Commission_Diff'] = filtered_data_df_sorted['Rent'] - filtered_data_df_sorted['Commission']

# Create a dropdown menu for selecting between 'City' and 'Zip' for Commission chart
x_axis_selection_commission = st.selectbox('Select City/Zip for Commission', ['City', 'Zip'], key='commission_select')

# Create a bar chart for Commission with colors for each year
fig_bar = px.bar(
    filtered_data_df_sorted,
    x='City',
    y='Commission',
    color='Year',
    title='Commission (City/Zip)'
)
fig_bar.update_layout(xaxis_title='City', yaxis_title='Commission', legend_title='Year')
fig_bar.update_xaxes(title=x_axis_selection_commission, type='category')
if x_axis_selection_commission == 'Zip':
    fig_bar.data[0]['x'] = filtered_data_df_sorted['Zip']
    fig_bar.update_xaxes(categoryorder='total descending')
fig_bar.update_layout(height=400, width=1600)
st.plotly_chart(fig_bar)

# Create a dropdown menu for selecting between 'City' and 'Zip' for Rent chart
x_axis_selection_rent = st.selectbox('Select City/Zip for Rent', ['City', 'Zip'], key='rent_select')

# Create a bar chart for Rent with colors for each year
fig_bar_rent = px.bar(
    filtered_data_df_sorted,
    x='City',
    y='Rent',
    color='Year',
    title='Rent (City/Zip)'
)
fig_bar_rent.update_layout(xaxis_title='City', yaxis_title='Rent', legend_title='Year')
fig_bar_rent.update_xaxes(title=x_axis_selection_rent, type='category')
if x_axis_selection_rent == 'Zip':
    fig_bar_rent.data[0]['x'] = filtered_data_df_sorted['Zip']
    fig_bar_rent.update_xaxes(categoryorder='total descending')
fig_bar_rent.update_layout(height=400, width=1600)
st.plotly_chart(fig_bar_rent)

# Create a dropdown menu for selecting between 'City' and 'Zip' for Difference chart
x_axis_selection_diff = st.selectbox('Select City/Zip for Rent-Commission Difference', ['City', 'Zip'], key='diff_select')

# Create a bar chart for Rent-Commission Difference with colors for each year
fig_bar_diff = px.bar(
    filtered_data_df_sorted,
    x='City',
    y='Rent_Commission_Diff',
    color='Year',
    title='Rent-Commission Difference (City/Zip)'
)
fig_bar_diff.update_layout(xaxis_title='City', yaxis_title='Rent-Commission Difference', legend_title='Year')
fig_bar_diff.update_xaxes(title=x_axis_selection_diff, type='category')
if x_axis_selection_diff == 'Zip':
    fig_bar_diff.data[0]['x'] = filtered_data_df_sorted['Zip']
    fig_bar_diff.update_xaxes(categoryorder='total descending')
fig_bar_diff.update_layout(height=400, width=1600)
st.plotly_chart(fig_bar_diff)




           
# Make a copy of the filtered DataFrame to avoid SettingWithCopyWarning
filtered_data_df = filtered_data_df.copy()

# Calculate the sum of 'Commission' for each year based on the filtered data
commission_sum_per_year = filtered_data_df.groupby('Year')['Commission'].sum().reset_index()
commission_sum_per_year.rename(columns={'Commission': 'Total Commission'}, inplace=True)

# Merge the sum back into the filtered DataFrame
filtered_data_df = pd.merge(filtered_data_df, commission_sum_per_year, on='Year', how='left')

# Create a bar chart for the sum of commissions per year based on the filtered data
fig_total_commission = px.bar(
    commission_sum_per_year,  # Use the sum DataFrame here
    x='Year',
    y='Total Commission',
    title='Total Commission Per Year (Filtered)',
    labels={'Total Commission': 'Total Commission'}
)

# Calculate the total number of transactions for each year based on the filtered data
transaction_count_per_year = filtered_data_df.groupby('Year').size().reset_index(name='Total Transactions')

# Merge the total transactions back into the filtered DataFrame
filtered_data_df = pd.merge(filtered_data_df, transaction_count_per_year, on='Year', how='left')

# Create a bar chart for the total transactions per year based on the filtered data
fig_total_transactions = px.bar(
    transaction_count_per_year,  # Use the transaction count DataFrame here
    x='Year',
    y='Total Transactions',
    title='Total Transactions Per Year (Filtered)',
    labels={'Total Transactions': 'Total Transactions'}
)










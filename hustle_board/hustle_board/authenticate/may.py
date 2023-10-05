##############################################################################################

# Bird's Eye:

# Style
# Authentication  
# Create users -> Conditional to seperate users -> Access personal data. 

##############################################################################################

# UX:

# Log into resmet (no sign up) then-> log into streamlit (Double auth required?)
# Embbed 'concept agents'(brokermint), followup boss etc. 

# how to endpoint to may.py
# django x streamlit compatibility
# insert html from ejs folder
# may.py -> 
# 1. Test Streamlit/ iFrame OR 
# 2. Test pyChart
# 3. prep data -> create CSV equilivant to may.py

##############################################################################################

# To-do 9/25:

# Create Users
# Group Users
# Style Hustleboard
# link django to st

##############################################################################################

# Overall To-Do:

# Create users in groups -> Admin, Agent.
# If username.group == agent { display agent data }
# Styling on Dashboard -> tell shingi where everything should go.

##############################################################################################

# link django to st

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from PIL import Image
import os

# Get the directory of the 'may.py' module
current_directory = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory containing 'may.py'
os.chdir(current_directory)

# Set the page configuration
st.set_page_config(
    page_title="Concept Hustleboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.markdown("""
    <style>
        .big-font {
            font-size:60px !important;
            text-align: center;
        }
        .small-font {
            font-size:30px !important;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Create 3 column layout
col1, col2, col3 = st.columns([1,5,1])

# Centralize the heading and make it bold in the center column
with col2:
    st.markdown("<h1 class='big-font'>The Hustleboard</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="small-font">
        Ready to unlock the secrets?
        </div>
        """, unsafe_allow_html=True)

# Display image in the third column
with col3:
    image = Image.open('Concept.png')
    st.image(image, width=200)

# Checkbox in the first column
with col1:
    show_markdown = st.checkbox('Click here to discover the secrets of our Dashboard')

if show_markdown:
    st.markdown("""
    <style>
        .big-font {
            font-size:6px !important;
        }
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="small-font center">
    <h1>Hello there, Data Explorer! Welcome aboard the Concept Hustleboard Dashboard!</h3>
    <p>This isn't your average dashboard - it's a ticket to unlocking insights about rental transactions. Interact with it. Question it. See the stories it tells through the lens of different parameters - year, month, agent name, city, zip code, and more.</p>

    <h2>Here's what you're looking at:</h4>

    <h3>1. Map Visualization:</h5> <p>It's a bird's-eye view of the business. Each marker on this interactive map is a transaction. Watch the landscape change color as you shift through the years.</p>

    <h3>2. Summary:</h5> <p>Like the highlights of a great match, this is where you'll see the top 3 performing cities. The magic? It's dynamic, changing with every filter you apply.</p>

    <h3>3. Data Filters:</h5> <p>Your control room for the data. Filter by year, month, agent name, city, and zip code. Watch as your choices reshape the map visualization and summary.</p>

    <h3>4. Statistics:</h5> <p>This is the heartbeat of the operation. Gaze at stats like average commission per year, revenue, count per year, and changes in revenue and sales.</p>

    <h3>5. Leaderboard:</h5> <p>The hall of fame for agents. Rankings are based on commission and number of transactions. Who will be the top performer today?</p>

    <h3>6. Rent Distribution and Comparison:</h5> <p>This is your graphical novel of the story of rent. Compare the distribution and cumulative rent for different years.</p>

    <p>We built this dashboard for you, for insights, for knowledge. Take it for a spin. And if you have any questions, ideas, or just want to chat about data, we're here for you. Happy exploring!</p>
    </div>
    """, unsafe_allow_html=True)




#Load data
df = pd.read_csv('2021-YTDRentals_geo.csv')
nd = pd.read_csv('new_deal_geo.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Commission'] = pd.to_numeric(df['Commission'], errors='coerce')

# Create Month and Year columns
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# Convert the 'Date' column to datetime format
nd['Date'] = pd.to_datetime(nd['Date'])

# Extract the year from the date and create a new column 'Year'
nd['Year'] = nd['Date'].dt.year
nd['Month'] = nd['Date'].dt.month

# Split the data into three dataframes based on the year
nd_2021 = nd[nd['Year'] == 2021]
nd_2022 = nd[nd['Year'] == 2022]
nd_2023 = nd[nd['Year'] == 2023]

# Calculate the total rent for each month
nd_2021['Month'] = nd_2021['Date'].dt.month
nd_2022['Month'] = nd_2022['Date'].dt.month
nd_2023['Month'] = nd_2023['Date'].dt.month

# Display the statistics for each year
print(nd_2021.describe(include='all'))
print(nd_2022.describe(include='all'))
print(nd_2023.describe(include='all'))

nd_2021['Rent_Cumulative'] = nd_2021['Rent'].cumsum()
nd_2022['Rent_Cumulative'] = nd_2022['Rent'].cumsum()
nd_2023['Rent_Cumulative'] = nd_2023['Rent'].cumsum()

# Calculate the total rent for each month
total_rent_month_2021 = nd_2021.groupby('Month')['Rent'].sum()
total_rent_month_2022 = nd_2022.groupby('Month')['Rent'].sum()
total_rent_month_2023 = nd_2023.groupby('Month')['Rent'].sum()




# graphs and visualizations:

# Set Seaborn theme and color palette
sns.set_theme(style="white")
sns.set_palette("bright")


    

# Sidebar


st.markdown("## Your Command Center")
st.markdown("Here is where you have control. Use these filters to focus on the data that's most important to you.")




# Year and Month selection
cols = st.columns(5)
with cols[0]:
    years_selected = st.multiselect('Select Year(s)', options=sorted(list(df['Year'].unique())), default=sorted(list(df['Year'].unique())))
    st.markdown("🔍 **Year Filter**: Want to travel through time? Use this filter to select the year(s) you're interested in.")

with cols[1]:
    start_month, end_month = st.slider('Select Month Range', 1, 12, (1, 12))  # Default is January to December
    st.markdown("📅 **Month Range**: Select the range of months for your exploration. By default, it's set to the whole year. Feel free to narrow it down.")

# Agent's Name filter
# Once the log in and name is there
# Match names to user

# user_rights = admin, agent, manager
# If user_rights == agent && agent.name == username 
# { Then only show data belonging to agent.name }

#I created authentication in django and want to create 

all_names = df['Name'].unique()
with cols[2]:
    agent_name = st.selectbox('Name', ['', *all_names])
    st.markdown("🕵️ **Agent Name**: Looking for someone specific? Select an agent's name from the dropdown list to focus on their transactions.")

# Agent's City filter
all_cities = df['City'].unique()
with cols[3]:
    agent_city = st.selectbox('City', ['', *all_cities])
    st.markdown("🏙️ **City**: Want to dig into a particular city? Choose it from this dropdown list and the data will obey.")

# Agent's Zip filter
all_zips = df['Zip'].unique()
with cols[4]:
    agent_zip = st.selectbox('Zip', ['', *all_zips])
    st.markdown("📍 **Zip Code**: Sometimes, the devil is in the details. Use this to zero in on a specific zip code.")

st.markdown("")

st.markdown("## The Data at Your Fingertips")
st.markdown("The filtered data is ready for your investigation. The world of rental transactions awaits your analysis.")






# Name match filter below: 

# Log in (access) -> Based on Name based on Agent Name, You only get access to the assigned name!

# Filter data
filtered_data = df[
(df['Year'].isin(years_selected)) &
(df['Month'].between(start_month, end_month)) &
(df['Name'].str.contains(agent_name, na=False) | (agent_name == '')) &
(df['City'].str.contains(agent_city, na=False) | (agent_city == '')) &
((df['Zip'].astype(str).str.contains(agent_zip, na=False, regex=False)) | (agent_zip == ''))
]






  
# Map

# Define a color dictionary for different years, add more colors for additional years
color_dict = {2021: 'red', 2022: 'blue', 2023: 'green'}  # Modify this based on your data years

# Create two columns with a ratio of 4:1
cols = st.columns([4, 1])


# Map 1-3 (Index)

# Place the map in the first column
with cols[0]:
    # Define a color dictionary for different years, add more colors for additional years
    color_dict = {2021: 'red', 2022: 'blue', 2023: 'green'}  # Modify this based on your data years

    # Create the map centered over Boston
    m1 = folium.Map([42.3601, -71.0589], zoom_start=13)
    marker_cluster1 = MarkerCluster().add_to(m1)
    for idx, row in filtered_data.iterrows():
        # Use the color dictionary to set the color of the marker based on the year
        folium.Marker([row['Latitude'], row['Longitude']], popup=row['Tag'], icon=folium.Icon(color=color_dict[row['Year']])).add_to(marker_cluster1)
    folium_static(m1, height=600, width=1600)


# Place the summary section in the second column
with cols[1]:
    st.markdown("## Top 3 Cities per Year")

    # Create an empty DataFrame to store the counts
    counts_df = pd.DataFrame()

    # Loop over each selected year
    for year in years_selected:
        # Filter the data for the current year
        yearly_data = filtered_data[filtered_data['Year'] == year]

        # Calculate the counts for the top 3 cities
        yearly_counts = yearly_data['City'].value_counts().nlargest(3)

        # Add the counts to the DataFrame
        counts_df = pd.concat([counts_df, yearly_counts.rename(year)], axis=1)

    # Fill NaN values with zeros
    counts_df.fillna(0, inplace=True)

    # Define a color dictionary for different years, add more colors for additional years
    color_dict = {2021: 'red', 2022: 'blue', 2023: 'green'}  # Modify this based on your data years

    # Create a bar chart with a custom color scheme
    fig, ax = plt.subplots()
    for i, year in enumerate(counts_df.columns):
        counts_df[year].plot(kind='bar', ax=ax, color=color_dict[year], position=i, label=str(year))

    # Display the legend
    ax.legend()

    # Display the bar chart
    st.pyplot(fig)


st.markdown("## 📊 An Animated Year in Review")
st.markdown("See the performance unfold through the years! View average commission, revenue, count and annual changes all in one place.")


 # Define 5 columns
cols = st.columns([1, 1, 1, 1, 1])

# First Column: Display pie chart for AvComm per Year
with cols[2]:
    # Group data by 'Year' and calculate the average commission per year
    AvComm = filtered_data.groupby('Year')['Commission'].mean().reset_index(name='AvComm')

    # Create pie chart for AvComm
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, labels, pct_texts = ax.pie(AvComm['AvComm'], labels=AvComm['Year'], autopct='%1.1f%%', wedgeprops={'linewidth': 3, 'edgecolor': 'white'})

    # Adjust percentage labels
    for pct_text, wedge in zip(pct_texts, wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = 0.7 * wedge.r * np.cos(np.deg2rad(angle))
        y = 0.7 * wedge.r * np.sin(np.deg2rad(angle))
        pct_text.set_position((x, y))

    # Display the AvComm amounts in labels
    for i, label in enumerate(labels):
        label.set_text(f"{label.get_text()} (${AvComm['AvComm'][i]:,.2f})")

    ax.set_title('Average Commission per Year')

    # Add a white circle in the center
    center_circle = plt.Circle((0, 0), 0.5, color='white', linewidth=3, edgecolor='gray')
    ax.add_patch(center_circle)

    # Set aspect ratio to be equal for a circular pie chart
    ax.axis('equal')

    # Display the figure
    cols[2].pyplot(fig)


# Second Column: Display pie chart for Revenue per Year
with cols[3]:
    # Group data by 'Year' and calculate the revenue per year
    revenue_per_year = filtered_data.groupby('Year')['Commission'].sum().reset_index(name='Revenue')

    # Create pie chart for revenue
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, labels, pct_texts = ax.pie(revenue_per_year['Revenue'], labels=revenue_per_year['Year'], autopct='%1.1f%%', wedgeprops={'linewidth': 3, 'edgecolor': 'white'})

    # Adjust percentage labels
    for pct_text, wedge in zip(pct_texts, wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = 0.7 * wedge.r * np.cos(np.deg2rad(angle))
        y = 0.7 * wedge.r * np.sin(np.deg2rad(angle))
        pct_text.set_position((x, y))

    # Display the revenue amounts in labels
    for i, label in enumerate(labels):
        label.set_text(f"{label.get_text()} (${revenue_per_year['Revenue'][i]:,.2f})")

    ax.set_title('Revenue per Year')

    # Add a white circle in the center
    center_circle = plt.Circle((0, 0), 0.5, color='white', linewidth=3, edgecolor='gray')
    ax.add_patch(center_circle)

    # Set aspect ratio to be equal for a circular pie chart
    ax.axis('equal')

    # Display the figure
    cols[3].pyplot(fig)


# Third Column: Display pie chart for Count per Year
with cols[1]:
    # Group data by 'Year' and count the 'Commission' to get the count per year
    count_per_year = filtered_data.groupby('Year')['Commission'].count().reset_index(name='Count')

    # Create pie chart for count
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, labels, pct_texts = ax.pie(count_per_year['Count'], labels=count_per_year['Year'], autopct='%1.1f%%', wedgeprops={'linewidth': 3, 'edgecolor': 'white'})

    # Adjust percentage labels
    for pct_text, wedge in zip(pct_texts, wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = 0.7 * wedge.r * np.cos(np.deg2rad(angle))
        y = 0.7 * wedge.r * np.sin(np.deg2rad(angle))
        pct_text.set_position((x, y))

    # Display the count amounts in labels
    for i, label in enumerate(labels):
        label.set_text(f"{label.get_text()} ({count_per_year['Count'][i]})")

    ax.set_title('Count per Year')

    # Add a white circle in the center
    center_circle = plt.Circle((0, 0), 0.5, color='white', linewidth=3, edgecolor='gray')
    ax.add_patch(center_circle)

    # Set aspect ratio to be equal for a circular pie chart
    ax.axis('equal')

    # Display the figure
    cols[1].pyplot(fig)


# First Column: Display bar chart for Count per Year
with cols[4]:
# Group data by 'Year' and count the 'Commission' to get the count per year
    count_per_year['Sales Change (%)'] = count_per_year['Count'].pct_change() * 100
    revenue_per_year['Revenue Change (%)'] = revenue_per_year['Revenue'].pct_change() * 100
    
# Create bar chart for revenue change
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=revenue_per_year, x='Year', y='Revenue Change (%)', ax=ax)
ax.set_title('Revenue Change (%)')

# Remove borders and lines from the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.grid(False)

# Add arrows indicating direction of change
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate('+', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom', color='green')
    elif height < 0:
        ax.annotate('-', (p.get_x() + p.get_width() / 2, height), ha='center', va='top', color='red')

# Display the figure
cols[4].pyplot(fig)

# Create bar chart for sales change
fig1, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=count_per_year, x='Year', y='Sales Change (%)', ax=ax)
ax.set_title('Sales Change (%)')

# Remove borders and lines from the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.grid(False)

# Add arrows indicating direction of change
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate('+', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom', color='green')
    elif height < 0:
        ax.annotate('-', (p.get_x() + p.get_width() / 2, height), ha='center', va='top', color='red')

# Display the figure
cols[0].pyplot(fig1)


from folium.plugins import HeatMap

# Create two columns with a ratio of 4:1
cols = st.columns([4, 1])

# Place the heat map in the first column
with cols[0]:
   # Generate a list of heat data with commission as weight
    heat_data = [[row['Latitude'], row['Longitude'], row['Commission']] for index, row in filtered_data.iterrows()]
    
    # Create a Map instance centered over Boston
    m2 = folium.Map(location=[42.3601, -71.0589], zoom_start=13)
    
    # Create a HeatMap and add it to the map
    HeatMap(heat_data).add_to(m2)
    
    # Show the map with the heatmap in Streamlit
    folium_static(m2, height=600, width=1600)


# Place the summary section in the second column
with cols[1]:
    st.markdown("## Top 3 Cities by Total Commission")

    # Create an empty DataFrame to store the commission sums
    commission_df = pd.DataFrame()

    # Loop over each selected year
    for year in years_selected:
        # Filter the data for the current year
        yearly_data = filtered_data[filtered_data['Year'] == year]

        # Calculate the commission sum for the top 3 cities
        yearly_commissions = yearly_data.groupby('City')['Commission'].sum().nlargest(3)

        # Add the commission sums to the DataFrame
        commission_df = pd.concat([commission_df, yearly_commissions.rename(year)], axis=1)

    # Fill NaN values with zeros
    commission_df.fillna(0, inplace=True)

    # Define a color dictionary for different years, add more colors for additional years
    color_dict = {2021: 'red', 2022: 'blue', 2023: 'green'}  # Modify this based on your data years

    # Create a stacked bar chart with a custom color scheme
    fig, ax = plt.subplots()
    commission_df.plot(kind='bar', stacked=True, color=[color_dict[year] for year in commission_df.columns], ax=ax)

    # Display the legend
    ax.legend()

    # Display the bar chart
    st.pyplot(fig)


st.markdown("## 🏆 The Leaderboard")
st.markdown("It's time for some healthy competition! Use these filters to see who's leading the pack. Who has the most transactions? Who's earned the most commission? Let's find out!")

# Create columns for settings
cols = st.columns(4)

# Add Year filter
with cols[0]:
    year_selected_leaderboard = st.selectbox('Select Year', options=[2021, 2022, 2023])
    st.markdown("🔍 **Year Filter**: Time travel to a specific year. See who was on top then.")

# Add Start and End Month filters
with cols[1]:
    start_month_leaderboard = st.selectbox('Start Month', options=range(1, 13))
with cols[2]:
    end_month_leaderboard = st.selectbox('End Month', options=range(1, 13))
    st.markdown("📅 **Month Range Filter**: Want to explore a particular season? Use these filters to narrow down your time frame.")

# Add checkbox to show commission
with cols[3]:
    show_commission = st.checkbox('Show Commission', key='show_commission')
    st.markdown("💲 **Show Commission**: Let's talk money! Check this box to see the commission for each transaction.")


# Filter data for leaderboard and year-to-date
filtered_leaderboard = df[(df['Year'] == year_selected_leaderboard) & (df['Month'].between(start_month_leaderboard, end_month_leaderboard))]
filtered_ytd = df[(df['Year'] == pd.to_datetime('today').year) & (df['Month'] <= pd.to_datetime('today').month)]

# Function to generate leaderboard
def get_leaderboard(df, type="Commission"):
    if type == "Commission":
        leaderboard = df.groupby('Name')[type].count().sort_values(ascending=False).reset_index()
        leaderboard[type] = '***'  # replace actual commission with placeholder
    else:
        leaderboard = df.groupby('Name').size().sort_values(ascending=False).reset_index().rename(columns={0: "Transactions"})
    leaderboard.index += 1
    leaderboard.index.name = 'Position'
    return leaderboard

# Create columns for leaderboards
cols_leaderboard = st.columns(4)

leaderboards = [("Commission: Current", filtered_leaderboard, show_commission, cols_leaderboard[0]),
                ("Commission: YTD", filtered_ytd, show_commission, cols_leaderboard[1]),
                ("Transactions: Current", filtered_leaderboard, True, cols_leaderboard[2]),
                ("Transactions: YTD", filtered_ytd, True, cols_leaderboard[3])]

# Display leaderboards
for title, filtered_data, show_commission, column in leaderboards:
    column.markdown(f'#### {title}')
    leaderboard = get_leaderboard(filtered_data, "Commission" if "Commission" in title else "Transactions")

    # Show only top 3 by default
    top_three = leaderboard.head(3)
    rest = leaderboard.iloc[3:]

    styled_top_three = (
        top_three.style
        .background_gradient(cmap='Blues')
        .set_properties(**{'font-size': '16pt', 'border-radius': '10px', 'padding': '10px', 'border': 'None', 'font-weight': 'bold'})
    )
    column.table(styled_top_three.format())

    # Show the rest of the leaderboard if the checkbox is checked
    if column.checkbox('Show more', key=title):
        styled_rest = (
            rest.style
            .background_gradient(cmap='Blues')
            .set_properties(**{'font-size': '16pt', 'border-radius': '10px', 'padding': '10px', 'border': 'None', 'font-weight': 'bold'})
        )
        column.table(styled_rest.format())




st.markdown("## 🔍 Find Your Focus")
st.markdown("Now that we've looked at the big picture, let's narrow our focus. Use these filters to select a specific timeframe and zero in on a particular agent, city, or zip code.")

# Define 5 columns
cols = st.columns(5)

# Year filter
with cols[0]:
    years_selected = st.multiselect('Select Year(s)', options=sorted(list(nd['Year'].unique())), default=sorted(list(nd['Year'].unique())), key='years')
    st.markdown("🕰️ **Year Filter**: Go back in time and relive the action. Analyze performance across different years.")

# Month Range filter
with cols[1]:
    start_month, end_month = st.slider('Select Month Range', 1, 12, (1, 12), key='month_range')  # Default is January to December
    st.markdown("📅 **Month Range Filter**: Need a closer look at a specific period? No problem! Just select your preferred range of months.")

# Agent Name filter
all_names = nd['Agent'].unique()
with cols[2]:
    agent_name = st.selectbox('Name', ['', *all_names], key='name')
    st.markdown("🕵️‍♂️ **Agent Filter**: Spot the all-stars! Choose an agent's name and dive into their performance details.")

# City filter
all_cities = nd['City'].unique()
with cols[3]:
    agent_city = st.selectbox('City', ['', *all_cities], key='city')
    st.markdown("🏙️ **City Filter**: Urban jungle or peaceful suburb? Select a city and see how it's faring in the real estate game.")

# Zip Code filter
all_zips = nd['Zip'].unique()
with cols[4]:
    agent_zip = st.selectbox('Zip', ['', *all_zips], key='zip')
    st.markdown("📍 **ZIP Filter**: Get hyperlocal! Use this filter to explore performance stats in a particular ZIP code.")

# Filter data based on selections
filtered_data = nd[
    (nd['Year'].isin(years_selected)) &
    (nd['Month'].between(start_month, end_month)) &
    (nd['Agent'].str.contains(agent_name, na=False) | (agent_name == '')) &
    (nd['City'].str.contains(agent_city, na=False) | (agent_city == '')) &
    ((nd['Zip'].astype(str).str.contains(agent_zip, na=False, regex=False)) | (agent_zip == ''))
]


filtered_data['DayOfWeek'] = filtered_data['Date'].dt.dayofweek
filtered_data['Day'] = filtered_data['Date'].dt.day
filtered_data['Year'] = filtered_data['Date'].dt.year

palette = {2021: 'blue', 2022: 'red', 2023: 'green'}

avg_rent_dayofweek_year = filtered_data.groupby(['Year', 'DayOfWeek'])['Rent'].mean().reset_index()

fig_avg_rent_day = plt.figure(figsize=(20, 6))

for year in years_selected:
    avg_rent_year = avg_rent_dayofweek_year[avg_rent_dayofweek_year['Year'] == year]
    sns.lineplot(x=avg_rent_year['DayOfWeek'], y=avg_rent_year['Rent'], marker='o', label=str(year))

plt.title('Average Rent per Day of the Week (Split by Year)')
plt.xlabel('Day of the Week')
plt.ylabel('Average Rent')
plt.xticks(np.arange(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.legend()
sns.despine(left=True, bottom=True)

count_transactions_dayofweek_year = filtered_data.groupby(['Year', 'DayOfWeek']).size().reset_index(name='Count')

fig_transactions_dayofweek = plt.figure(figsize=(20, 6))

for year in years_selected:
    count_year = count_transactions_dayofweek_year[count_transactions_dayofweek_year['Year'] == year]
    sns.lineplot(x=count_year['DayOfWeek'], y=count_year['Count'], marker='o', label=str(year), color=palette[year])

plt.title('Number of Transactions per Day of the Week for Each Year')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Transactions')
plt.xticks(np.arange(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.legend()
sns.despine(left=True, bottom=True)




avg_rent_month_day = filtered_data.groupby(['Year', 'Month', 'Day'])['Rent'].mean().reset_index()

count_transactions_month_day = filtered_data.groupby(['Year', 'Month', 'Day']).size().reset_index(name='Count')

fig_avg_rent = plt.figure(figsize=(20, 6))

for year in years_selected:
    avg_rent_year = avg_rent_month_day[avg_rent_month_day['Year'] == year]
    sns.lineplot(x=avg_rent_year['Day'], y=avg_rent_year['Rent'], marker='o', label=str(year))

plt.title('Average Rent per Month Day (Split by Year)')
plt.xlabel('Day of the Month')
plt.ylabel('Average Rent')
plt.legend()
sns.despine(left=True, bottom=True)


fig_transactions_month_day = plt.figure(figsize=(20, 6))

for year in years_selected:
    transactions_year = count_transactions_month_day[count_transactions_month_day['Year'] == year]
    sns.lineplot(x=transactions_year['Day'], y=transactions_year['Count'], marker='o', label=str(year))

plt.title('Number of Transactions per Month Day (Split by Year)')
plt.xlabel('Day of the Month')
plt.ylabel('Number of Transactions')
plt.legend()
sns.despine(left=True, bottom=True)


# Create two columns
col1, col2 = st.columns(2)

# Plot in the first column
with col1:
    st.pyplot(fig_avg_rent_day)
    st.pyplot(fig_transactions_month_day)

# Plot in the second column
with col2:
    st.pyplot(fig_avg_rent)
    st.pyplot(fig_transactions_dayofweek)

# Define 3 columns
cols = st.columns(3)

# Plot 1: Rent Distribution Comparison
with cols[0]:
    fig, ax = plt.subplots(figsize=(20, 10))
    for year in years_selected:
        sns.kdeplot(filtered_data[filtered_data['Year'] == year]['Rent'], label=str(year), fill=True)
    plt.title('New Deal Distribution Comparison', fontweight='bold')
    plt.xlabel('Rent', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.legend()
    ax.spines['top'].set_visible(False)  # Hide the top border
    ax.spines['right'].set_visible(False)  # Hide the right border
    ax.spines['bottom'].set_visible(False)  # Hide the bottom border
    ax.spines['left'].set_visible(False)  # Hide the left border
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    st.pyplot(fig)

# Plot 2: Cumulative Rent Comparison
with cols[1]:
    fig, ax = plt.subplots(figsize=(20, 10))
    for year in years_selected:
        df_year = filtered_data[filtered_data['Year'] == year]
        df_year['Rent_Cumulative'] = df_year['Rent'].cumsum()
        sns.lineplot(data=df_year, x=df_year.index, y='Rent_Cumulative', label=str(year))
    plt.title('Cumulative New Deal Comparison', fontweight='bold')
    plt.xlabel('Index', fontweight='bold')
    plt.ylabel('Cumulative Rent', fontweight='bold')
    plt.legend()
    ax.spines['top'].set_visible(False)  # Hide the top border
    ax.spines['right'].set_visible(False)  # Hide the right border
    ax.spines['bottom'].set_visible(False)  # Hide the bottom border
    ax.spines['left'].set_visible(False)  # Hide the left border
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    st.pyplot(fig)

# Plot 3: Total Rent Comparison for Each Month
with cols[2]:
    fig, ax = plt.subplots(figsize=(20, 10))
    # Group the filtered data by year and month, and calculate total rent for each group
    total_rent_month_df = filtered_data.groupby(['Year', 'Month'])['Rent'].sum().unstack()
    # Plot the bar chart
    total_rent_month_df.plot(kind='bar', ax=ax)
    plt.title('Total New Deal Comparison for Each Month', fontweight='bold')
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Total Rent', fontweight='bold')
    ax.spines['top'].set_visible(False)  # Hide the top border
    ax.spines['right'].set_visible(False)  # Hide the right border
    ax.spines['bottom'].set_visible(False)  # Hide the bottom border
    ax.spines['left'].set_visible(False)  # Hide the left border
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    st.pyplot(fig)


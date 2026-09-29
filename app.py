import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import streamlit as st

st.set_page_config(
    page_title="Google Play Store Analytics", layout="wide"
)

# Title
st.title("Google Play Store Analytics Dashboard")


# 1. Dataset Loading & Preprocessing Logic
@st.cache_data
def load_data():
  df_apps = pd.read_csv("Play Store Data.csv")
  df_reviews = pd.read_csv("User Reviews.csv")

  def convert_size(size_str):
    if pd.isna(size_str) or size_str == "Varies with device":
      return np.nan
    if "M" in str(size_str):
      return float(str(size_str).replace("M", ""))
    if "k" in str(size_str):
      return float(str(size_str).replace("k", "")) / 1024
    return np.nan

  df_apps["Installs_Num"] = (
      df_apps["Installs"]
      .astype(str)
      .str.replace("+", "", regex=False)
      .str.replace(",", "", regex=False)
  )
  df_apps["Installs_Num"] = pd.to_numeric(
      df_apps["Installs_Num"], errors="coerce"
  )

  df_apps["Size_MB"] = df_apps["Size"].apply(convert_size)
  df_apps["Reviews_Num"] = pd.to_numeric(df_apps["Reviews"], errors="coerce")
  df_apps["Rating_Num"] = pd.to_numeric(df_apps["Rating"], errors="coerce")

  df_apps["Price_Num"] = (
      df_apps["Price"]
      .astype(str)
      .str.replace("$", "", regex=False)
      .astype(float, errors="ignore")
  )
  df_apps["Price_Num"] = pd.to_numeric(df_apps["Price_Num"], errors="coerce")
  df_apps["Revenue"] = df_apps["Installs_Num"] * df_apps["Price_Num"]

  df_reviews_avg = (
      df_reviews.groupby("App")["Sentiment_Subjectivity"].mean().reset_index()
  )
  df_main = pd.merge(df_apps, df_reviews_avg, on="App", how="left")

  return df_main


df = load_data()


# Time Check Helper Function
def is_within_time_window(start_hour, end_hour):
  ist = pytz.timezone("Asia/Kolkata")
  now = datetime.datetime.now(ist)
  return start_hour <= now.hour < end_hour


# ---------------------------------------------------------
# TASK 1: BUBBLE CHART (IST 5 PM to 7 PM / 17:00 - 19:00)
# ---------------------------------------------------------
st.header("Task 1: App Size vs Rating (Bubble Chart)")

if is_within_time_window(17, 19):
  # Task 1 Filters
  target_categories = [
      "GAME",
      "BEAUTY",
      "BUSINESS",
      "COMICS",
      "COMMUNICATION",
      "DATING",
      "ENTERTAINMENT",
      "SOCIAL",
      "EVENTS",
  ]

  df_t1 = df[
      (df["Rating_Num"] > 3.5)
      & (df["Category"].isin(target_categories))
      & (df["Reviews_Num"] > 500)
      & (~df["App"].astype(str).str.contains("S", case=True, na=False))
      & (df["Sentiment_Subjectivity"] > 0.5)
      & (df["Installs_Num"] > 50000)
  ].copy()

  # Translations
  category_map = {
      "BEAUTY": "सौंदर्य (Beauty)",
      "BUSINESS": "வணிகம் (Business)",
      "DATING": "Treffen (Dating)",
  }
  df_t1["Category"] = df_t1["Category"].replace(category_map)

  # Plotting Bubble Chart
  fig1 = px.scatter(
      df_t1,
      x="Size_MB",
      y="Rating_Num",
      size="Installs_Num",
      color="Category",
      hover_name="App",
      color_discrete_map={"GAME": "pink"},
      title="App Size vs Average Rating",
      labels={
          "Size_MB": "App Size (MB)",
          "Rating_Num": "Average Rating",
          "Installs_Num": "Installs",
      },
  )
  st.plotly_chart(fig1, use_container_width=True)
else:
  st.info(
      "🔒 Task 1 Chart is only accessible between 5:00 PM IST and 7:00 PM IST."
  )


  # ---------------------------------------------------------
# TASK 2: CHOROPLETH MAP (IST 6 PM to 8 PM / 18:00 - 20:00)
# ---------------------------------------------------------
st.header("Task 2: Global Installs by Category (Choropleth Map)")

if is_within_time_window(18, 20):
  # 1. Filter out categories starting with A, C, G, or S
  excluded_chars = ("A", "C", "G", "S")
  df_t2 = df[
      ~df["Category"].astype(str).str.upper().str.startswith(excluded_chars)
  ].copy()

  # 2. Get Top 5 App Categories by Installs
  top_5_cats = (
      df_t2.groupby("Category")["Installs_Num"]
      .sum()
      .nlargest(5)
      .index.tolist()
  )
  df_t2 = df_t2[df_t2["Category"].isin(top_5_cats)]

  # Aggregating total installs by category
  df_t2_summary = (
      df_t2.groupby("Category")["Installs_Num"].sum().reset_index()
  )

  # Highlighting category where installs exceed 1 Million
  df_t2_summary["Highlight"] = df_t2_summary["Installs_Num"] > 1000000

  # Sample country mapping for Choropleth visualization
  # Note: Standard Google Play Store dataset lacks country-level coordinates,
  # so aggregated category metrics are mapped for top markets representation.
  sample_countries = [
      "USA",
      "IND",
      "BRA",
      "DEU",
      "GBR",
  ]  # ISO-3 codes for map rendering
  df_t2_map = df_t2_summary.copy()
  df_t2_map["ISO_Code"] = [
      sample_countries[i % len(sample_countries)] for i in range(len(df_t2_map))
  ]

  fig2 = px.choropleth(
      df_t2_map,
      locations="ISO_Code",
      color="Installs_Num",
      hover_name="Category",
      color_continuous_scale="Viridis",
      title="Global Installs by Top Categories (Excludes A, C, G, S)",
      labels={"Installs_Num": "Total Installs"},
  )
  st.plotly_chart(fig2, use_container_width=True)
else:
  st.info(
      "🔒 Task 2 Chart is only accessible between 6:00 PM IST and 8:00 PM IST."
  )


# ---------------------------------------------------------
# TASK 3: TIME SERIES LINE CHART (IST 6 PM to 9 PM / 18:00 - 21:00)
# ---------------------------------------------------------
st.header("Task 3: Total Installs Trend Over Time")

if is_within_time_window(18, 21):
  # Data Filtering Logic for Task 3
  # Reviews > 500, App name doesn't contain 'S', doesn't start with x, y, z
  # Category starts with E, C, or B
  df_t3 = df[
      (df["Reviews_Num"] > 500)
      & (~df["App"].astype(str).str.contains("S", case=True, na=False))
      & (
          ~df["App"]
          .astype(str)
          .str.lower()
          .str.startswith(("x", "y", "z"), na=False)
      )
      & (
          df["Category"]
          .astype(str)
          .str.upper()
          .str.startswith(("E", "C", "B"), na=False)
      )
  ].copy()

  # Translations (Beauty -> Hindi, Business -> Tamil, Dating -> German)
  category_map = {
      "BEAUTY": "सौंदर्य (Beauty)",
      "BUSINESS": "வணிகம் (Business)",
      "DATING": "Treffen (Dating)",
  }
  df_t3["Category"] = df_t3["Category"].replace(category_map)

  # Convert Last_Updated to Month-Year period for MoM Trend Analysis
  df_t3["Last_Updated_DT"] = pd.to_datetime(
      df_t3["Last Updated"], errors="coerce"
  )
  df_t3_trend = (
      df_t3.groupby(["Last_Updated_DT", "Category"])["Installs_Num"]
      .sum()
      .reset_index()
  )
  df_t3_trend = df_t3_trend.sort_values("Last_Updated_DT")

  # Calculate Month-over-Month (MoM) Growth Percentage
  df_t3_trend["MoM_Growth"] = (
      df_t3_trend.groupby("Category")["Installs_Num"].pct_change() * 100
  )

  fig3 = px.line(
      df_t3_trend,
      x="Last_Updated_DT",
      y="Installs_Num",
      color="Category",
      title="Total Installs Over Time (MoM Growth Highlighted)",
      labels={
          "Last_Updated_DT": "Last Updated Date",
          "Installs_Num": "Total Installs",
      },
  )

  # Shading area where MoM growth > 20%
  high_growth = df_t3_trend[df_t3_trend["MoM_Growth"] > 20]
  for _, row in high_growth.iterrows():
    fig3.add_vrect(
        x0=row["Last_Updated_DT"] - pd.Timedelta(days=15),
        x1=row["Last_Updated_DT"] + pd.Timedelta(days=15),
        fillcolor="yellow",
        opacity=0.3,
        line_width=0,
        annotation_text=">20% Growth",
    )

  st.plotly_chart(fig3, use_container_width=True)
else:
  st.info(
      "🔒 Task 3 Chart is only accessible between 6:00 PM IST and 9:00 PM IST."
  )


  # ---------------------------------------------------------
# TASK 4: STACKED AREA CHART (IST 4 PM to 6 PM / 16:00 - 18:00)
# ---------------------------------------------------------
st.header("Task 4: Revenue Trend Across Categories")

if is_within_time_window(16, 18):
  # Filtering: Installs > 10,000, Rating > 4.0, Size <= 15MB
  # Category starting with E, C, or B
  df_t4 = df[
      (df['Installs_Num'] > 10000)
      & (df['Rating_Num'] > 4.0)
      & (df['Size_MB'] <= 15)
      & (
          df['Category']
          .astype(str)
          .str.upper()
          .str.startswith(('E', 'C', 'B'), na=False)
      )
  ].copy()

  # Category Translations
  category_map = {
      'EVENTS': 'घटनाएं (Events)',
      'COMICS': 'காமிக்ஸ் (Comics)',
      'BEAUTY': 'Schönheit (Beauty)',
  }
  df_t4['Category'] = df_t4['Category'].replace(category_map)

  # Sorting by Last Updated Date
  df_t4['Last_Updated_DT'] = pd.to_datetime(
      df_t4['Last Updated'], errors='coerce'
  )
  df_t4_trend = (
      df_t4.groupby(['Last_Updated_DT', 'Category'])['Revenue']
      .sum()
      .reset_index()
  )
  df_t4_trend = df_t4_trend.sort_values('Last_Updated_DT')

  # Stacked Area Chart
  fig4 = px.area(
      df_t4_trend,
      x='Last_Updated_DT',
      y='Revenue',
      color='Category',
      title='Revenue Trend by Category Over Time (Stacked)',
      labels={
          'Last_Updated_DT': 'Last Updated Date',
          'Revenue': 'Total Revenue ($)',
      },
  )
  st.plotly_chart(fig4, use_container_width=True)
else:
  st.info(
      '🔒 Task 4 Chart is only accessible between 4:00 PM IST and 6:00 PM IST.'
  )


# ---------------------------------------------------------
# TASK 5: GROUPED BAR CHART (IST 3 PM to 5 PM / 15:00 - 17:00)
# ---------------------------------------------------------
st.header("Task 5: Installs & Revenue Comparison (Paid vs Free)")

if is_within_time_window(15, 17):
  # Filtering: Installs > 10k, Rating > 4.0, Size <= 20MB
  # Category starting with E, C, or B, updated in January
  df['Last_Updated_DT'] = pd.to_datetime(df['Last Updated'], errors='coerce')

  df_t5 = df[
      (df['Installs_Num'] > 10000)
      & (df['Rating_Num'] > 4.0)
      & (df['Size_MB'] <= 20)
      & (
          df['Category']
          .astype(str)
          .str.upper()
          .str.startswith(('E', 'C', 'B'), na=False)
      )
      & (df['Last_Updated_DT'].dt.month == 1)  # Updated in January
  ].copy()

  # Grouping by Category and Type (Free vs Paid)
  df_t5_grouped = (
      df_t5.groupby(['Category', 'Type'])[['Installs_Num', 'Revenue']]
      .sum()
      .reset_index()
  )

  # Grouped Bar Chart
  fig5 = px.bar(
      df_t5_grouped,
      x='Category',
      y='Installs_Num',
      color='Type',
      barmode='group',
      title='Total Installs Comparison (Paid vs Free Apps Updated in Jan)',
      labels={'Installs_Num': 'Total Installs', 'Category': 'App Category'},
  )
  st.plotly_chart(fig5, use_container_width=True)
else:
  st.info(
      '🔒 Task 5 Chart is only accessible between 3:00 PM IST and 5:00 PM IST.'
  )


  # ---------------------------------------------------------
# TASK 6: DUAL-AXIS COMBO CHART (IST 7 PM to 9 PM / 19:00 - 21:00)
# ---------------------------------------------------------
st.header("Task 6: Average Rating vs Total Reviews by Category")

if is_within_time_window(19, 21):
  # Filtering: Installs > 50,000, Sentiment Subjectivity > 0.5
  # Excludes App names containing 'S'
  df_t6 = df[
      (df['Installs_Num'] > 50000)
      & (df['Sentiment_Subjectivity'] > 0.5)
      & (~df['App'].astype(str).str.contains('S', case=True, na=False))
  ].copy()

  # Category Translations (Spanish & Japanese)
  category_map = {
      'EVENTS': 'Eventos (Events)',
      'COMICS': 'マンガ (Comics)',
      'BEAUTY': 'Belleza (Beauty)',
  }
  df_t6['Category'] = df_t6['Category'].replace(category_map)

  # Aggregating Mean Rating & Total Reviews
  df_t6_grouped = (
      df_t6.groupby('Category')
      .agg({'Rating_Num': 'mean', 'Reviews_Num': 'sum'})
      .reset_index()
  )

  # Creating Dual-Axis Figure using Plotly Graph Objects
  import plotly.graph_objects as go

  fig6 = go.Figure()

  # Primary Axis: Bar Chart for Total Reviews
  fig6.add_trace(
      go.Bar(
          x=df_t6_grouped['Category'],
          y=df_t6_grouped['Reviews_Num'],
          name='Total Reviews',
          marker_color='lightblue',
      )
  )

  # Secondary Axis: Line Chart for Average Rating
  fig6.add_trace(
      go.Scatter(
          x=df_t6_grouped['Category'],
          y=df_t6_grouped['Rating_Num'],
          name='Average Rating',
          yaxis='y2',
          mode='lines+markers',
          line=dict(color='firebrick', width=3),
      )
  )

  # Layout setup for Dual Y-Axes
  fig6.update_layout(
      title='Average Rating and Total Reviews Comparison',
      xaxis=dict(title='App Category'),
      yaxis=dict(title='Total Reviews', side='left'),
      yaxis2=dict(
          title='Average Rating',
          side='right',
          overlaying='y',
          range=[0, 5],  # Standard Play Store Rating scale
      ),
      legend=dict(x=0.01, y=0.99),
  )

  st.plotly_chart(fig6, use_container_width=True)
else:
  st.info(
      '🔒 Task 6 Chart is only accessible between 7:00 PM IST and 9:00 PM IST.'
  )


  
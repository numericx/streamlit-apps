import pandas as pd
import streamlit as st
import plotly.express as px

st.title(body="SF Trees Map")

# Load data
trees_df = pd.read_csv(filepath_or_buffer="./data/trees.csv")

# Show map
st.write("Trees by Location")
trees_df = trees_df.dropna(subset=["latitude", "longitude"])
trees_df = trees_df.sample(n=1000, replace=True)
st.map(data=trees_df)
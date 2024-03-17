import pandas as pd
import streamlit as st
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="San Francisco Trees",
    page_icon="🌲",
    layout="wide")

# Set page title
st.title(body="San Francisco Trees")
st.write("""This app analyses tress in San Francisco using
         a dataset kindly provided by SF DPW.""")

# Load data
trees_df = pd.read_csv(filepath_or_buffer="./data/trees.csv")

# Time features
today = pd.to_datetime(arg="today")
trees_df["date"] = pd.to_datetime(arg=trees_df["date"])
trees_df["age"] = (today - trees_df["date"]).dt.days

# Find unique owners
owners = st.sidebar.multiselect(label="Tree Owner Filter",
                                options=trees_df["caretaker"].unique())

# Graph colors
graph_color = st.sidebar.color_picker(label="Graph Color")                                

# Filter on caretaker
if owners:
    trees_df = trees_df[trees_df["caretaker"].isin(owners)]
else:
    trees_dbh_grouped = pd.DataFrame(data=trees_df.groupby(by="dbh").count()["tree_id"])
    trees_dbh_grouped.columns = ["tree_count"]
    st.line_chart(data=trees_dbh_grouped, color=graph_color)

col1, col2 = st.columns(spec=2)
with col1:
    fig = px.histogram(data_frame=trees_df, 
                       x="dbh", 
                       title="Tree Width",
                       color_discrete_sequence=[graph_color])
    fig.update_xaxes(title_text="Tree Width")
    st.plotly_chart(figure_or_data=fig, use_container_width=True)

with col2:
    fig = px.histogram(data_frame=trees_df, 
                       x="age", 
                       title="Tree Age",
                       color_discrete_sequence=[graph_color])
    fig.update_xaxes(title_text="Tree Age")
    st.plotly_chart(figure_or_data=fig, use_container_width=True)

# Show map
# st.write("Trees by Location")
# trees_df = trees_df.dropna(subset=["latitude", "longitude"])
# trees_df = trees_df.sample(n=1000, replace=True)
# st.map(data=trees_df)    
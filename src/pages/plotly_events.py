import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events


st.title(body="Streamlit Plotly Events Example: Penguins")

penguins_df = pd.read_csv(filepath_or_buffer="./data/penguins.csv")

fig = px.scatter(data_frame=penguins_df, 
                 x="bill_length_mm", 
                 y="bill_depth_mm", 
                 color="species")

selected_point = plotly_events(plot_fig=fig, click_event=True)
if len(selected_point) == 0:
    st.stop()
else:
    selected_x_value = selected_point[0]["x"]
    selected_y_value = selected_point[0]["y"]
    df_selected = penguins_df[(penguins_df["bill_length_mm"] == selected_x_value) &
                              (penguins_df["bill_depth_mm"] == selected_y_value)]
    
st.write("Selected Point:")
st.write(df_selected)
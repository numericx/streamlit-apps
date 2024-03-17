import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

st.title(body="Streamlit AgGrid Example: Penguins")

penguins_df = pd.read_csv(filepath_or_buffer="./data/penguins.csv")

st.write("AgGrid DataFrame:")

response = AgGrid(data=penguins_df, 
                  height=500, 
                  width='100%',
                  fit_columns_on_grid_load=True,
                  editable=True)

df_edited = response['data']
st.write("Edited DataFrame:")
st.dataframe(data=df_edited)
import pandas as pd
import streamlit as st

st.title(body="SF Trees Data Quality App")
st.write("""This app is a data quality tool for the SF Trees dataset.
         Edit the data and save the file.""")

trees_df = pd.read_csv(filepath_or_buffer="./data/trees.csv")
trees_df.dropna(subset=["latitude", "longitude"], inplace=True)
trees_df_filtered = trees_df[trees_df["legal_status"] == "Private"]
edited_df = st.data_editor(data=trees_df)

if st.button(label="Save data and overwrite"):
    edited_df.to_csv(path_or_buf="./data/edited_trees.csv", index=False)
    st.write("Data saved!")
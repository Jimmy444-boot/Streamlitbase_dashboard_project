import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Superstore Dashboard")
st.markdown('<style>div.block-container{padding-top:2rem;}<style>', unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "txt", "xlsx", "xls"])

if fl is not None: 
    st.write(f"Loaded file: {fl.name}")
    # Directly read from uploaded file object
    df = pd.read_csv(fl)  
else:
    # Make sure file exists before trying to read it
    default_path = "/Users/jamesosuji/Documents/streamapp"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.error(f"Default file not found at: {default_path}")
        st.stop()

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

StartDate = df["Order Date"].min()
EndDate = df["Order Date"].max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", StartDate))

with col1:
    date2 = pd.to_datetime(st.date_input("Start Date", StartDate))

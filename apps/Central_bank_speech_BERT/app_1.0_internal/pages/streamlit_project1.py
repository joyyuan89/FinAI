"""
Created on July 2023

@author: jiayue.yuan
"""

import streamlit as st

# data processing
import pandas as pd
import numpy as np
import datetime as dt
from functions_project1 import *

# plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Intraday liquidity risk reporting",
    page_icon="📊",
    layout="wide",
)

#%% main page

st.title("📈 Intraday liquidity risk reporting")
st.header("")

with st.expander("ℹ️ - About this page", expanded=True):

    st.write(
        """     
- This page is designed to help you understand the intraday liquidity risk of your bank.
	    """
    )

    st.markdown("")

st.markdown("")

# define variables using sidebar input

with st.form(key="my_form"):

    st.sidebar.header("Define Variables")

    input_folder_name = st.sidebar.text_input("Input Folder Name", "INPUT")
    primary_key = st.sidebar.text_input("Primary Key", "RTGS Ref No")
    delay_hours = st.sidebar.number_input("Delayed Hours", value=2)

    start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 5, 23))
    end_date = st.sidebar.date_input("End Date", datetime.date(2023, 5, 25))

    output_folder_name = st.sidebar.text_input("Output Folder Name", "OUTPUT")
    file_name_bau = st.sidebar.text_input("Output File Name for BAU", "OUTPUT_BAU.xlsx")
    file_name_delayed = st.sidebar.text_input("Output File Name for Delayed", "OUTPUT_Delayed.xlsx")

    submit_button = st.form_submit_button(label="✨ Get me the result!")
    
    if not submit_button:
        st.stop()

# main
# 1. Load data
df_raw = load_data(input_folder_name , primary_key)
df_clean = clean_data(df_raw)

# 2. add new columns
df_tag = df_clean.copy()
df_tag['type'] = df_clean.apply(assign_type, axis=1)
df_tag['direction'] = df_tag.apply(assign_direction, axis=1)
df_tag['cashflow'] = df_tag.apply(assign_cashflow, axis=1)
df_tag['date'] = df_tag['Updated on'].dt.date
df_tag['time'] = df_tag['Updated on'].dt.time

# 3. create detailed table
df_detailed = calculate_cumsum(df_tag,'time')

df_detailed_delayed = df_tag.copy()
df_detailed_delayed['time_delayed'] = df_detailed_delayed.apply(delay_time, args=(dealyed_hours,), axis=1)
df_detailed_delayed = calculate_cumsum(df_detailed_delayed,'time_delayed')

# 4. create summary table
df_daily_summary = create_daily_summary(df_detailed)
df_daily_summary_delayed = create_daily_summary(df_detailed_delayed)

# 5. create period summary
df_period_summary = create_period_summary(start_date, end_date, df_daily_summary)
df_period_summary_delayed = create_period_summary(start_date, end_date, df_daily_summary_delayed)


# show results
st.table(df_period_summary)
        


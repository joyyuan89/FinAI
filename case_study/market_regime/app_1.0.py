# 8 Sep 2023
# jiayue

import streamlit as st
import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

#ignore warining
import warnings
warnings.filterwarnings('ignore')

from datapreprocessing import DataPreprocessing
from model import HighDimensionalClustering
from visualizer import ClusterVisualizer
from functions_bayes import *


from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# config
st.set_page_config(
    page_title="Market Regime",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 0. define functions
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# rename the labels by statistics of each cluster
def rename_labels(df_labeled, relabel_col):

    # rename lablels: rank of mean of US_equity_MA65 in each kmeans cluster(label) 
    df_summary = df_labeled.groupby("label")[relabel_col].describe()
    #df_summary['relabel'] = df_summary['mean'].rank(ascending= False).astype(int)
    df_summary['relabel'] = df_summary['mean'].rank().astype(int)

    # add column 'relabel' to df_labeled, keep the index of df_labeled
    df_relabeled = df_labeled.join(df_summary['relabel'], on = 'label')

    return df_relabeled

# 1. load data
dir = os.getcwd()

# Load dataset
raw_data = load_data(os.path.join(dir,"INPUT/index_data.csv"))
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d/%m/%y') # Convert date to datetime format from 8/2/23 to 2023-08-02
raw_data['Date'] = raw_data['Date'].dt.date
raw_data.set_index('Date', inplace=True) # Set date as index
raw_data.dropna(axis='columns', inplace=True) # Data preprocessing - remove error/outliers

# Load reference data
ref_data = load_data(os.path.join(dir,'INPUT/index_set.csv'))
df_index_set = ref_data[ref_data['group']==1][['set','index']]

# 2. Page intro
st.title("Market Regime Analysis")
st.markdown(
    """
    This app is a demo of market regime analysis.
    """
)

# 3. sidebar
# Define your variables
li_base_index_name = ["XAU Curncy","EUCRBRDT Index"]  # Gold, Crude Oil
li_rolling_window = [10, 21, 42, 65]
li_calculation_type = ["MA", "STD", "DIFF"]
li_set = df_index_set['set'].unique()
#start_date = None
#end_date = None

with st.form(key="my_form"):
    st.sidebar.header("Variables for data preprocessing and model")

    # Add widgets for each variable to the sidebar
    base_index_name = st.sidebar.selectbox("Base Index Name", li_base_index_name)
    rolling_windows = st.sidebar.multiselect("Rolling Window", li_rolling_window, default=li_rolling_window)
    calculation_types = st.sidebar.multiselect("Calculation Type", li_calculation_type, default=li_calculation_type)
    start_date = st.sidebar.date_input("Start Date", raw_data.index.min())
    end_date = st.sidebar.date_input("End Date", raw_data.index.max())
    relabeling_index_name = st.sidebar.selectbox("Column for relabeling", li_set, index = 0)
    relabeling_window_number = st.sidebar.selectbox("Window number for relabeling", li_rolling_window, index = 1)

    # model parameters
    #reducer_name = 'UMAP', dimension= 2, Nneighbor=30, clustering_model_name = 'kMeans', Ncluster = 9

    submit_button = st.form_submit_button(label="✨ Get me the result!")
    
    if not submit_button:
        st.stop()


# 4. Feature Engineering

data_pre = DataPreprocessing(data = raw_data, 
                                base_index = base_index_name,
                                rolling_windows = li_rolling_window,
                                calculation_types = li_calculation_type,
                                start_date = start_date,
                                end_date = end_date,
                                grouped = True,
                                index_set = df_index_set,
                                rounded = True,
                                round_level = 8)

df_derived = data_pre.construct_market_data()
market_data_PCA = df_derived.copy()
market_data = df_derived.loc[:,~df_derived.columns.str.startswith('US_equity_AMZN')]     # why?
st.success("Data preprocessing successfully!", icon="✅")

# 5. Clustering Model
st.markdown("## 🌡️ Current Market Regime")
st.markdown("Utilizing umap for dimension reduction and k-means for clustering, every day is divided into a market regime. The darkest and largest point represents :red[Today].")
umap_kmeans = HighDimensionalClustering(reducer_name = 'UMAP', dimension= 2, Nneighbor=30, clustering_model_name = 'kMeans', Ncluster = 9)
umap_kmeans.clustering(market_data)
# add labels to the data
clusterable_embedding = umap_kmeans.low_dimension_embedding
labels = umap_kmeans.labels
df_labeled = market_data.copy()
df_labeled['label'] = labels
# rename labels
relabel_col = f'{relabeling_index_name}_MA{relabeling_window_number}'
df_relabeled = rename_labels(df_labeled, relabel_col)
# plot
visualizer = ClusterVisualizer(clusterable_embedding, labels = np.array(df_relabeled['relabel']),trace = True)
fig_2d = visualizer.plot_2d()
st.markdown(f"The cluters are renamed by :blue[**{relabel_col}**]. The larger number means larger mean value of {relabel_col} of the cluster.")
st.plotly_chart(fig_2d)

# 7. Bayesian Model
st.markdown("## 📈 Bayesian Model")
nb_model = CategoricalNB()
fig1, fig2, summary_table, mapping, Mean, Std = display(df_labeled, relabel_col, monitor_period = 600)
st.markdown(f"The cluters/stage are renamed by :blue[**{relabel_col}**]. The larger number means larger mean value of {relabel_col} of the stage.")
# show results
col1, space, col2 = st.columns([1,0.1,1])

with col1:
    st.plotly_chart(fig1)

with col2:
     st.plotly_chart(fig2)

# st.dataframe(summary_table.round(2))
st.markdown(f"Chance of rally in {relabel_col}: {Mean}")
#st.markdown(f"Standard deviation: {Std}")

# 4. heatmap of correlation
st.markdown("")
st.markdown("### 📎 Appendix: correlation of indexes in raw data")
corr_matrix = raw_data.corr()  # Replace selected_columns with your selected columns
fig_hm, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', ax=ax)
st.pyplot(fig_hm)
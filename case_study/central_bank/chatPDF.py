import streamlit as st
import pandas as pd

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

uploaded_file = st.file_uploader("Choose a file")
#uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

if uploaded_file is not None:

    # load and split by page

    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()

print(pages[0])





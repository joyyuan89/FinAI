import streamlit as st
import tempfile
import pandas as pd

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# config for streamlit app
st.set_page_config(
    page_title="ChatPDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


uploaded_file = st.file_uploader("Choose a file")
#uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name

            loader = PyPDFLoader(file_path)
            pages = loader.load()
    

            text = pages[4].page_content

            st.write(text)





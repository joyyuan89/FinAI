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

 ## split and add metadata
text_splitter1 = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

uploaded_file = st.file_uploader("Choose a file")
#uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name

            loader = PyPDFLoader(file_path)
            pages = loader.load()

            st.write("load PDF data sucessfully")


            ## join all pages into one string
            number_of_pages = len(pages)
            full_text = ' '.join([pages[i].page_content for i in range(number_of_pages)])

            texts = text_splitter1.create_documents([full_text])





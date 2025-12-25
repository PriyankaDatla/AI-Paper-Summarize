# app.py

import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import re
import os
import tempfile

# Initialize the AI Summarizer (BART is excellent for this)
# This model will download the first time you run the script.
@st.cache_resource
def load_summarizer():
    # We switch to a distilled model which is ~300MB instead of 1.6GB
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

SUMMARIZER = load_summarizer()
MAX_CHUNK_SIZE = 1000  # Token limit for BART model input
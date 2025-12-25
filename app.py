import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import re
import tempfile
import os

# 1. Setup Page Config
st.set_page_config(page_title="AI Research Brief", page_icon="📄")

# 2. Load the lightest model for deployment stability
@st.cache_resource
def load_model():
    # T5-small is ~242MB, perfect for free-tier hosting
    return pipeline("summarization", model="t5-small")

summarizer = load_model()

def extract_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    # Simple cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_summary(text):
    # T5 needs this prefix
    input_text = "summarize: " + text
    # Standard research papers are long, so we take the first 3000 chars 
    # to avoid memory crashes on the free tier
    truncated_text = input_text[:3000] 
    
    summary = summarizer(truncated_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# 3. UI Layout
st.title("📄 AI Research Paper Briefing")
st.write("Upload a PDF to get a quick AI summary.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    if st.button("Generate Brief"):
        with st.spinner("Analyzing paper..."):
            raw_text = extract_text(path)
            if len(raw_text) > 100:
                result = generate_summary(raw_text)
                st.subheader("Summary")
                st.success(result)
            else:
                st.error("Could not read enough text from PDF.")
    
    os.unlink(path)

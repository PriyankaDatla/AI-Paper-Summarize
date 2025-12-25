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
    # 't5-base' is smarter than 't5-small' but still fits in Streamlit memory
    return pipeline("summarization", model="t5-base")

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
    input_text = "summarize: " + text
    # Increase the character limit slightly to give the AI more context
    truncated_text = input_text[:4000] 
    
    summary = summarizer(
        truncated_text, 
        max_length=300,    # Increase this for a longer summary
        min_length=100,    # Set a minimum length so it's not too short
        length_penalty=2.0, 
        num_beams=4,       # Beam search helps produce higher quality text
        early_stopping=True
    )
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


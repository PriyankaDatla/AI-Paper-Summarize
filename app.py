import streamlit as st
import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import tempfile
import os

# 1. Setup Page Config
st.set_page_config(page_title="AI Research Brief", page_icon="📄")


# 2. Load T5-small model
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model


tokenizer, model = load_model()


def extract_text(file_path):
    text = ""

    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()

    # Simple cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_summary(text):
    input_text = "summarize: " + text

    # Limit input size
    truncated_text = input_text[:4000]

    inputs = tokenizer(
        truncated_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=300,
        min_length=100,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# 3. UI Layout
st.title("📄 AI Research Paper Briefing")
st.write("Upload a PDF to get a quick AI summary.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

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

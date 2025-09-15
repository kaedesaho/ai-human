import re
import streamlit as st
from pypdf import PdfReader
from docx import Document


def clean_text(text):
    cleaning_pattern = r"[^a-zA-Z0-9\s.,;:!?'\"()\{\}\-—]"
    text = text.replace('\n', ' ')  # replace newline characters with space
    text = re.sub(cleaning_pattern, '', text)  # remove unwanted characters
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces into one
    text = text.strip()  # remove leading/trailing spaces
    return text

def read_file(file):
    if file.type == "text/plain":  # txt
        return str(file.read(), "utf-8")

    elif file.type == "application/pdf":  # pdf
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # docx
        doc = Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text

    else:
        st.error("Unsupported file type.")
        return None

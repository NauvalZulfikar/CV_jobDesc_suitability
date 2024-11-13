import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, DebertaV2Tokenizer
import torch
import numpy as np
from docx import Document  # Import python-docx for handling DOCX files

# Streamlit app code
st.title("CV Suitability Checker using Large Language Model (LLM)")
st.write("This tool surpasses typical ATS systems by using semantic analysis to assess CV-job alignment contextually, not just through keyword matching. It leverages SentenceTransformer for nuanced similarity scoring and integrates advanced language models to generate tailored feedback, helping applicants refine their CVs to better fit specific job descriptions.")
st.write("Upload your CV and paste the job description to see how well your CV aligns with the job requirements and get improvement suggestions.")

# Load the pre-trained DeBERTa-v3-large model and tokenizer
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your CV file", type=['pdf','docx', 'txt'])

# Job description input
job_description = st.text_area("Enter the Job Description")

# Function to extract text from DOCX file
def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Check if file and job description are provided
if uploaded_file is not None and job_description:
    try:
        # Check file type and read content accordingly
        if uploaded_file.name.endswith('.docx'):
            file_content = read_docx(uploaded_file)
        else:
            file_content = uploaded_file.read().decode('utf-8')  # For plain text files

        # Encode the CV and job description
        cv_embedding = model.encode(file_content, convert_to_tensor=True)
        job_desc_embedding = model.encode(job_description, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_score = cosine_similarity(cv_embedding.reshape(1, -1), job_desc_embedding.reshape(1, -1))[0][0]

        # Display the similarity score
        st.write(f"Suitability Score: {similarity_score*100:.2f}%")

        # Interpretation of the score
        if similarity_score >= 0.8:
            st.success("High Suitability!")
        elif similarity_score >= 0.5:
            st.info("Moderate Suitability")
        else:
            st.warning("Low Suitability")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload your CV and enter a job description.")

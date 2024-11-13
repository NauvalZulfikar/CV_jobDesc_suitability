import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, DebertaV2Tokenizer
import torch
from docx import Document  # Import python-docx for handling DOCX files

# Streamlit app title and description
st.title("CV Suitability Checker using DeBERTa-v3-large")
st.write("This tool surpasses typical ATS systems by using semantic analysis to assess CV-job alignment contextually. It leverages DeBERTa-v3-large for nuanced similarity scoring and generates tailored feedback to help refine your CV.")

# Load the DeBERTa-v3-large model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "microsoft/deberta-v3-large"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Streamlit file uploader for CV
uploaded_file = st.file_uploader("Upload your CV file", type=['docx', 'txt'])

# Job description input
job_description = st.text_area("Enter the Job Description")

# Function to extract text from a DOCX file
def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Function to generate embeddings using DeBERTa-v3-large
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Main logic to process CV and Job Description
if uploaded_file is not None and job_description:
    try:
        # Read the content of the uploaded file
        if uploaded_file.name.endswith('.docx'):
            file_content = read_docx(uploaded_file)
        else:
            file_content = uploaded_file.read().decode('utf-8')  # For plain text files

        # Generate embeddings for CV and Job Description
        cv_embedding = get_embeddings(file_content, tokenizer, model)
        job_desc_embedding = get_embeddings(job_description, tokenizer, model)

        # Calculate cosine similarity
        similarity_score = cosine_similarity(cv_embedding.cpu().numpy(), job_desc_embedding.cpu().numpy())[0][0]

        # Display the suitability score
        st.write(f"Suitability Score: {similarity_score * 100:.2f}%")

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

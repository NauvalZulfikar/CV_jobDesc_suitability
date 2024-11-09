import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import numpy as np
from docx import Document  # Import python-docx for handling DOCX files

# Load NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")  # Ensure spacy model is installed

# Load SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract skills and entities from job description
def extract_skills(job_description):
    doc = nlp(job_description)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]  # Modify as needed
    return skills

# Streamlit app
st.title("Enhanced CV Suitability Checker")
st.write("Upload your CV and enter the job description to see how well it aligns. Get specific feedback on skills, experience, and ATS compatibility.")

# File upload and job description input
uploaded_file = st.file_uploader("Upload your CV file", type=['docx', 'txt'])
job_description = st.text_area("Enter the Job Description")

if uploaded_file and job_description:
    try:
        # Extract and analyze job description skills
        job_skills = extract_skills(job_description)
        st.write(f"Key Skills Extracted from Job Description: {', '.join(job_skills)}")

        # Read CV content
        def read_docx(file):
            doc = Document(file)
            full_text = [para.text for para in doc.paragraphs]
            return "\n".join(full_text)
        
        file_content = read_docx(uploaded_file) if uploaded_file.name.endswith('.docx') else uploaded_file.read().decode('utf-8')

        # Semantic similarity scoring
        cv_embedding = model.encode(file_content, convert_to_tensor=True)
        job_desc_embedding = model.encode(job_description, convert_to_tensor=True)
        similarity_score = cosine_similarity(cv_embedding.reshape(1, -1), job_desc_embedding.reshape(1, -1))[0][0]
        st.write(f"Suitability Score: {similarity_score*100:.2f}%")

        # Feedback on missing skills
        cv_skills = extract_skills(file_content)
        missing_skills = [skill for skill in job_skills if skill not in cv_skills]
        if missing_skills:
            st.warning(f"Consider adding these skills: {', '.join(missing_skills)}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CV and enter a job description.")

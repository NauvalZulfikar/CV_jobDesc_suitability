import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
from docx import Document  # Import python-docx for handling DOCX files

# Streamlit app code
st.title("CV Suitability Checker")
st.write("Upload your CV and paste the job description to see how well your CV aligns with the job requirements.")

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load a text generation model for feedback
@st.cache_resource
def load_feedback_model():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

feedback_model = load_feedback_model()

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your CV file", type=['docx', 'txt'])

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

        # Generate feedback using GPT-Neo
        feedback_prompt = f"The job description is:\n{job_description}\n\nThe CV content is:\n{file_content}\n\nPlease provide feedback on what could be improved in the CV to better match the job description."
        feedback = feedback_model(feedback_prompt, max_length=150, num_return_sequences=1)[0]['generated_text']

        # Display improvement suggestions
        st.subheader("Improvement Suggestions")
        st.write(feedback)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload your CV and enter a job description.")

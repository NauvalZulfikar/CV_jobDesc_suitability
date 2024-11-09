import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from docx import Document  # Import python-docx for handling DOCX files

# Streamlit app code
st.title("CV Suitability Checker with Feedback")
st.write("Upload your CV and paste the job description to see how well your CV aligns with the job requirements and get improvement suggestions.")

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load GPT-4-Alpaca model and tokenizer
@st.cache_resource
def load_feedback_model():
    model_name = "chavinlo/gpt4-alpaca"  # Replace with the correct Hugging Face model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    alpaca_model = AutoModelForCausalLM.from_pretrained(model_name)
    return alpaca_model, tokenizer

alpaca_model, tokenizer = load_feedback_model()

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

        # Truncate the CV content and job description to avoid exceeding model input limits
        truncated_cv_content = file_content[:1000]  # Truncate CV text to first 1000 characters
        truncated_job_description = job_description[:1000]  # Truncate job description to first 1000 characters

        # Generate feedback using GPT-4-Alpaca with specific prompt
        feedback_prompt = f"The job description is:\n{truncated_job_description}\n\nThe CV content is:\n{truncated_cv_content}\n\nPlease provide feedback on what could be improved in the CV to better match the job description."
        
        inputs = tokenizer(feedback_prompt, return_tensors="pt").to(alpaca_model.device)
        output = alpaca_model.generate(inputs.input_ids, max_length=500, num_return_sequences=1)
        feedback = tokenizer.decode(output[0], skip_special_tokens=True)

        # Display improvement suggestions
        st.subheader("Improvement Suggestions")
        st.write(feedback)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload your CV and enter a job description.")

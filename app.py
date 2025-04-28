import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import pdfminer.high_level
from pdfminer.layout import LAParams
import os
import spacy
import logging
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK resources (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Extract text from PDF file using pdfminer.six
def extract_text_from_pdf(pdf_path):
    try:
        output_string = StringIO()
        with open(pdf_path, 'rb') as in_file:
            pdfminer.high_level.extract_text_to_fp(in_file, output_string, laparams=LAParams())
        return output_string.getvalue()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

# Preprocess the resume text (remove punctuation, lowercase, tokenize, and remove stopwords)
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))  # Get stopwords
    filtered_tokens = [w for w in tokens if not w in stop_words]  # Remove stopwords
    return " ".join(filtered_tokens)

# Extract person's name from resume using spaCy
def extract_name(resume_text):
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Name not found"

# Extract phone number from resume using regex pattern
def extract_phone_number(resume_text):
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}'
    match = re.search(phone_pattern, resume_text)
    return match.group() if match else None

# Extract email address from resume using regex pattern
def extract_email(resume_text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, resume_text)
    return match.group() if match else None

# Extract total years of experience from resume
def extract_experience_years(resume_text):
    experience_years = re.findall(r'(\d+)\+?\s+(?:years|yrs)', resume_text, re.I)
    if experience_years:
        return max(map(int, experience_years))
    return None

# Analyze resume against job description (SAP CPI specific)
def analyze_resume_cpi(resume_path, job_description, resume_text):
    processed_resume = preprocess_text(resume_text)
    processed_job_description = preprocess_text(job_description)

    # Calculate similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([processed_resume, processed_job_description])
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Basic Keyword Matching
    job_keywords = processed_job_description.split()
    resume_keywords = processed_resume.split()
    matched_keywords = list(set(job_keywords) & set(resume_keywords))

    # SAP CPI Specific Skills Analysis
    cpi_skills = [
        "sap cpi", "sap cloud platform integration", "cloud platform integration",
        "integration flow", "iflow", "odata", "soap", "rest", "sftp", "http",
        "successfactors", "s4hana", "ariba", "fieldglass", "sap erp",
        "groovy script", "xslt mapping", "message mapping", "value mapping",
        "content modifier", "router", "process direct", "request reply",
        "integration adapter", "api management", "security artifacts",
        "certificate", "keystore", "oauth", "cpi monitoring", "hci", "hana cloud integration",
        "cloud foundry", "scp", "sap cloud platform", "api proxy",
        "camel", "apache camel", "edi", "idoc", "as2", "xpath", "json",
        "bpm", "business process management", "cpi administration",
        "transport management", "cpi security", "odata services", "sap analytics cloud",
        "bapi", "rfc", "s4/hana", "cpi developer", "successfactors integration", "api management",
        "camel context", "message queue", "cloud integration", "integration suite"
    ]

    cpi_skills_found = [skill for skill in cpi_skills if skill in processed_resume]

    # Seniority Level Detection
    seniority_keywords = ["lead", "architect", "senior", "expert", "consultant", "principal"]
    seniority_found = any(keyword in processed_resume for keyword in seniority_keywords)

    # Certification Keywords
    cert_keywords = ["certified", "certificate"]
    certs_found = any(keyword in processed_resume for keyword in cert_keywords)

    analysis_results = {
        "similarity_score": similarity_score,
        "matched_keywords": matched_keywords,
        "cpi_skills_found": cpi_skills_found,
        "seniority_level": "Senior" if seniority_found else "Intermediate/Junior",
        "overall_fit": "Good" if similarity_score > 0.4 and seniority_found and certs_found else "Average",
        "certifications": "Present" if certs_found else "Absent"
    }

    return analysis_results

# Get all PDF and TXT files from the folder
def get_files_from_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.txt'))]
    return files

# Save results to Excel
def save_to_excel(results):
    df = pd.DataFrame(results)
    output_file = "resume_analysis_results.xlsx"
    df.to_excel(output_file, index=False)
    return output_file

# Streamlit app setup
st.title("Resume Analysis Web App")

job_description = st.text_area("Job Description", "We are seeking a Senior SAP CPI Consultant to design and implement integration solutions. Integration iflows, JMS message queues, event broker, event mesh, api management")

folder_path = st.text_input("Enter Folder Path", "Enter the full path of the folder containing resumes")

if folder_path:
    if os.path.isdir(folder_path):
        files = get_files_from_folder(folder_path)
        if files:
            st.write(f"Found {len(files)} files to process.")

            # Process resumes and display results
            results = []
            for resume_path in files:
                with open(resume_path, "r", encoding="utf-8") as f:
                    resume_text = f.read() if resume_path.lower().endswith(".txt") else extract_text_from_pdf(resume_path)

                # Extract details from resume
                name = extract_name(resume_text)
                phone = extract_phone_number(resume_text)
                email = extract_email(resume_text)
                experience_years = extract_experience_years(resume_text)

                # Analyze resume
                analysis_results = analyze_resume_cpi(resume_path, job_description, resume_text)

                result = {
                    "File Name": os.path.basename(resume_path),
                    "Name": name,
                    "Phone": phone,
                    "Email": email,
                    "Experience (Years)": experience_years,
                    "Similarity Score": analysis_results['similarity_score'],
                    "Matched Keywords": ', '.join(analysis_results['matched_keywords']),
                    "CPI Skills Found": ', '.join(analysis_results['cpi_skills_found']),
                    "Seniority Level": analysis_results['seniority_level'],
                    "Overall Fit": analysis_results['overall_fit'],
                    "Certifications": analysis_results['certifications']
                }
                results.append(result)

            if results:
                st.write("Analysis Complete!")
                st.dataframe(results)

                # Provide option to download the results as Excel
                output_file = save_to_excel(results)
                st.download_button("Download Excel", output_file, file_name="resume_analysis_results.xlsx")
        else:
            st.write("No PDF or TXT files found in the specified folder.")
    else:
        st.write("Invalid folder path.")

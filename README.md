# Resume Analysis Web App

This web application analyzes resumes and compares them to a job description. It uses NLP techniques to extract relevant information from resumes, such as:

- Name
- Phone number
- Email
- Total experience years
- Relevant skills
- Seniority level
- Certifications

## Features

- Upload a folder containing resumes in PDF or TXT format.
- Analyze resumes against a specified job description.
- Download the results as an Excel file.

## How to Run Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/resume-analysis-web-app.git
    cd resume-analysis-web-app
    ```

2. Set up a virtual environment (optional):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use .\venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

5. Open your browser at `http://localhost:8501`.

## License

This project is licensed under the MIT License.

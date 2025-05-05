import os
import json
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize AI Model
llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.7, api_key=GROQ_API_KEY)

# ===================== Helper Functions ===================== #

def generate_interview_questions(job_description):
    """Generates a detailed list of interview questions based on the job description."""
    prompt = f"""
    Generate a **detailed** list of **technical and behavioral** interview questions for the following job description:
    
    {job_description}
    
    **STRICTLY return only JSON**. Do NOT include any extra text, headers, or explanations.
    
    Output Format:
    ```json
    {{
        "technical_questions": [
            "Question 1",
            "Question 2",
            ...
        ],
        "behavioral_questions": [
            "Question 1",
            "Question 2",
            ...
        ]
    }}
    ```
    """

    response = llm.invoke(prompt).content

    # Extract JSON using regex if extra text appears
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        response = json_match.group(0)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Error: Model output is not valid JSON.")
        return {"technical_questions": [], "behavioral_questions": []}

def evaluate_answer(question, user_answer):
    """Uses LLM to evaluate the candidate's response with structured feedback."""
    prompt = f"""
    Evaluate the following response to the question: "{question}"

    Candidate's answer:
    {user_answer}

    Provide a **score out of 10** and **constructive feedback** in JSON format:

    ```json
    {{
        "score": X,
        "feedback": "Your feedback here"
    }}
    ```
    """

    response = llm.invoke(prompt).content

    # Extract JSON using regex if extra text appears
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        response = json_match.group(0)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"score": "N/A", "feedback": "Could not evaluate response."}

# ===================== Streamlit UI ===================== #

st.set_page_config(page_title="Resume & Interview Chatbot", layout="wide")
st.title("🚀 Resume Optimization & Interview Preparation")

# Store job description in session state
if "job_description" not in st.session_state:
    st.session_state.job_description = None

# Input for job description
job_description = st.text_area("📄 Paste Job Description Below", height=250)
if job_description:
    st.session_state.job_description = job_description

# Interview Preparation Section
st.subheader("🎤 AI-Powered Interview Chatbot")

# Generate Interview Questions
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = None

if st.button("Generate Interview Questions") and st.session_state.get("job_description"):
    with st.spinner("Generating comprehensive interview questions..."):
        st.session_state.interview_questions = generate_interview_questions(st.session_state["job_description"])

# Display Interview Questions
if st.session_state.get("interview_questions"):
    st.subheader("📌 Technical Questions")
    for i, question in enumerate(st.session_state.interview_questions["technical_questions"]):
        st.write(f"**Q{i+1}:** {question}")

    st.subheader("💼 Behavioral Questions")
    for i, question in enumerate(st.session_state.interview_questions["behavioral_questions"]):
        st.write(f"**Q{i+1}:** {question}")

    # Answer Input Section
    st.subheader("📝 Answer a Question")
    all_questions = (
        st.session_state.interview_questions["technical_questions"] + 
        st.session_state.interview_questions["behavioral_questions"]
    )
    
    question_selection = st.selectbox("Select a question to answer", all_questions)
    user_answer = st.text_area("Your Answer", "")

    if st.button("Get Feedback"):
        if user_answer:
            with st.spinner("Evaluating your answer..."):
                feedback = evaluate_answer(question_selection, user_answer)
            st.subheader("📊 Feedback")
            st.write(f"**Score:** {feedback['score']}/10")
            st.write(f"**Feedback:** {feedback['feedback']}")
        else:
            st.warning("⚠️ Please enter an answer before getting feedback.")

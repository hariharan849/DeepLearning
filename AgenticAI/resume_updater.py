import time, requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from docx import Document
from pptx import Presentation
import fitz  # PyMuPDF for PDFs
import os
from difflib import unified_diff, HtmlDiff
from difflib import ndiff
from dotenv import load_dotenv
load_dotenv()
temp_dir = "temp_files"
os.makedirs(temp_dir, exist_ok=True)

# Retrieve LangChain API Key securely
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up LLM
llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.7, api_key=GROQ_API_KEY)

def rebuild_resume_with_format(updated_text, formatting, file_type):
    file_path = os.path.join(temp_dir, f"updated_resume.{file_type}")
    
    if file_type == "docx":
        doc = Document()
        for entry in formatting:
            para = doc.add_paragraph()
            run = para.add_run(entry["text"])
            if entry.get("bold"):
                run.bold = True
            if entry.get("italic"):
                run.italic = True
        doc.save(file_path)
    elif file_type == "pptx":
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        textbox = slide.shapes.add_textbox(100, 100, 500, 300)
        text_frame = textbox.text_frame
        text_frame.text = updated_text
        prs.save(file_path)
    elif file_type == "pdf":
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), updated_text)
        doc.save(file_path)
    
    return file_path


# Define state
class ResumeUpdateState:
    def __init__(self, resume: str, job_description: str, extracted_requirements="", resume_gaps="", updated_resume=""):
        self.resume = resume
        self.job_description = job_description
        self.extracted_requirements = extracted_requirements
        self.resume_gaps = resume_gaps
        self.updated_resume = updated_resume

def extract_text_and_format_from_resume(file, file_type):
    file_path = os.path.join(temp_dir, file.name)
    
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    extracted_text = ""
    formatting = []
    
    if file_type == "docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            extracted_text += para.text + "\n"
            formatting.append({
                "text": para.text,
                "bold": any(run.bold for run in para.runs),
                "italic": any(run.italic for run in para.runs),
                "font_size": para.runs[0].font.size if para.runs else None
            })
    elif file_type == "pptx":
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    extracted_text += shape.text + "\n"
                    formatting.append({"text": shape.text, "font_size": None})
    elif file_type == "pdf":
        doc = fitz.open(file_path)
        for page in doc:
            for text in page.get_text("dict")["blocks"]:
                if "lines" in text:
                    for line in text["lines"]:
                        for span in line["spans"]:
                            extracted_text += span["text"] + " "
                            formatting.append({"text": span["text"], "font_size": span["size"]})
            extracted_text += "\n"
    else:
        return "", []
    
    safe_remove(file_path)  # Ensure file deletion

    return extracted_text.strip(), formatting

def safe_remove(file_path):
    for _ in range(5):  # Try for a short period
        try:
            os.remove(file_path)
            break
        except PermissionError:
            time.sleep(0.5)  # Wait and retry

def fetch_job_description_from_llm(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Extract raw text from the webpage
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)

        # Use LLM to extract the job description from the raw text
        job_description = llm.invoke(f"""
        Extract the job description from the following webpage text. Focus only on the responsibilities, qualifications, and skills mentioned for the role. 
        Ignore navigation menus, company descriptions, and unrelated content. Show each skill or requirement on a new line.

        Webpage Content:
        {page_text}

        Extracted Job Description:
        """)

        return job_description
    
    except Exception as e:
        print (f"Error fetching job description: {str(e)}")
        return

def highlight_differences(original, updated):
    diff = list(ndiff(original.split(), updated.split()))
    highlighted = " ".join([
        f'**{word[2:]}**' if word.startswith('+ ') else word[2:]  
        for word in diff if not word.startswith('- ')
    ])
    return highlighted

def extract_job_requirements(state):
    extracted_text = llm.invoke(extract_prompt.format(job_description=state["job_description"]))
    return {
        "resume": state["resume"],
        "job_description": state["job_description"],
        "extracted_requirements": extracted_text,
        "formatting": state["formatting"],  # Ensure formatting is passed forward
        "file_type": state["file_type"]  # Ensure file type is also passed forward
    }

def analyze_resume_gaps(state):
    gap_analysis = llm.invoke(analyze_prompt.format(resume=state["resume"], extracted_requirements=state["extracted_requirements"]))
    return {
        "resume": state["resume"],
        "job_description": state["job_description"],
        "extracted_requirements": state["extracted_requirements"],
        "resume_gaps": gap_analysis.content,
        "formatting": state["formatting"],  # Ensure formatting is passed forward
        "file_type": state["file_type"]  # Ensure file_type is also passed forward
    }

def update_resume(state):
    updated_text = llm.invoke(f"""
        Modify the resume below to align with the job requirements while keeping it truthful and professional.
        Highlight missing skills and reframe experience to better match the role.
        
        Resume:
        {state["resume"]}
        
        Identified Gaps:
        {state["resume_gaps"]}
        
        Where possible, subtly incorporate missing but relevant skills into existing experience sections.
        Return only the modified text against original resume
    """)
    highlighted_text = highlight_differences(state["resume"], updated_text.content)
    return {
        "resume": state["resume"],
        "job_description": state["job_description"],
        "extracted_requirements": state["extracted_requirements"],
        "resume_gaps": state["resume_gaps"],
        "updated_resume": highlighted_text
    }

# Extract key skills from job description
extract_prompt = PromptTemplate(
    input_variables=["job_description"],
    template="""
    Extract key skills, responsibilities, and qualifications from the following job description:
    {job_description}
    """
)

# Step 2: Analyze gaps in resume
analyze_prompt = PromptTemplate(
    input_variables=["resume", "extracted_requirements"],
    template="""
    Compare the given resume with the extracted job requirements and identify missing skills, experience gaps, and areas of improvement.
    Resume:
    {resume}
    Job Requirements:
    {extracted_requirements}
    """
)

graph = StateGraph(dict)
graph.add_node("extract", extract_job_requirements)
graph.add_node("analyze", analyze_resume_gaps)
graph.add_node("update", update_resume)

graph.add_edge("extract", "analyze")
graph.add_edge("analyze", "update")
graph.add_edge("update", END)

graph.set_entry_point("extract")
graph = graph.compile()

st.title("Resume Comparison against Job Description")
st.write("This tool helps you optimize your resume by comparing it against a job description and identifying key gaps.")

uploaded_resume = st.file_uploader("Upload your resume (Word, PPT, or PDF)", type=["docx", "pptx", "pdf"])
# Display information about the uploaded file
if uploaded_resume is not None:
    st.session_state.otto_file = uploaded_resume
    st.header("Uploaded File Details:")

    # Display file type and size
    file_details = {
        "File Name": uploaded_resume.name,
        "File Type": uploaded_resume.type,
        "File Size (bytes)": uploaded_resume.size,
    }
    st.write(file_details)

job_url = st.text_input("Paste the job listing URL")

if "job_url" not in st.session_state:
    st.session_state.job_url = None

st.session_state.job_url = job_url

if "job_description" not in st.session_state:
    st.session_state.job_description = None

if st.button("Fetch Job Description") and st.session_state.job_url:
    print()
    job_description = fetch_job_description_from_llm(st.session_state.job_url)
    if job_description:
        st.session_state.job_description = job_description.content  # Store in session state
        st.success("Job description fetched successfully!")

if st.session_state.job_description:
    st.subheader("Extracted Job Description")
    st.text_area("Job Description", st.session_state.job_description, height=300)

if st.button("Optimize Resume") and uploaded_resume and st.session_state.job_description:
    file_type = uploaded_resume.name.split(".")[-1]
    resume_text, formatting = extract_text_and_format_from_resume(uploaded_resume, file_type)
    
    if not resume_text:
        st.error("Unsupported file type or extraction error.")
    else:
        state = {"resume": resume_text, "job_description": st.session_state.job_description, "formatting": formatting, "file_type": file_type}
        output = graph.invoke(state)
        
        st.subheader("Identified Resume Gaps:")
        st.write(output["resume_gaps"])

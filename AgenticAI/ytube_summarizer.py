import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from typing import Dict, TypedDict, Annotated

# Load environment variables
load_dotenv()

# Initialize LLM models
llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
code_llm = ChatOllama(model="llama3.2:1b", temperature=0.7)

# Define state schema
class VideoState(TypedDict):
    video_id: Annotated[str, "input"] 
    transcript: Annotated[str, "transcript"]
    extracted_code: Annotated[str, "extracted_code"]
    reviewed_code: Annotated[str, "reviewed_code"]  # Human-reviewed code
    extracted_summary: Annotated[str, "extracted_summary"]
    reviewed_summary: Annotated[str, "reviewed_summary"]  # Human-reviewed summary
    collaborated_summary: Annotated[str, "collaborated_summary"]
    blog: Annotated[str, "blog"]

def get_video_transcript(video_id):
    """Fetch transcript for a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        st.warning(f"Could not fetch transcript for {video_id}: {e}")
        return ""

def extract_code_from_video(state: VideoState) -> Dict:
    """Extract code snippets from video captions."""
    transcript = state["transcript"]
    prompt = f"Extract all code snippets from the transcript below:\n{transcript}"
    response = code_llm.invoke(prompt)
    return {"extracted_code": response.content if hasattr(response, "content") else str(response)}

def generate_summary(state: VideoState) -> Dict:
    """Generate a detailed summary of the transcript."""
    transcript = state["transcript"]
    prompt = f"Summarize the following transcript:\n{transcript}"
    response = llm.invoke(prompt)
    return {"extracted_summary": response.content if hasattr(response, "content") else str(response)}

def generate_collaborated_summary(state: VideoState) -> Dict:
    """Combine reviewed summary and reviewed code."""
    summary = state.get("reviewed_summary", "")
    code_snippets = state.get("reviewed_code", "")

    prompt = f"Merge the summary with code snippets:\nSummary: {summary}\nCode: {code_snippets}"
    response = llm.invoke(prompt)
    return {"collaborated_summary": response.content if hasattr(response, "content") else str(response)}

def generate_blog(state: VideoState) -> Dict:
    """Generate a blog post from the collaborated summary."""
    collaborated_summary = state.get("collaborated_summary", "")

    prompt = f"Write a blog based on the following:\n{collaborated_summary}"
    response = llm.invoke(prompt)
    return {"blog": response.content if hasattr(response, "content") else str(response)}

def process_video(state: VideoState) -> Dict:
    """Fetch transcript for a video."""
    transcript = get_video_transcript(state["video_id"])
    return {**state, "transcript": transcript}

# Define LangGraph StateGraph
graph = StateGraph(VideoState)

# Add processing nodes
graph.add_node("process_video", process_video)
graph.add_node("extract_code", extract_code_from_video)
graph.add_node("generate_summary", generate_summary)
graph.add_node("generate_collaborated_summary", generate_collaborated_summary)
graph.add_node("generate_blog", generate_blog)

# Define execution order
graph.add_edge("process_video", "generate_summary")
graph.add_edge("process_video", "extract_code")

graph.set_entry_point("process_video")
state_machine = graph.compile()

# Streamlit UI
st.title("AI-Assistant YouTube Video Summarizer")

url = st.text_input("Enter YouTube video URL:", "")

if st.button("Process Video") or "video_processed" in st.session_state:
    if "video_processed" not in st.session_state:
        st.session_state["video_processed"] = True

        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
            st.subheader(f"Processing Video: {video_id}")

            if "state" not in st.session_state:
                st.session_state["state"] = state_machine.invoke({"video_id": video_id})

    if "state" in st.session_state:
        with st.expander("Transcript"):
            st.write(st.session_state["state"]["transcript"])

        with st.expander("Extracted Code (Review & Edit)"):
            reviewed_code = st.text_area("Review Extracted Code:", st.session_state["state"]["extracted_code"], height=600)
            if st.button("Confirm Code"):
                st.session_state["state"]["reviewed_code"] = reviewed_code
                st.success("Code Confirmed!")

        with st.expander("Extracted Summary (Review & Edit)"):
            reviewed_summary = st.text_area("Review Summary:", st.session_state["state"]["extracted_summary"], height=600)
            if st.button("Confirm Summary"):
                st.session_state["reviewed_summary"] = reviewed_summary
                st.success("Summary Confirmed!")

        if "reviewed_code" in st.session_state["state"] and "reviewed_summary" in st.session_state:
            state = generate_collaborated_summary(st.session_state["state"])
            with st.expander("Collaborated Summary"):
                st.write(state["collaborated_summary"])

    else:
        st.error("Invalid YouTube URL. Please enter a valid video link.")

# --- Step 4: AI Chatbot ---
st.header("Chat with the Video Summary")

if "reviewed_summary" in st.session_state and st.session_state["reviewed_summary"]:
    # Ensure message history exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_input = st.chat_input("Ask something about the video summary...")
    
    if user_input:
        # Append user query
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Generate AI Response
        prompt = f"""
        You are an AI assistant answering questions based on this video summary:
        {st.session_state["reviewed_summary"]}

        User's question: {user_input}
        """
        response = llm.invoke(prompt)

        ai_response = response.content if hasattr(response, "content") else str(response)
        st.session_state["messages"].append({"role": "assistant", "content": ai_response})

        # Display AI Response
        with st.chat_message("assistant"):
            st.markdown(ai_response)

else:
    st.info("Please confirm the summary first before using the chatbot.")

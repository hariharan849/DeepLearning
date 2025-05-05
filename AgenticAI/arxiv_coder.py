# from langchain import hub
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders import ArxivLoader
# from dotenv import load_dotenv

# load_dotenv()
# loader = ArxivLoader(
#     query="Semantic Image Synthesis with Spatially-Adaptive Normalization",
#     load_max_docs=2,  # max number of documents
#     load_all_available_meta=True,  # load all available metadata
# )
# code_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# chain = code_llm | StrOutputParser()

# docs = loader.load()

# def format_docs(docs):
#     """Format documents into a single readable string."""
#     return "\n\n".join(doc.page_content for doc in docs)

# formatted_docs = format_docs(docs)

# # Step 2: Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# splits = text_splitter.split_documents(docs)

# # Step 3: Embed & Store in FAISS
# vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-small"))
# retriever = vectorstore.as_retriever()

# # Step 4: Define the retrieval query
# query = "Extract and generate all PyTorch code snippets from the document."
# retrieved_chunks = retriever.invoke(query)

# # Step 5: Format retrieved documents
# formatted_chunks = format_docs(retrieved_chunks)

# # Step 6: Ask the LLM
# llm_prompt = [
#     {
#         "role": "system",
#         "content": "Extract paper detailed summarization and generate all PyTorch code snippets from the document with relevant pytorch opensource dataset with model training and evaluation."},
#     {"role": "user", "content": formatted_chunks},
# ]
# response = chain.invoke(llm_prompt)

# # Print the response
# print(response)

from fastapi import FastAPI, Query
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Arxiv PyTorch Code Extractor")

# LangChain Models
code_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Request Model
class ArxivQuery(BaseModel):
    query: str
    max_docs: int = 2

@app.post("/extract_code/")
async def extract_code(query: ArxivQuery):
    """
    Extracts PyTorch code snippets from an Arxiv paper.
    """
    # Load documents
    loader = ArxivLoader(query=query.query, load_max_docs=query.max_docs, load_all_available_meta=True)
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Embed and store in FAISS
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vectorstore.as_retriever()

    # Retrieve relevant sections
    retrieved_chunks = retriever.invoke("Extract all PyTorch code snippets from the document.")
    formatted_chunks = format_docs(retrieved_chunks)

    # Generate code snippets
    llm_prompt = [
        {"role": "system", "content": "Extract all PyTorch code snippets from the document and return as JSON."},
        {"role": "user", "content": formatted_chunks},
    ]
    
    response = (code_llm | parser).invoke(llm_prompt)
    return {"query": query.query, "code_snippets": response}

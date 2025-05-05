from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.storage import InMemoryByteStore
from langchain.tools.retriever import create_retriever_tool

from llms import groq
from state import utils
from loader.arxiv import get_arxiv_content_and_summary_from_title

# ### Enter the research topic you want to search for in the Query parameter

class AgenticRAG:
    def  __init__(self, title: str, llm):
        self.title = title
        self.summary = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        self.ollama_embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # model=<model-name>
        )
                # Create an in-memory byte store
        store = InMemoryByteStore()

        self.cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.ollama_embeddings, store, namespace=self.ollama_embeddings.model
        )
        self.llm = llm
        self.retriever_tool = None

    def process(self):
        docs = get_arxiv_content_and_summary_from_title(self.title)

        papers = []
        for doc in docs:
            metadata = doc.metadata
            split_content = self.text_splitter.split_text(doc.page_content)  # Split text into chunks
            self.summary = doc.page_content
            
            if metadata.get("Title"):
                self.title = metadata.get("Title")
            paper = utils.ArxivPaper(
                title=metadata.get("Title", "No Title"),
                authors=metadata.get("Authors", "").split(", "),
                summary=metadata.get("Summary", ""),
                published_date=metadata.get("Published", "Unknown Date"),
                url=metadata.get("links", "No URL"),
                chunks=split_content  # Store split content
            )
            papers.append(paper)

        vectorstore=FAISS.from_documents(
            documents=docs,
            embedding=self.cached_embedder
        )
        retriever = vectorstore.as_retriever()
        self.retriever_tool = create_retriever_tool(
            retriever,
            "arxiv_retriever",
            "Search and run information about Arxiv papers"
        )
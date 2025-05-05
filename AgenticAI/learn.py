import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS

# Initialize OpenAI embeddings
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Create FAISS index
embedding_size = 768  # Size for OpenAI embeddings
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

from langchain.memory import VectorStoreRetrieverMemory

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=retriever)
# sample_embedding = embeddings_model.embed_query("test")
# embedding_size = len(sample_embedding)
# print(embedding_size)

# Save arbitrary conversations
memory.save_context(
    inputs={
        "human": "Hello, thank you for coming to the interview today. Please introduce yourself."
    },
    outputs={
        "ai": "Hello. I'm a junior developer who majored in Computer Science. In college, I mainly used Java and Python, and recently, I participated in a web development project where I gained experience developing services for real users."
    },
)
memory.save_context(
    inputs={"human": "What was your role in the project?"},
    outputs={
        "ai": "My role was as a backend developer. I was responsible for processing user data and developing server logic, implementing RESTful APIs for communication with the frontend. I also participated in database design."
    },
)
memory.save_context(
    inputs={
        "human": "If you faced any difficulties in team projects, how did you resolve them?"
    },
    outputs={
        "ai": "We had some communication issues at the beginning of the project. To resolve this, our team held regular meetings to share each person's progress. Also, when problems arose, we actively shared opinions and worked to find reasonable solutions."
    },
)
memory.save_context(
    inputs={"human": "What do you consider your strengths as a developer?"},
    outputs={
        "ai": "My strengths are quick learning ability and problem-solving skills. I can quickly acquire new technologies and tools, and when faced with complex problems, I can propose creative solutions. Also, I value teamwork and consider collaboration with colleagues important."
    },
)

print(
    memory.load_memory_variables({"prompt": "What was the interviewee's strength?"})[
        "history"
    ]
)
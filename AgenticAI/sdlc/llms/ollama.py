from langchain_ollama import ChatOllama
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory


class OllamaLLMModel:
    _instance = {}

    def __new__(cls, model_name, temperature):
        if (model_name, temperature) not in cls._instance:
            super(OllamaLLMModel, cls).__new__(cls)

            model = ChatOllama(model="llama3.2:1b", temperature=temperature)
            cls._instance[(model_name, temperature)] = model

        return cls._instance[(model_name, temperature)]

    def get_model(self):
        return self.model


def get_ollama_llm_instance(model_name, temperature):
    # Set InMemoryCache
    set_llm_cache(InMemoryCache())
    ollama_llm_instance = OllamaLLMModel(model_name, temperature)
    return ollama_llm_instance

def get_converstation_ollama(prompt, llm):
    memory = ConversationBufferMemory(llm=llm)
    conversation = RunnableWithMessageHistory(
        runnable=llm,
        get_session_history=lambda: memory.load_memory_variables({})["history"],
        memory=memory,
        prompt=prompt,
        input_variables=["title", "user_input"]
    )
    return conversation
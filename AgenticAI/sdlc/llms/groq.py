from langchain_groq import ChatGroq


class GroqLLMModel:
    _instance = {}

    def __new__(cls, model_name, api_key, temperature):
        if (model_name, api_key, temperature) not in cls._instance:
            super(GroqLLMModel, cls).__new__(cls)
            model = ChatGroq(model_name=model_name, api_key=api_key, temperature=temperature)

            cls._instance[(model_name, api_key, temperature)] = model

        return cls._instance[(model_name, api_key, temperature)]

    def get_model(self):
        return self.model


def get_groq_llm_instance(model_name, api_key, temperature):
    groq_llm_instance = GroqLLMModel(model_name, api_key, temperature)
    return groq_llm_instance
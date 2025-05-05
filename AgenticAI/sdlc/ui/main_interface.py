import streamlit as st
from dataclasses import dataclass
from llms.ollama import get_ollama_llm_instance
from ui.base_interface import BaseInterface
from ui.product_owner_interface import ProductOwnerUI
from ui.design_owner_interface import DesignOwnerUI
from ui.developer_interface import DeveloperUI
from ui.security_owner_interface import SecurityUI
from ui.unit_tester_interface import UnitTesterUI
from ui.qa_tester_interface import QATesterUI


@dataclass
class AppState:
    """Class for keeping track of SDLC."""
    current_step: str = "Product Owner"
    selected_llm: str = ""
    selected_model: str = ""
    GROQ_API_KEY: str = ""


class SDLCInterface(BaseInterface):

    def _get_ui_config(self):
        return {
            "app": AppState()
        }

    def set_model_state(self):
        # Get options from config
        llm_options = self.config.get_llm_options()

        # LLM selection
        st.session_state.app.selected_llm = st.selectbox("Select LLM", llm_options)

        if st.session_state.app.selected_llm == 'Groq':
            # Model selection
            model_options = self.config.get_groq_model_options()
            st.session_state.app.selected_model = st.selectbox("Select Model", model_options)
            # API key input
            st.session_state.app.GROQ_API_KEY = st.session_state["GROQ_API_KEY"] = st.text_input("API Key", "", type="password")
            # Validate API key
            if not st.session_state.app.GROQ_API_KEY:
                st.warning("⚠️ Please enter your GROQ API key to proceed. Don't have? refer : https://console.groq.com/keys ")
            else:
                model = "llama3.2:1b"
                model = get_ollama_llm_instance(model, 0.3)
                st.session_state.app.llm = model

        elif st.session_state.app.selected_llm == 'Ollama':
            # Model selection
            model_options = self.config.get_ollama_model_options()
            st.session_state.app.selected_model = st.selectbox("Select Model", model_options)
            model = get_ollama_llm_instance(st.session_state.app.selected_model, 0.7)
            st.session_state.app.llm = model

    def set_main_interface(self):
        section = st.session_state.app.current_step

        # Load respective UI based on selection
        if section == "Product Owner":
            ProductOwnerUI().load_ui()
        elif section == "Designer":
            DesignOwnerUI().load_ui()
        elif section == "Developer":
            DeveloperUI().load_ui()
        elif section == "Security":
            SecurityUI().load_ui()
        elif section == "Unit Test":
            UnitTesterUI().load_ui()
        elif section == "QA Tester":
            QATesterUI().load_ui()

    def load_streamlit_ui(self):
        st.set_page_config(page_title= "🤖 " + self.config.get_page_title(), layout="wide")
        st.header("🤖 " + self.config.get_page_title())

        with st.sidebar:
            # Get options from config
            usecase_options = self.config.get_usecase_options()
            self.set_model_state()
            # Use case selection
            st.session_state.app.selected_usecase = st.selectbox(
                "Select Usecases",
                usecase_options,
                index=usecase_options.index(st.session_state.app.current_step) if st.session_state.get("state") else 0
            )

        self.set_main_interface()

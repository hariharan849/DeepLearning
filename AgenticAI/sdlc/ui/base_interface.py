import streamlit as st
from utils import UIConfig

class BaseInterface:
    def __init__(self):
        self.config =  UIConfig()
        self._sesstion_state_set_default_key()

    def _get_ui_config(self):
        return {}
    
    def _sesstion_state_set_default_key(self):
        for key, value in self._get_ui_config().items():
            if key not in st.session_state:
                st.session_state[key] = value

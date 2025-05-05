import streamlit as st
from dataclasses import dataclass
from graph.security_graph import SecurityWorkflow
from state.security_owner import SecurityState
from state.developer_doc import DeveloperDocState
from ui.base_interface import BaseInterface
from utils import highlight_diff


@dataclass
class SecurityOwnerUIState:
    owner: SecurityWorkflow = None
    owner_state: SecurityState = None

class SecurityUI(BaseInterface):

    def _get_ui_config(self):
        return {
            "security_owner": SecurityOwnerUIState()
        }

    def load_ui(self):
        st.title("Security Interface")
        st.write("This is the Security Interface. Here, you can provide feedback for the generated code.")

        if not st.session_state.security_owner.owner:
            security_owner_assistant = SecurityWorkflow(st.session_state.app.llm)
            st.session_state.security_owner.owner = security_owner_assistant
            security_owner_assistant.build_graph()
            response = st.session_state.security_owner.owner.executor.invoke(
                {
                    "developer": st.session_state.developer_owner.owner_state
                }
            )
            st.session_state.security_owner.owner_state =  SecurityState(
                security_issues=response["security_issues"], developer=response["developer"]
            )

        if st.session_state.security_owner.owner_state:
            generated_security_issues = st.session_state.security_owner.owner_state.security_issues

            if not generated_security_issues:
                st.write("No pending security code for review.")
            else:
                with st.expander(f"Generated Security concern", expanded=True):
                    st.markdown(generated_security_issues, unsafe_allow_html=True)
                    if st.session_state.security_owner.owner_state.reviewed_security_issues:
                        st.markdown("### Changes in the Story")
                        diff_html = highlight_diff(generated_security_issues.reviewed_security_issues, generated_security_issues.security_issues)
                        st.markdown(diff_html, unsafe_allow_html=True)
                    feedback = st.text_area(f"Reviewer Security Feedback:", "", height=400, key=f"feedback")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Approve", key=f"approve"):
                            st.session_state.security_owner.owner_state.human_approved = 1
                            st.success("Story approved and stored!")

                            st.session_state.app.current_step = "Unit Test"
                            # Force Streamlit to rerun the UI to reflect the update
                            st.rerun()

                    with col2:
                        if st.button("Refine", key=f"refine"):
                            st.session_state.security_owner.owner_state.human_approved = -1
                            st.session_state.security_owner.owner_state.feedback = feedback

                            generated_code = st.session_state.developer_owner.owner_state.code
                            generated_code.security_issues = generated_security_issues
                            generated_code.security_feedback = feedback
                            response = st.session_state.developer_owner.owner.executor.invoke({
                                "designer": st.session_state.design_owner.owner_state,
                                "code": generated_code
                            })
                            st.session_state.developer_owner.owner_state = DeveloperDocState(
                                designer=st.session_state.design_owner.owner_state.model_dump(),
                                code=response["code"].model_dump()
                            )
                            st.warning("Code Review sent for refinement based on feedback.")
                            st.session_state.app.current_step = "Developer"
                            # Force Streamlit to rerun the UI to reflect the update
                            st.rerun()
                            
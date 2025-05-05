import streamlit as st
from dataclasses import dataclass
from graph.qa_test_graph import QATestWorkflow
from state.qa_tester import QATestState
from state.developer_doc import DeveloperDocState
from ui.base_interface import BaseInterface

@dataclass
class QATestUIState:
    owner: QATestWorkflow = None
    owner_state: QATestState = None

class QATesterUI(BaseInterface):
    def _get_ui_config(self):
        return {
            "qa_tester": QATestUIState()
        }

    def load_ui(self):
        st.title("QA Tester Interface")
        st.write("This is the QA Tester Interface. Here, you can provide feedback for the generated test code.")

        if not st.session_state.qa_tester.owner:
            qa_tester_assistant = QATestWorkflow(st.session_state.app.llm)
            st.session_state.qa_tester.owner = qa_tester_assistant
            qa_tester_assistant.build_graph()

            response = st.session_state.qa_tester.owner.executor.invoke(
                {
                    "developer": st.session_state.developer_owner.owner_state,
                    "unit_test_code": st.session_state.unit_tester.owner_state.unit_test_code,
                    "security_issue": st.session_state.security_owner.owner_state.security_issues
                }
            )

            st.session_state.qa_tester.owner_state = QATestState(**response)


        if st.session_state.qa_tester.owner_state.unit_test_code:
            generated_qa_test = st.session_state.unit_tester.owner_state.unit_test_code

            if not generated_qa_test:
                st.write("No pending QA test code for review.")
            else:
                with st.expander(f"Generated QA Test", expanded=True):
                    st.markdown(generated_qa_test, unsafe_allow_html=True)
                    feedback = st.text_area(f"Reviewer QA Tester Feedback:", "", height=100, key=f"feedback")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Approve", key=f"approve"):
                            st.session_state.qa_tester.owner_state.human_approved = 1
                            st.success("QA approved and stored!")
                            st.stop()

                    with col2:
                        if st.button("Refine", key=f"refine"):
                            st.session_state.qa_tester.owner_state.human_approved = -1
                            st.session_state.qa_tester.owner_state.feedback = feedback

                            generated_code = st.session_state.developer_owner.owner_state.code
                            generated_code.qa_issues = generated_qa_test
                            generated_code.qa_feedback = feedback
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

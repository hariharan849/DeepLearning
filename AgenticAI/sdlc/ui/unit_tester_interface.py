import streamlit as st
from dataclasses import dataclass
from graph.unit_test_graph import UnitTestWorkflow
from state.unit_tester import UnitTestState
from ui.base_interface import BaseInterface
from utils import highlight_diff

@dataclass
class UnitTestUIState:
    owner: UnitTestWorkflow = None
    owner_state: UnitTestState = None

class UnitTesterUI(BaseInterface):
    def _get_ui_config(self):
        return {
            "unit_tester": UnitTestUIState()
        }

    def load_ui(self):
        st.title("Unit Tester Interface")
        st.write("This is the Unit Tester Interface. Here, you can provide feedback for the generated test code.")

        if not st.session_state.unit_tester.owner:
            unit_tester_assistant = UnitTestWorkflow(st.session_state.app.llm)
            st.session_state.unit_tester.owner = unit_tester_assistant
            unit_tester_assistant.build_graph()

            response = st.session_state.unit_tester.owner.executor.invoke(
                {
                    "developer": st.session_state.developer_owner.owner_state
                }
            )
            st.session_state.unit_tester.owner_state = UnitTestState(
                developer=response["developer"], unit_test_code=response["unit_test_code"]
            )


        if st.session_state.unit_tester.owner_state.unit_test_code:
            generated_unit_test = st.session_state.unit_tester.owner_state.unit_test_code

            if not generated_unit_test:
                st.write("No pending unit test code for review.")
            else:
                with st.expander(f"Generated Unit Test", expanded=True):
                    st.markdown(generated_unit_test, unsafe_allow_html=True)
                    if st.session_state.unit_tester.owner_state.reviewed_unit_test_issues:
                        st.markdown("### Changes in the Story")
                        diff_html = highlight_diff(st.session_state.unit_tester.owner_state.reviewed_unit_test_issues, generated_unit_test)
                        st.markdown(diff_html, unsafe_allow_html=True)
                    feedback = st.text_area(f"Reviewer Unit Tester Feedback:", "", height=100, key=f"feedback")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Approve", key=f"approve"):
                            st.session_state.unit_tester.owner_state.human_approved = 1
                            st.success("Story approved and stored!")

                            st.session_state.app.current_step = "QA Tester"
                            # Force Streamlit to rerun the UI to reflect the update
                            st.rerun()

                    with col2:
                        if st.button("Refine", key=f"refine"):
                            st.session_state.unit_tester.owner_state.human_approved = -1
                            st.session_state.unit_tester.owner_state.feedback = feedback

                            response = st.session_state.unit_tester.owner.executor.invoke(
                                {
                                    "developer": st.session_state.developer_owner.owner_state,
                                    "unit_test_code": generated_unit_test,
                                    "feedback":feedback
                                }
                            )
                            st.session_state.unit_tester.owner_state = UnitTestState(
                                developer=response["developer"], unit_test_code=response["unit_test_code"],
                                human_approved=0, feedback=response["feedback"]
                            )

                            # generated_code = st.session_state.developer_owner.owner_state.code
                            # generated_code.unit_test = generated_unit_test
                            # response = st.session_state.developer_owner.owner.executor.invoke({
                            #     "designer": st.session_state.design_owner.owner_state,
                            #     "code": generated_code
                            # })
                            # st.session_state.developer_owner.owner_state = DeveloperDocState(
                            #     designer=st.session_state.design_owner.owner_state.model_dump(),
                            #     code=response["code"].model_dump()
                            # )
                            st.warning("Code Review sent for refinement based on feedback.")
                            # Force Streamlit to rerun the UI to reflect the update
                            st.rerun()

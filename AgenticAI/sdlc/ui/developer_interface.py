import streamlit as st
from dataclasses import dataclass, field
from utils import deploy_to_github, upload_pr_to_jira, highlight_diff
from graph.developer_graph import DeveloperWorkflow
from state.developer_doc import DeveloperDocState
from ui.base_interface import BaseInterface

@dataclass
class DeveloperOwnerUIState:
    owner: DeveloperWorkflow = None
    owner_state: DeveloperDocState = None

class DeveloperUI(BaseInterface):

    def _get_ui_config(self):
        return {
            "developer_owner": DeveloperOwnerUIState()
        }

    def load_ui(self):
        st.title("Developer Interface")
        st.write("This is the Developer Interface. Here, you can provide feedback for the generated code.")

        if not st.session_state.developer_owner.owner:
            developer_owner_assistant = DeveloperWorkflow(st.session_state.app.llm)
            st.session_state.developer_owner.owner = developer_owner_assistant
            developer_owner_assistant.build_graph()

            response = st.session_state.developer_owner.owner.executor.invoke(
                {
                    "designer": st.session_state.design_owner.owner_state
                }
            )
            st.session_state.developer_owner.owner_state =  DeveloperDocState(
                code=response["code"], designer=response["designer"]
            )

        
        if st.session_state.developer_owner.owner_state:
            generated_code = st.session_state.developer_owner.owner_state.code

            if not generated_code:
                st.write("No pending code for review.")
            else:
                with st.expander(f"Developer Code", expanded=True):
                    st.markdown(generated_code.code, unsafe_allow_html=True)
                    if generated_code.reviewed_code:
                        st.markdown("### Changes in the Story")
                        diff_html = highlight_diff(generated_code.reviewed_code, generated_code.code)
                        st.markdown(diff_html, unsafe_allow_html=True)
                    st.markdown(generated_code.auto_review, unsafe_allow_html=True)
                    feedback = st.text_area(f"Reviewer Feedback:", "", height=100, key=f"feedback")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Approve", key=f"approve"):
                            generated_code.human_approved = 1
                            # Save the generated code to a temporary file
                            file_path = "./generated_code.md"
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(generated_code.code)
                            # Get Jira ticket from session state
                            jira_ticket = st.session_state.design_owner.owner_state.story.jira_ticket

                            if jira_ticket:
                                # Push to GitHub & create PR
                                pr_link = deploy_to_github(file_path, "src/generated_code.md", "Deploy generated code", jira_ticket)

                                if pr_link:
                                    # Update Jira with PR link
                                    upload_pr_to_jira(jira_ticket, pr_link)
                                    st.success(f"✅ Code deployed & PR created: {pr_link}")
                            else:
                                st.success("Story approved and stored!")

                            st.session_state.app.current_step = "Security"
                            # Force Streamlit to rerun the UI to reflect the update
                            st.rerun()

                    with col2:
                        if st.button("Refine", key=f"refine"):
                            generated_code.human_approved = -1
                            generated_code.feedback = feedback
                            #todo
                            response = st.session_state.developer_owner.owner.executor.invoke({
                                "designer": st.session_state.design_owner.owner_state,
                                "code": generated_code
                            })

                            st.session_state.developer_owner.owner_state = DeveloperDocState(
                                designer=st.session_state.design_owner.owner_state.model_dump(),
                                code=response["code"].model_dump()
                            )
                            st.session_state.developer_owner.owner_state.code.feedback = ""
                            st.rerun()
                            st.warning("Code Review sent for refinement based on feedback.")

import streamlit as st
from dataclasses import dataclass, field
from graph.design_owner_graph import DesignDocumentWorkflow
from state.design_doc import DesignDocState
from ui.base_interface import BaseInterface
from utils import highlight_diff, upload_design_doc

@dataclass
class DesignOwnerUIState:
    owner: DesignDocumentWorkflow = None
    owner_state: DesignDocState = None

class DesignOwnerUI(BaseInterface):
    def _get_ui_config(self):
        return {
            "design_owner": DesignOwnerUIState()
        }

    def load_ui(self):
        st.title("Design Owner Interface")
        st.write("This is the Design Owner Interface. Here, you can provide requirements, user stories, and feedback for the generated code.")

        if not st.session_state.design_owner.owner:
            design_owner_assistant = DesignDocumentWorkflow(st.session_state.app.llm)
            st.session_state.design_owner.owner = design_owner_assistant
            design_owner_assistant.build_graph()

            response = st.session_state.design_owner.owner.executor.invoke({"story": st.session_state.product_owner.owner_state})
            st.session_state.design_owner.owner_state =  DesignDocState(**response)

        if st.session_state.design_owner.owner_state:
            design_doc = st.session_state.design_owner.owner_state.design_docs

            if not design_doc:
                st.write("No pending design docs for review.")
            else:
                with st.expander(f"Design Doc", expanded=True):
                    st.write(design_doc.doc)
                    if design_doc.reviewed_doc:
                        st.markdown("### Changes in the Design documents")
                        diff_html = highlight_diff(design_doc.reviewed_doc, design_doc.doc)
                        st.markdown(diff_html, unsafe_allow_html=True)
                    feedback = st.text_area(f"Reviewer Feedback:", "", height=100)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Approve"):
                            with st.spinner("Processing Title..."):
                                st.toast("Approving story... Please wait.")
                                design_doc.human_approved = 1

                                # Save the updated design doc locally
                                file_path = "./design_doc.txt"
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(design_doc.doc)

                                if st.session_state.design_owner.owner_state.story.jira_ticket:
                                    upload_design_doc(st.session_state.design_owner.owner_state.story.jira_ticket, file_path)

                                st.success("Story approved and stored!")

                                st.session_state.app.current_step = "Developer"
                                # Force Streamlit to rerun the UI to reflect the update
                                st.rerun()
                            
                    with col2:
                        if st.button("Refine"):
                            design_doc.human_approved = -1
                            design_doc.feedback = feedback
                            #todo
                            response = st.session_state.design_owner.owner_state = st.session_state.design_owner.owner.executor.invoke(
                                st.session_state.design_owner.owner_state.model_dump()
                            )
                            st.session_state.design_owner.owner_state =  DesignDocState(**response)
                            st.session_state.design_owner.owner_state.design_docs.feedback = ""
                            st.warning("Story sent for refinement based on feedback.")
                            st.rerun()

import streamlit as st
from dataclasses import dataclass, field
from llms.ollama import get_converstation_ollama
from ui.base_interface import BaseInterface
from state.product_owner import UserStory
from graph import product_owner_graph
from graph.product_owner_graph import ProductOwnerWorkflow
from prompts.product_owner_prompt import ProductOwnerPrompt
from langchain.memory import ConversationBufferMemory
from utils import highlight_diff, create_jira_ticket

@dataclass
class ProductOwnerUIState:
    title_processed: bool = False
    title: str = ""
    title_summary: str = ""
    feedback: str = ""
    owner: ProductOwnerWorkflow = None
    owner_state: UserStory = None
    memory: ConversationBufferMemory = None


class ProductOwnerUI(BaseInterface):

    def _get_ui_config(self):
        return {
            "product_owner": ProductOwnerUIState()
        }

    def _handle_feature_req_load(self, title):
        if not st.session_state.product_owner.title_processed:

            st.session_state.product_owner.title = title
            st.session_state.product_owner.title_processed = True

            self._title_summary_ui()

    def _title_summary_ui(self):

        if st.button("Generate User stories"):
            with st.spinner("Processing Title..."):
                product_owner_assistant = product_owner_graph.ProductOwnerWorkflow(st.session_state.app.llm)
                product_owner_assistant.build_graph()
                st.session_state.product_owner.owner = product_owner_assistant

                if not st.session_state.product_owner.owner_state:
                    response = st.session_state.product_owner.owner.executor.invoke({"feature_request": st.session_state.product_owner.title})
                    st.session_state.product_owner.owner_state = UserStory(**response)

        if st.session_state.product_owner.owner_state:
            if not st.session_state.product_owner.owner_state.generated_story:
                st.write("No pending user stories for review.")
            else:
                story = st.session_state.product_owner.owner_state
                with st.expander(f"Feature Request: {story.feature_request}", expanded=True):
                    if story.reviewed_story:
                        st.markdown("### Changes in the Story")
                        diff_html = highlight_diff(story.reviewed_story, story.generated_story)
                        st.markdown(diff_html, unsafe_allow_html=True)

                    st.text_area("User Story:", story.generated_story, height=500, disabled=True)
                    feedback = st.text_area(f"Reviewer Feedback:", "", height=100)
                    
                    if story.review_cycles >= 9:
                        st.warning(f"⚠️ This story is nearing the 10-cycle limit ({story.review_cycles}/10). Final review required!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Approve"):
                            with st.spinner("Processing Title..."):
                                st.toast("Approving story... Please wait.")
                                story.human_approved = 1
                                create_jira_ticket(story)
                                 # Display Jira ticket if created
                                if story.jira_ticket:
                                    st.success(f"Story approved! Jira Ticket: {story.jira_ticket}")
                                else:
                                    st.error("Story approved, but Jira ticket creation failed!")
                                st.session_state.app.current_step = "Designer"
                                # Force Streamlit to rerun the UI to reflect the update
                                st.rerun()

                    with col2:
                        if st.button("Refine"):
                            st.toast("Refining story... Please wait.")
                            if story.review_cycles < 10:
                                story.feedback = feedback
                                story.human_approved = -1
                                story.review_cycles += 1
                                
                                response = st.session_state.product_owner.owner.executor.invoke(
                                    story.model_dump()
                                )
                                st.session_state.product_owner.owner_state = UserStory(**response)
                                st.session_state.product_owner.owner_state.feedback = ""
                                st.warning("Story sent for refinement based on feedback.")
                                st.rerun()
                            else:
                                st.error("Story reached max review limit and is removed.")


    def load_ui(self):
        st.title("Product Owner Interface")
        st.write("This is the Product Owner Interface. Here, you can provide requirements, user stories, and feedback for the generated code.")

        title = st.text_input("Enter Feature Request:", "Find Longest Substring Without Repeating Characters?")

        if st.session_state.product_owner.title_processed is False:
            self._handle_feature_req_load(title)
        else:
            self._title_summary_ui()

        st.header("Chat with the Video Summary")

        if title:
            # Ensure message history exists
            if not st.session_state.product_owner.memory:
                st.session_state.product_owner.memory = get_converstation_ollama(ProductOwnerPrompt.chat_prompt, st.session_state.app.llm)

            # Display previous messages
            # for message in st.session_state["messages"]:
            #     with st.chat_message(message["role"]):
            #         st.markdown(message["content"])

            # Chat Input
            user_input = st.chat_input("Ask something about the feature request...")

            if user_input:
                # Generate AI Response
                response = st.session_state.product_owner.memory.invoke({"title": title, "user_input": user_input})

                ai_response = response.content if hasattr(response, "content") else str(response)
                # st.session_state["messages"].append({"role": "assistant", "content": ai_response})

                # Display AI Response
                with st.chat_message("assistant"):
                    st.markdown(ai_response)

        else:
            st.info("Please confirm the summary first before using the chatbot.")
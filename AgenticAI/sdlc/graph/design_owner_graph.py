from langgraph.graph import StateGraph, END
from state import design_doc
from prompts import design_owner_prompt
from langchain_core.output_parsers import StrOutputParser

class DesignDocumentWorkflow():
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(design_doc.DesignDocState)

    def build_graph(self):
        """Create design document workflow."""
        self.graph.add_node("generate_design_docs", self.generate_design_docs)
        self.graph.add_node("refine_rejected_design_doc", self.refine_rejected_design_doc)
        
        # Define entry and exit points
        self.graph.set_entry_point("generate_design_docs")
        
        self.graph.add_edge("generate_design_docs", "refine_rejected_design_doc")
        self.graph.add_edge("refine_rejected_design_doc", END)

        self.executor = self.graph.compile()

    def generate_design_docs(self, state):
        """Generate design documents from approved stories."""
        if state.design_docs:
            return {"design_docs": state.design_docs, "story": state.story}
        story = state.story
        parser = StrOutputParser()
        
        # Invoke LLM to generate design docs
        chain = design_owner_prompt.DesignOwnerPrompt.design_prompt | self.llm | parser
        response = chain.invoke({"story_text": story.generated_story, "feature_request": story.feature_request})
        
        design_document = design_doc.DesignDocument(doc=response)

        return {"design_docs": design_document, "story": story}
    
    def refine_rejected_design_doc(self, design_docs: dict):
        design_document = design_docs.design_docs
        if not design_document.feedback:
            return design_document
        
        story = design_docs.story
        prompt = design_owner_prompt.DesignOwnerPrompt.feedback_prompt
        req = {
            "feature_request": story.feature_request,
            "feedback": story.feedback,
            "user_story": story.generated_story,
            "design_doc": design_document.doc
        }

        output_parser = StrOutputParser()
        chain = prompt | self.llm | output_parser
        response = chain.invoke(req)

        if not response:
            return design_doc.DesignDocState(
                story=story,
                design_docs=design_doc.DesignDocument(
                    doc="",
                    reviewed_doc=design_document.doc,
                    human_approved=0,
                    feedback=design_document.feedback
                )
            )

        return design_doc.DesignDocState(
                story=story,
                design_docs=design_doc.DesignDocument(
                    doc=response,
                    reviewed_doc=design_document.doc,
                    human_approved=0,
                    feedback=design_document.feedback
                )
            )

from langgraph.graph import StateGraph, END
from state import developer_doc
from prompts.developer_prompt import DeveloperOwnerPrompt
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

class DeveloperWorkflow():
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(developer_doc.DeveloperDocState)

    def build_graph(self):
        """Create design document workflow."""
        self.graph.add_node("generate_code", self.generate_code)
        self.graph.add_node("code_review", self.code_review)
        
        # Define entry and exit points
        self.graph.set_entry_point("generate_code")
        self.graph.add_edge("generate_code", "code_review")
        self.graph.add_edge("code_review", END)

        self.executor = self.graph.compile()

    def generate_code(self, state):
        """Generate design documents from approved stories."""
        story = state.designer.story
        design_docs = state.designer.design_docs
        generated_code_obj = state.code

        if generated_code_obj:
            
            if generated_code_obj.qa_issues:
                req = {
                    "story_text": story.generated_story, "design_doc": design_docs,
                    "security_feedback": generated_code_obj.security_feedback, "current_code": generated_code_obj.code,
                    "security_issues": generated_code_obj.security_issues,
                    "unit_test_issues": generated_code_obj.unit_test_issues, "qa_test_feedback": generated_code_obj.qa_test_feedback
                }
                prompt = DeveloperOwnerPrompt.code_developer_qa_test_prompt
            elif generated_code_obj.security_issues:
                req = {
                    "story_text": story.generated_story, "design_doc": design_docs,
                    "security_feedback": generated_code_obj.security_feedback, "current_code": generated_code_obj.code,
                    "security_issues": generated_code_obj.security_issues
                }
                prompt = DeveloperOwnerPrompt.code_developer_security_prompt
            else:
                req = {
                    "story_text": story.generated_story, "design_doc": design_docs,
                    "feedback": generated_code_obj.feedback, "current_code": generated_code_obj.code
                }
                prompt = DeveloperOwnerPrompt.code_update_prompt
            parser = StrOutputParser()
            
            # Invoke LLM to generate design docs
            chain = prompt | self.llm | parser
            response = chain.invoke(req)
            generated_code = developer_doc.CodeResponse(code=response, reviewed_code=generated_code_obj.code)
        else:
            
            req = {"story_text": story.generated_story, "design_doc": design_docs}

            #TODO
            # parser = PydanticOutputParser(pydantic_object=design_doc.DesignDocument)
            parser = StrOutputParser()
            
            # Invoke LLM to generate design docs
            chain = DeveloperOwnerPrompt.code_developer_prompt | self.llm | parser
            response = chain.invoke(req)
            
            generated_code = developer_doc.CodeResponse(code=response)

        return {"code": generated_code}
    
    def code_review(self, state):
        parser = StrOutputParser()
        # Invoke LLM to generate design docs
        chain = DeveloperOwnerPrompt.code_review_prompt | self.llm | parser
        response = chain.invoke({"code": state.code.code})
        state.code.auto_review = response

        return state
    
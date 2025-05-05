from langgraph.graph import StateGraph, END
from state import security_owner
from prompts.security_prompt import SecurityPrompt
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

class SecurityWorkflow():
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(security_owner.SecurityState)

    def build_graph(self):
        """Create design document workflow."""
        self.graph.add_node("security_review", self.security_review)
        
        # Define entry and exit points
        self.graph.set_entry_point("security_review")
        self.graph.add_edge("security_review", END)

        self.executor = self.graph.compile()

    def security_review(self, sec_state):
        """Generate design documents from approved stories."""
        if sec_state.feedback:
            parser = StrOutputParser()
            generated_code = sec_state.developer.code
            # Invoke LLM to generate design docs
            chain = SecurityPrompt.feedback_prompt | self.llm | parser
            response = chain.invoke({"code": generated_code, "feedback": sec_state.feedback})
            sec_state.feedback = response
            return {"security_issues": response}
        else:
            parser = StrOutputParser()
            generated_code = sec_state.developer.code
            # Invoke LLM to generate design docs
            chain = SecurityPrompt.security_prompt | self.llm | parser
            response = chain.invoke({"code": generated_code})
            return {"security_issues": response}
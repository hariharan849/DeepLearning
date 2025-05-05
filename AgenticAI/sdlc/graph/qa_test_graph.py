from langgraph.graph import StateGraph, END
from state import qa_tester
from prompts.qa_tester_prompt import QATesterPrompt
from langchain_core.output_parsers import StrOutputParser

class QATestWorkflow():
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(qa_tester.QATestState)

    def build_graph(self):
        """Create design document workflow."""
        self.graph.add_node("start_qa_test", self.generate_qa_test)
        
        # Define entry and exit points
        self.graph.set_entry_point("start_qa_test")
        self.graph.add_edge("start_qa_test", END)

        self.executor = self.graph.compile()

    def generate_qa_test(self, unit_test_state):
        """Generate design documents from approved stories."""
        parser = StrOutputParser()
            
        # Invoke LLM to generate design docs
        chain = QATesterPrompt.qa_test_feedback_prompt | self.llm | parser
        response = chain.invoke(
            {
                "code": unit_test_state.developer.code,
                "security_fixes": unit_test_state.security_issue,
                "unit_tests": unit_test_state.unit_test_code
            }
        )
        unit_test_state.unit_test_code = response
        return unit_test_state
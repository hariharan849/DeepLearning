from langgraph.graph import StateGraph, END
from state import unit_tester
from prompts.unit_tester_prompt import UnitTesterPrompt
from langchain_core.output_parsers import StrOutputParser

class UnitTestWorkflow():
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(unit_tester.UnitTestState)

    def build_graph(self):
        """Create design document workflow."""
        self.graph.add_node("generate_unit_test", self.generate_unit_test)
        
        # Define entry and exit points
        self.graph.set_entry_point("generate_unit_test")
        self.graph.add_edge("generate_unit_test", END)

        self.executor = self.graph.compile()

    def generate_unit_test(self, unit_test_state):
        """Generate design documents from approved stories."""
        if unit_test_state.feedback:
            parser = StrOutputParser()
            chain = UnitTesterPrompt.unit_test_feedback_prompt | self.llm | parser
            response = chain.invoke({"existing_code": unit_test_state.developer.code, "existing_tests": unit_test_state.unit_test_code, "feedback": unit_test_state.feedback})
            unit_test_state.reviewed_unit_test_issues = unit_test_state.unit_test_code
            unit_test_state.unit_test_code = response
            return unit_test_state

        else:
            parser = StrOutputParser()
                
            # Invoke LLM to generate design docs
            chain = UnitTesterPrompt.unit_test_prompt | self.llm | parser
            response = chain.invoke({"code": unit_test_state.developer.code})
            unit_test_state.unit_test_code = response
            return unit_test_state
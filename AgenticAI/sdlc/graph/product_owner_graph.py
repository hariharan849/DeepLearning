import json
from collections import ChainMap  
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts.product_owner_prompt import ProductOwnerPrompt
from state import product_owner
from typing import Dict, List



from pydantic import BaseModel, Field

class Story(BaseModel):
    feature_request: str
    story_text: str
    acceptance_criteria: list
    example_use_cases: list
    non_functional_requirements: list

STORY_FORMAT = """
    {story_text}
    \n\n
    "Acceptance Criteria:"
    {acceptance_criteria}
    \n\n
    "Use Cases:"
    {example_use_cases}
    \n\n
    "Others:"
    {non_functional_requirements}
"""

class ProductOwnerWorkflow():
    def __init__(self, llm: ChatGoogleGenerativeAI):
        super().__init__()
        self.llm = llm

        self.graph = StateGraph(product_owner.UserStory)
        self.executor = None

    
    def build_graph(self):
        # Add nodes
        self.graph.add_node("generate_user_stories", self.generate_stories)
        # self.graph.add_node("review_user_story", self.review_stories)
        self.graph.add_node("refine_rejected_stories", self.refine_rejected_stories)

        self.graph.add_edge("generate_user_stories", "refine_rejected_stories")
        self.graph.add_edge("refine_rejected_stories", END)
        # self.graph.add_edge("refine_rejected_stories", "review_user_story")
        # self.graph.add_conditional_edges(
        #     "review_user_story",
        #     lambda result: "refine_rejected_stories" if result.get("rejected") else END,
        #     "refine_rejected_stories"
        # )

        self.graph.set_entry_point("generate_user_stories")
        self.executor = self.graph.compile()

    def generate_stories(self, feature_request: dict) -> product_owner.UserStory:
        """Generate multiple user stories for a feature request."""
        story_map = {}

        #Reinvoking for refining stories
        if feature_request.feedback:
            return feature_request

        prompt = ProductOwnerPrompt.user_story_prompt
        req = {"feature_request": feature_request}
        
        feature_request = feature_request.feature_request

        output_parser = JsonOutputParser()
        chain = prompt | self.llm
        response = chain.invoke(req).content

        if not isinstance(response, dict):
            response = output_parser.parse(response)

        llm_stories = response['stories']
        if not llm_stories:
            return product_owner.UserStory(
                feature_request=feature_request,
                generated_story="Error: Unable to parse response.",
                summary="",
                human_approved=0,
                jira_ticket="",
                review_cycles=0,
                feedback=""
            )

        #filter user stories based on
        llm_with_tool = self.llm.with_structured_output(product_owner.Grade)
        product_owner_chain = ProductOwnerPrompt.po_grader_prompt | llm_with_tool
        user_story_chain = ProductOwnerPrompt.detailed_story_prompt | self.llm | PydanticOutputParser(pydantic_object=Story)

        # Split response into multiple user stories

        for story_text in llm_stories:
            if story_text["story_text"].strip():  # Ignore empty responses
                story_text = story_text["story_text"]
                scored_result = product_owner_chain.invoke({"story": story_text, "feature_request": feature_request})
                #TODO: not sure why its none
                if not scored_result:
                    return None
                score = scored_result.binary_score
                if score == "yes":

                    response = user_story_chain.invoke({"summary": story_text.strip()})
                    #Todo: fix hard coding
                    if response.example_use_cases and isinstance(response.example_use_cases[0], dict):
                        example_use_cases = "\n".join(response.example_use_cases.values()
                        )
                    else:
                        example_use_cases = "\n".join(response.example_use_cases)

                    if response.acceptance_criteria and isinstance(response.acceptance_criteria[0], dict):
                        acceptance_criteria = "\n".join(response.acceptance_criteria.values())
                    else:
                        acceptance_criteria = "\n".join(response.acceptance_criteria)

                    return product_owner.UserStory(
                            feature_request=feature_request,
                            generated_story=STORY_FORMAT.format(**{
                                "story_text": response.story_text,
                                "acceptance_criteria": acceptance_criteria,
                                "example_use_cases": example_use_cases,
                                "non_functional_requirements": "\n".join(response.non_functional_requirements),
                            }),
                            reviewed_summary="",
                            human_approved=0,
                            jira_ticket="",
                            review_cycles=0,
                            feedback=""
                        )

    
    def review_stories(self, story: product_owner.UserStory) -> Dict[str, product_owner.UserStory]:
        """Send stories for human review. Approved stories are separated from rejected ones."""

        if story.human_approved == 1:
            return {
                "approved": story,
                "rejected": None
            }
        elif story.human_approved == -1:
            return {
                "approved": None,
                "rejected": story
            }


    def refine_rejected_stories(self, story: product_owner.UserStory) -> product_owner.UserStory:
        """Refine rejected user stories based on feedback."""
        if not story.feedback:
            return story
        prompt = ProductOwnerPrompt.user_story_feedback_prompt
        req = {"feature_request": story.feature_request, "feedback": story.feedback, "user_story": story.generated_story}
        
        feature_request = story.feature_request

        output_parser = JsonOutputParser()
        chain = prompt | self.llm
        response = chain.invoke(req).content

        if not isinstance(response, dict):
            response = output_parser.parse(response)

        llm_stories = response['stories']
        if not llm_stories:
            return product_owner.UserStory(
                feature_request=feature_request,
                generated_story="Error: Unable to parse response.",
                summary="",
                human_approved=0,
                jira_ticket="",
                review_cycles=0,
                feedback=""
            )

        #filter user stories based on
        llm_with_tool = self.llm.with_structured_output(product_owner.Grade)
        product_owner_chain = ProductOwnerPrompt.po_grader_prompt | llm_with_tool
        user_story_chain = ProductOwnerPrompt.detailed_story_prompt | self.llm | PydanticOutputParser(pydantic_object=Story)

        # Split response into multiple user stories

        for story_text in llm_stories:
            if story_text["story_text"].strip():  # Ignore empty responses
                story_text = story_text["story_text"]
                scored_result = product_owner_chain.invoke({"story": story_text, "feature_request": feature_request})
                #TODO: not sure why its none
                if not scored_result:
                    return None
                score = scored_result.binary_score
                if score == "yes":

                    response = user_story_chain.invoke({"summary": story_text.strip()})
                    #Todo: fix hard coding
                    if response.example_use_cases and isinstance(response.example_use_cases[0], dict):
                        example_use_cases = "\n".join(response.example_use_cases.values()
                        )
                    else:
                        example_use_cases = "\n".join(response.example_use_cases)

                    if response.acceptance_criteria and isinstance(response.acceptance_criteria[0], dict):
                        acceptance_criteria = "\n".join(response.acceptance_criteria.values())
                    else:
                        acceptance_criteria = "\n".join(response.acceptance_criteria)

                    return product_owner.UserStory(
                            feature_request=feature_request,
                            generated_story=STORY_FORMAT.format(**{
                                "story_text": response.story_text,
                                "acceptance_criteria": acceptance_criteria,
                                "example_use_cases": example_use_cases,
                                "non_functional_requirements": "\n".join(response.non_functional_requirements),
                            }),
                            reviewed_story=story.generated_story,
                            reviewed_summary="",
                            human_approved=0,
                            jira_ticket="",
                            review_cycles=story.review_cycles,
                            feedback=""
                        )


from langchain.prompts import PromptTemplate


class ProductOwnerPrompt:
    
    user_story_prompt = PromptTemplate(
            # template="""
            #     Generate a structured user story for the following feature request: {feature_request}.
            #     Return the response as a valid JSON strictly following this format:

            #     {{
            #         "stories": [
            #             {{
            #                 "feature_request": "<feature_request>",
            #                 "story_text": "<user_story>",
            #                 "summary": "<short_summary>",
            #             }}
            #         ]
            #     }}
            #     """,
            template="""Generate a structured user story for the following feature request: {feature_request}.  
                Return the response as a valid JSON strictly following this format:  

                {{  
                    "stories": [  
                        {{  
                            "feature_request": "<feature_request>",  
                            "story_text": "<user_story>",  
                            "summary": "<short_summary>"  
                        }}  
                    ]  
                }}  

                ### Examples:  

                Feature Request: "Enable dark mode for the application"  
                Response:  
                {{  
                    "stories": [  
                        {{  
                            "feature_request": "Enable dark mode for the application",  
                            "story_text": "As a user, I want to switch to dark mode so that I can reduce eye strain during nighttime use.",  
                            "summary": "Add dark mode toggle to settings."  
                        }}  
                    ]  
                }}  

                Feature Request: "Allow exporting reports in CSV format"  
                Response:  
                {{  
                    "stories": [  
                        {{  
                            "feature_request": "Allow exporting reports in CSV format",  
                            "story_text": "As a business analyst, I want to export reports in CSV format so that I can analyze data in Excel.",  
                            "summary": "Add CSV export functionality to reports."  
                        }}  
                    ]  
                }}  

                ### New Feature Request:  
                {feature_request}
                Now generate the response in the same JSON format.""",
                input_variables=["feature_request"]
            )

    po_grader_prompt = PromptTemplate(
        template="""You are a product owner assessing relevance of user stories created for feature request. \n 
        Here is the created user story: \n\n {story} \n\n
        Here is the feature request: {feature_request} \n
        If the created user story has semantic meaning related to the feature request, grade it as relevant. \n
        Strictly Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["story", "feature_request"],
    )

    detailed_story_prompt = PromptTemplate(
        template="""
        Given the following summary of a feature request, generate a **detailed** and structured user story.

        **Summary:** {summary}

        Your response should strictly follow this JSON structure:
        {{
            "feature_request": "<feature_request>",
            "story_text": "<detailed_user_story>",
            "acceptance_criteria": [
                "<criterion>"
            ],
            "example_use_cases": [
                "<example_use_case>"
            ],
            "non_functional_requirements": [
                "<requirement>"
            ]
        }}
        Ensure the **user story** follows the standard agile format:
        - "As a [user], I want [goal] so that [reason]."
        """,
        input_variables=["summary"]
    )

    user_story_feedback_prompt = PromptTemplate(
        # template = """
        #     Generate a structured user story for the following feature request: {feature_request}.
        #     Based on the {feedback} refine the previous {user_story} based on it.

        #     Return the response as a valid JSON strictly following this format:

        #     {{
        #         "stories": [
        #             {{
        #                 "feature_request": "<feature_request>",
        #                 "story_text": "<user_story>",
        #                 "summary": "<short_summary>",
        #             }}
        #         ]
        #     }}
        #     """,
        template = """
            Generate a structured user story for the following feature request: {feature_request}.  
            Based on the provided feedback: "{feedback}", refine the previous user story: "{user_story}" accordingly.  

            Ensure that:
            - The response is **strictly valid JSON**.
            - Do **not** include extra explanations, line breaks, or additional fields.
            - Follow this exact format:

            ```json  
            {{  
                "stories": [  
                    {{  
                        "feature_request": "<feature_request>",  
                        "story_text": "<refined_user_story>",  
                        "summary": "<short_summary>"  
                    }}
                ]  
            }}

            ### Examples:  

            #### Example 1  
            **Feature Request:** "Enable dark mode for the application"  
            **Previous User Story:** "As a user, I want to switch to dark mode so that I can reduce eye strain during nighttime use."  
            **Feedback:** "Specify that dark mode should follow system settings."  
            **Refined Response:**  
            {{  
                "stories": [  
                    {{  
                        "feature_request": "Enable dark mode for the application",  
                        "story_text": "As a user, I want the application to support dark mode and follow my system settings to automatically switch modes.",  
                        "summary": "Implement dark mode with system setting integration."  
                    }} 
                ]  
            }}

            #### Example 2  
            **Feature Request:** "Allow exporting reports in CSV format"  
            **Previous User Story:** "As a business analyst, I want to export reports in CSV format so that I can analyze data in Excel."  
            **Feedback:** "Ensure exported CSV includes column headers and correct formatting."  
            **Refined Response:**  
            {{  
                "stories": [  
                    {{  
                        "feature_request": "Allow exporting reports in CSV format",  
                        "story_text": "As a business analyst, I want to export reports in CSV format with properly formatted data and column headers so that I can analyze data efficiently in Excel.",  
                        "summary": "Ensure CSV exports include headers and correct formatting."  
                    }}  
                ]  
            }}

            ### New Input:  
            Feature Request: "{feature_request}"  
            Previous User Story: "{user_story}"  
            Feedback: "{feedback}"  
            Now refine the user story and return the JSON response in the same format.

        """,
            input_variables=["feature_request", "feedback", "user_story"]
        )
    
    # Generate AI Response
    chat_prompt = PromptTemplate(
        template = """
            You are an AI assistant answering questions based on this feature request:
            {title}

            User's question: {user_input}
            """,
            input_variables=["title", "user_input"]
    )
from langchain.prompts import PromptTemplate

class UnitTesterPrompt:
    unit_test_feedback_prompt = PromptTemplate(
        template="""
        You are a senior software engineer responsible for ensuring high-quality unit tests. Based on the given user feedback, update the existing unit tests to improve coverage, correctness, and maintainability.

        ### Existing Code:
        {existing_code}

        ### Existing Unit Tests:
        {existing_tests}

        ### User Feedback:
        {feedback}

        ### Requirements:
        1. Incorporate feedback to improve test coverage.
        2. Fix any identified issues or gaps in test assertions.
        3. Improve clarity and maintainability of test cases.
        4. Use `pytest` for structured and readable test execution.
        5. Mock external dependencies where necessary.
        6. Ensure compliance with best practices for unit testing.

        Provide the updated unit test code incorporating these improvements.
        """
    )

    unit_test_prompt = PromptTemplate(
                template="""
                You are a senior software engineer specializing in writing robust and well-structured unit tests. Generate a set of unit tests for the following implementation.

                ### Code to Test:
                {code}

                ### Requirements:
                1. Use `pytest` as the testing framework.
                2. Cover various test cases, including:
                - Valid inputs
                - Edge cases
                - Invalid inputs
                3. Implement parameterized tests where applicable.
                4. Ensure proper assertions and meaningful test descriptions.
                5. Use mocking where necessary to isolate dependencies.
                6. Verify API response formats if applicable.
                7. Ensure high test coverage and adherence to best practices.

                ### Expected Output:
                - A structured `test_sample.py` file with well-organized test cases.
                - Use `pytest.raises` to validate error handling.
                - Ensure compliance with industry standards for testing.

                Provide the complete implementation of the unit tests.

            """
            )
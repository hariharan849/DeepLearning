
from langchain.prompts import PromptTemplate

class DeveloperOwnerPrompt:
    code_developer_prompt = PromptTemplate(
        template="""
        You are a senior software developer. Based on the following software design document, generate a complete implementation for {story_text}.

        Design Document
        {design_doc}

        Requirements:
        Use Pydantic models to validate request and response data.
        Skip writing unit tests.
        Include logging for request handling.
        Follow best practices for scalability and security.
        Generate a production-ready, well-documented, and structured codebase.

        Ensure any code you provide can be executed with all required imports and variables defined.
        """,
        input_variables=["story_text", "design_doc"]
    )

    code_review_prompt = PromptTemplate(
        template="""
            You are a senior software engineer specializing in code quality, security, and maintainability. Perform a detailed code review for the following implementation, ensuring compliance with PEP8, SonarQube best practices, and Flake8 linting rules.

            ### Code to Review:
            {code}

            ### Code Review Requirements:
            1. **PEP8 Compliance**:
            - Ensure proper indentation, spacing, and naming conventions.
            - Check for unnecessary line breaks and long lines exceeding recommended length.
            - Verify docstrings and comments follow PEP257.

            2. **SonarQube Code Quality**:
            - Identify potential code smells, bugs, and security vulnerabilities.
            - Suggest improvements for better maintainability, reliability, and test coverage.

            3. **Flake8 Linting**:
            - Detect unused imports, undefined variables, and stylistic inconsistencies.
            - Highlight redundant code blocks and unnecessary complexity.

            4. **General Best Practices**:
            - Ensure modularity, readability, and proper function/method decomposition.
            - Recommend refactoring opportunities to enhance efficiency.
            - Verify exception handling and logging for robustness.

            ### Expected Output:
            - A summary of detected issues categorized under PEP8, SonarQube, and Flake8.
            - Specific recommendations and fixes for improving code quality.
            - A revised, optimized, and production-ready version of the code.

            """
        )
    
    code_update_prompt = PromptTemplate(
        template="""
            You are a senior software developer. Based on the following feedback, modify the existing implementation to meet the updated requirements for {story_text}.

            Design Document
            {design_doc}

            ### User Feedback:
            {feedback}

            ### Existing Implementation:
            {current_code}

            ### Requirements:
            - Apply the requested changes while maintaining best practices for readability, security, and scalability.
            - Ensure that all modifications align with the given design document.
            - If the feedback suggests improvements, refactor the code accordingly while preserving existing functionality.
            - Keep unit tests and logging updated to reflect the changes.

            Provide a revised, production-ready codebase that incorporates the feedback.
            """,
            input_variables=["story_text", "design_doc", "feedback", "current_code"]
        )
    
    code_developer_security_prompt = PromptTemplate(
        template="""
            You are a **senior software developer**. Based on the following **software design document**, **security review feedback**, **generated code** and secure implementation for **{story_text}**.

            ---
            
            ### **📜 Design Document**
            ```markdown
            {design_doc}
            ```

            ---
            
            ### **📜 Existing code**
            ```markdown
            {current_code}
            ```

            ---
            
            ### **🛡️ Security Review Feedback**
            **The following security vulnerabilities were identified and must be addressed in the implementation along with user feedback for security: {security_feedback}:**  
            ```markdown
            {security_issues}
            ```

            ---
            
            ### **🔹 Requirements**
            - ✅ **Use Pydantic models** to validate request and response data.  
            - ✅ **Include logging** for request handling.  
            - ✅ **Follow best practices** for **scalability and security**.  
            - ✅ **Ensure security issues are mitigated** based on feedback.  
            - 🚫 **Skip writing unit tests.**  
            - 🎯 **Generate a production-ready, well-documented, and structured codebase.**  

            ---
            
            ### **🔍 Expected Output**
            - **Secure & Scalable Implementation** (Incorporating security fixes).  
            - **Code should follow best practices** for maintainability.  
            - **Logging should be included** where necessary.  
            - **All identified security vulnerabilities must be fixed.**  

            ---
            
            🎯 **Objective:** Your implementation must be **production-ready**, addressing **both functional and security concerns**.
        """,
        input_variables=["story_text", "design_doc", "current_code", "security_issues", "security_feedback"]
    )

    code_developer_unit_test_prompt = PromptTemplate(
        template="""
            You are a **senior software developer**. Based on the following **software design document**, **security review feedback**, and **existing test case issues**, generate a complete and secure implementation for **{story_text}**.

            ---
            
            ### **📜 Design Document**
            ```markdown
            {design_doc}
            ```

            ---
            
            ### **📜 Existing code**
            ```markdown
            {current_code}
            ```

            ---

            ---
            
            ### **🛡️ Security Review Feedback**
            **The following security vulnerabilities were identified and must be addressed in the implementation along with user feedback for security: {security_feedback}:**  
            ```markdown
            {security_issues}
            ```

            ---
            
            ### **🧪 Existing Unit Test Issues**
            **The following problems have been found in existing unit tests and must be fixed:**  
            ```markdown
            {unit_test_issues}
            ```

            ---
            
            ### **🔹 Requirements**
            - ✅ **Use Pydantic models** to validate request and response data.  
            - ✅ **Include logging** for request handling.  
            - ✅ **Follow best practices** for **scalability and security**.  
            - ✅ **Fix all identified security vulnerabilities**.  
            - ✅ **Ensure all unit tests are correct, well-structured, and pass successfully**.  
            - ✅ **Use `pytest` to write or correct unit tests** with proper assertions and coverage.  
            - 🎯 **Generate a production-ready, well-documented, and structured codebase.**  

            ---
            
            ### **🔍 Expected Output**
            - **Secure & Scalable Implementation** (Fixing security and test issues).  
            - **Fixed Unit Tests** using `pytest`, ensuring proper validation.  
            - **Logging should be included** where necessary.  
            - **All identified issues in unit tests should be resolved.**  

            ---
            
            🎯 **Objective:** Your implementation must be **production-ready**, addressing **functional correctness, security vulnerabilities, and unit test issues**.
        """,
        input_variables=["story_text", "design_doc", "current_code", "security_issues", "security_feedback", "unit_test_issues"]
    )

    code_developer_qa_test_prompt = PromptTemplate(
        template="""
            You are a **senior software developer**. Based on the following **software design document**, **security review feedback**, **QA test feedback**, and **existing test case issues**, generate a complete and secure implementation for **{story_text}**.

            ---
            
            ### **📜 Design Document**
            ```markdown
            {design_doc}
            ```

            ---
            
            ### **📜 Existing Code**
            ```python
            {current_code}
            ```

            ---
            
            ### **🛡️ Security Review Feedback**
            **The following security vulnerabilities were identified and must be addressed along with user feedback for security:**  
            ```markdown
            {security_issues}
            ```
            
            **Security feedback from testers:**  
            ```markdown
            {security_feedback}
            ```

            ---
            
            ### **🧪 Existing Unit Test Issues**
            **The following problems have been found in existing unit tests and must be fixed:**  
            ```markdown
            {unit_test_issues}
            ```

            ---
            
            ### **🔍 QA Test Feedback**
            **The QA team has identified the following issues that need to be addressed in the implementation and unit tests:**  
            ```markdown
            {qa_test_feedback}
            ```

            ---
            
            ### **🔹 Requirements**
            - ✅ **Use Pydantic models** to validate request and response data.  
            - ✅ **Include logging** for request handling.  
            - ✅ **Follow best practices** for **scalability, security, and maintainability**.  
            - ✅ **Fix all identified security vulnerabilities and unit test issues**.  
            - ✅ **Address QA test feedback to improve reliability**.  
            - ✅ **Ensure all unit tests are correct, well-structured, and pass successfully**.  
            - ✅ **Use `pytest` to write or correct unit tests** with proper assertions and coverage.  
            - 🎯 **Generate a production-ready, well-documented, and structured codebase.**  

            ---
            
            ### **📌 Expected Output**
            - **Secure & Scalable Implementation** (Fixing security and test issues).  
            - **Fixed Unit Tests** using `pytest`, ensuring proper validation.  
            - **Logging should be included** where necessary.  
            - **All identified issues in unit tests should be resolved.**  
            - **QA Test Cases should pass successfully** with full coverage.  

            ---
            
            🎯 **Objective:** Your implementation must be **production-ready**, addressing **functional correctness, security vulnerabilities, QA feedback, and unit test issues**.
        """,
        input_variables=[
            "story_text", "design_doc", "current_code", "security_issues", 
            "security_feedback", "unit_test_issues", "qa_test_feedback"
        ]
    )


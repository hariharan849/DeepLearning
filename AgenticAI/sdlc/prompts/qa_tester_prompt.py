from langchain.prompts import PromptTemplate

class QATesterPrompt:
    qa_test_feedback_prompt = PromptTemplate(
        template="""
        You are a **senior QA engineer** responsible for testing a software implementation.  
        Based on the provided **code implementation**, **security fixes**, and **unit tests**, generate a comprehensive **test suite** to validate correctness, security, and performance.

        ---
        
        ### **📜 Code Implementation**
        ```python
        {code}
        ```

        ---
        
        ### **🛡️ Security Fixes Applied**
        **The following security improvements were made and must be tested:**  
        ```markdown
        {security_fixes}
        ```

        ---
        
        ### **🧪 Unit Tests**
        **Existing unit tests:**  
        ```python
        {unit_tests}
        ```

        ---
        
        ### **🔍 Test Requirements**
        - ✅ **Functional Testing** (Ensure all features work as expected).  
        - ✅ **Security Testing** (Validate security patches and ensure no new vulnerabilities).  
        - ✅ **Performance Testing** (Check response times, load handling, and scalability).  
        - ✅ **Regression Testing** (Ensure no previous functionality is broken).  
        - ✅ **Edge Case & Boundary Testing** (Test invalid inputs, large datasets, etc.).  
        - ✅ **Automated Testing** (Provide `pytest` test cases for automation).  
        - ✅ **API Testing** (Verify API responses, error handling, and security).  

        ---
        
        ### **📌 Expected Test Case Format**
        | **Test Case ID** | **Test Description** | **Input** | **Expected Output** | **Result** |
        |-----------------|--------------------|----------|-------------------|----------|
        | TC-001 | Validate API authentication | Valid credentials | Access granted | ✅ Pass |
        | TC-002 | Test SQL injection prevention | `' OR 1=1 --` | Request rejected | ✅ Pass |
        | TC-003 | Check API rate limiting | 1000 requests/sec | Some requests blocked | ✅ Pass |
        | TC-004 | Large payload upload | 100MB file | File uploaded successfully | ✅ Pass |

        ---
        
        ### **🔄 Automated Tests**
        Provide `pytest` test cases to automate the above test scenarios.  

        ---
        
        🎯 **Objective:** Deliver a **detailed QA test suite** that ensures the software is **fully tested, secure, and production-ready**.
    """,
    input_variables=["code", "security_fixes", "unit_tests"]
    )

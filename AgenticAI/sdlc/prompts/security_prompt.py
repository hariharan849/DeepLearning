from langchain.prompts import PromptTemplate


class SecurityPrompt:

    security_prompt = PromptTemplate(
        template="""
            You are a cybersecurity expert specializing in secure software development. Perform a comprehensive security analysis of the following code to identify vulnerabilities and suggest improvements.

            ### Code to Analyze:
            {code}

            ### Security Check Requirements:
            1. **Input Validation**: Ensure all user inputs are properly validated to prevent injection attacks.
            2. **Authentication & Authorization**: Verify that proper access control mechanisms are in place.
            3. **Data Protection**: Ensure sensitive data is encrypted and securely stored.
            4. **API Security**: Check for rate limiting, authentication mechanisms, and secure endpoint exposure.
            5. **Logging & Monitoring**: Ensure logs capture relevant security events without exposing sensitive data.
            6. **Error Handling**: Identify any unhandled exceptions or information leakage through error messages.
            7. **Performance & Scalability Risks**: Detect any security risks related to high traffic and resource exhaustion.
            8. **Compliance**: Ensure adherence to best security practices (e.g., OWASP Top 10, GDPR compliance if applicable).

            ### Expected Output:
            - A list of detected security vulnerabilities.
            - A summary of potential risks.
            - Code modifications to mitigate risks while maintaining functionality.
            - Best practices to enhance the security posture of the system.
            """,
            input_variables=["code"]
            )
    
    feedback_prompt = PromptTemplate(
        template="""
            You are a **cybersecurity expert** specializing in **secure software development**. Perform a **comprehensive security analysis** of the given code, identifying vulnerabilities and providing fixes.

            ---
            
            ### **🔍 Code to Analyze:**
            ```python
            {code}
            ```

            ---
            
            ### **🛡️ Security Check Requirements:**
            1️⃣ **Input Validation:** Prevent SQL injection, XSS, command injection, and buffer overflows.  
            2️⃣ **Authentication & Authorization:** Verify **RBAC (Role-Based Access Control)** and proper session management.  
            3️⃣ **Data Protection:** Ensure **encryption (AES, TLS)** and **secure password storage (bcrypt, Argon2)**.  
            4️⃣ **API Security:** Check **rate limiting, JWT security, CORS, API key exposure**.  
            5️⃣ **Logging & Monitoring:** Ensure **secure logging** without sensitive data exposure.  
            6️⃣ **Error Handling:** Identify unhandled exceptions and **prevent stack trace leaks**.  
            7️⃣ **Performance & Scalability Risks:** Detect **DoS (Denial of Service), Race Conditions, and Memory Leaks**.  
            8️⃣ **Compliance & Best Practices:** Ensure adherence to **OWASP Top 10, NIST, GDPR, HIPAA** if applicable.  

            ---
            
            ### **🔁 User Feedback for Refinement**
            - **User Feedback:** {feedback}  
            - **Objective:** Adjust the security analysis and recommendations based on the feedback provided.  
            - **Ensure improvements are aligned with the user’s specific security concerns.**

            ---
            
            ### **📌 Expected Output Format:**
            - **🛑 Security Vulnerabilities** (Categorized by severity: **Critical, High, Medium, Low**).
            - **⚠️ Risks Summary** (Impact & Exploitation Scenarios).
            - **✅ Secure Code Modifications** (Fixes with explanations).
            - **📚 Best Practices** (Long-term security improvements).
            - **🔄 Adjustments Based on Feedback** (What changed based on user input?).

            ---
            
            ### **Example Response Structure:**
            **🛑 Found Vulnerability:** Hardcoded API Key Exposure  
            **Severity:** 🔴 Critical  
            **Risk:** Attackers can extract the API key and abuse system access.  
            **✅ Fix:** Store keys in **environment variables** and load via `os.getenv("API_KEY")`.  
            **🔄 User Feedback Applied:** "Check if API key rotation is handled" → _Added key rotation suggestion._  

            ---
            
            🎯 **Objective:** Provide **precise, actionable security feedback** with **secure code improvements** while maintaining functionality.
        """,
        input_variables=["code", "feedback"]
    )

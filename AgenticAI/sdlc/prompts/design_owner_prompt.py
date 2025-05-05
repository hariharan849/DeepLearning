from langchain.prompts import PromptTemplate

class DesignOwnerPrompt:
    design_prompt = PromptTemplate(
        template="""
        You are a software architect. Generate a detailed design document for the following user story. The document should include the following sections:

        Overview – Briefly describe the system and its purpose based on the user story.
        Architecture Overview – Define the high-level architecture (e.g., microservices, monolithic, event-driven) with a diagram if needed.
        System Components – List and explain the major components and their interactions.
        Design Patterns – Specify design patterns used (e.g., MVC, CQRS, Factory, Observer) and their justification.
        Technology Stack – List the programming languages, frameworks, databases, cloud services, and tools.
        Data Flow & Sequence Diagrams – Illustrate the interaction between components using sequence diagrams, activity diagrams, or data flow diagrams.
        API Specifications (if applicable) – Provide details on REST/GraphQL APIs, including endpoints, request/response formats, authentication, and error handling.
        Security Considerations – Address authentication, authorization, encryption, and data protection.
        Scalability & Performance – Explain how the system will scale, caching strategies, load balancing, and performance optimizations.
        Potential Challenges & Mitigations – Identify risks and propose solutions to handle them.
        Extensibility & Maintainability – Ensure the design is modular, loosely coupled, and future-proof.
        Ensure the document includes diagrams where necessary (such as UML, sequence, and architecture diagrams). Use industry best practices and ensure the design is scalable, secure, and maintainable.
        """,
        input_variables=["story_text", "feature_request"]
    )

    feedback_prompt = PromptTemplate(
        template="""
        You are a senior software architect reviewing a system design document. Your task is to provide **detailed feedback** on the following aspects:

        1. **Completeness** – Does the document address all key areas, such as architecture, system components, security, scalability, and API specifications? Are any sections missing or lacking depth?
        
        2. **Clarity & Readability** – Is the document well-structured and easy to understand? Are technical details clearly explained?
        
        3. **Architecture & Design Patterns** – Is the chosen architecture (monolithic, microservices, event-driven, etc.) appropriate for the problem? Are the design patterns used correctly and justified?
        
        4. **Technology Stack Justification** – Are the programming languages, frameworks, and tools chosen optimal for this use case? Are there better alternatives?
        
        5. **Diagrams & Visual Aids** – Are the architecture diagrams, sequence diagrams, and data flow diagrams clear and relevant? If missing, what should be added?
        
        6. **Security Considerations** – Are authentication, authorization, encryption, and data protection strategies well-defined and robust?
        
        7. **Scalability & Performance** – Does the document explain how the system will handle increased load, caching, and optimization strategies? Are there bottlenecks?
        
        8. **Extensibility & Maintainability** – Is the system designed to be modular and easy to extend in the future? Are there any potential technical debts?
        
        9. **Potential Challenges & Mitigations** – Are key risks identified, and are solutions provided to handle them effectively?
        
        Based on the above criteria, provide **constructive feedback** on how to improve the design document.

        **User Story:**
        {{story_text}}

        **Feature Request:**
        {{feature_request}}

        **System Design Document:**
        {{design_document}}

        Provide feedback in the following format:
        - **Strengths:** List the well-executed parts of the design.
        - **Areas for Improvement:** Identify gaps or potential issues.
        - **Suggestions:** Offer specific recommendations for improvement.
        """,
        input_variables=["story_text", "feature_request", "design_document"]
    )

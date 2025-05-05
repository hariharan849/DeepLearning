import os
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def search_recruiters(company):
    search_query = f"Recruiter site:linkedin.com/in AND {company}"
    url = f"https://serpapi.com/search.json?q={search_query}&engine=google&api_key={SERPAPI_KEY}"

    response = requests.get(url)
    results = response.json().get("organic_results", [])
    
    recruiters = []
    for result in results:
        recruiters.append({
            "name": result.get("title", "").split(" | ")[0],
            "title": result.get("title", ""),
            "link": result.get("link"),
            "company": company
        })

    return recruiters

def summarize_recruiters(recruiters):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    print(llm.invoke("""Based on the recruiter profiles below, identify the best recruiter for DataScience/ML/AI/Software roles at ExxonMobil.
        Provide a ranked list with insights on each recruiter.

        [{'name': 'Ana Wilburn - Recruiting Consultant - Exxon Mobil ...', 'title': 'Ana Wilburn - Recruiting Consultant - Exxon Mobil ...', 'link': 'https://www.linkedin.com/in/ana-wilburn-22b0589'}]"""))
        
    # company = recruiters[0]["company"]
    # prompt = PromptTemplate(
    #     input_variables=["recruiters", "company"],
    #     template="""
    #     Based on the recruiter profiles below, identify the best recruiter for DataScience/ML/AI/Software roles at {company}.
    #     Provide a ranked list with insights on each recruiter.

    #     {recruiters}
    #     """
    # )
    # print(prompt.format(recruiters=recruiters, company=company))
    return llm.invoke(prompt.format(recruiters=recruiters, company=company))

company_name = "ExxonMobil"
recruiters = search_recruiters(company_name)
print(recruiters)
summary = summarize_recruiters(recruiters)
print("Recruiter Insights:\n", summary)
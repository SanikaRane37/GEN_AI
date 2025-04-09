import os
import streamlit as st
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
import json

# Set API keys
from constants import openai_key, serpapi_key
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["SERPAPI_API_KEY"] = serpapi_key

# Streamlit UI
st.title('Company Insights with Real Data (AmbitionBox)')
company = st.text_input("Enter company name (e.g., TCS, Infosys):")
role = st.text_input("Enter job role (e.g., Software Engineer):")

# Use SerpAPI to pull info from AmbitionBox
search = SerpAPIWrapper()

def get_salary(company, role):
    query = f"site:ambitionbox.com average salary for {role} at {company}"
    return search.run(query)

def get_ratings(company):
    query = f"site:ambitionbox.com {company} employee reviews rating"
    return search.run(query)

def get_about(company):
    query = f"What does {company} do?"
    return search.run(query)

# Agent tools
tools = [
    Tool(
        name="Get Company Salary",
        func=lambda x: get_salary(company, role),
        description="Use this to fetch average salary for a role at a company from AmbitionBox"
    ),
    Tool(
        name="Get Company Rating",
        func=lambda x: get_ratings(company),
        description="Use this to fetch company rating from AmbitionBox"
    ),
    Tool(
        name="Get Company Description",
        func=lambda x: get_about(company),
        description="Use this to get a short company description"
    )
]

# Create agent
llm = OpenAI(temperature=0.6)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run when inputs provided
if company and role:
    with st.spinner("Fetching real-time insights..."):
        inputs = {"company": company, "role": role}

        # response = agent.run(f"""
        # Use the tools provided to:
        # 1. Get a brief and accurate company description of {company}
        # 2. Retrieve the average salary of a {role} at {company} using salary sources
        # 3. Retrieve employee ratings of {company} from review sites

        # Do not make up data. Use tools only.
        # """)
        # st.markdown(response)


        prompt = f"""
        Use the tools provided to answer the following. Respond only with a valid JSON object.

        Return clean values:
        - A short 1-2 sentence company description
        - The most relevant average salary for a {role} at {company}, in INR per year, as one line
        - The employee rating out of 5

        Example output format:

        {{
        "company": "{company}",
        "description": "Company does X, Y, and Z...",
        "average_salary": "₹6.5 Lakhs/year",
        "employee_rating": "3.9/5"
        }}

        Only return this JSON. Do not include multiple salary results or extra commentary.
        Do not make up any values. Only use the tools.
        """

        response = agent.run(prompt)
        try:
            json_output = json.loads(response)
            st.json(json_output)
        except:
            st.write("⚠️ Failed to parse response as JSON:")
            st.write(response)

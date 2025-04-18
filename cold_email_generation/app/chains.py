import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers  import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = llm = ChatGroq(temperature = 0,groq_api_key=os.getenv("GROQ_API_KEY") ,model_name="llama3-70b-8192")

    def extract_jobs(self , cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the carrer's page of a website.
            Your job is to extract the job postings and return them in JSON format containing
            following kets: `role` , `experience`,`skills` , `description`.
            only return the valid JSON. No explanation.
            ### VALID JSON(NO PREAMBLE)

            """

   )
        chain_extract = prompt_extract| self.llm
        res=chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException():
            raise OutputParserException("Context too big. Unable to parse jobs")
        return res if isinstance(res , list) else [res]
    
    def write_email(self , job , links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Paul, a business development executive at Caramel.io. Caramel.io is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Caramel.io 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Caramel.io portfolio: {link_list}
            Remember you are Paul, BDE at Caramel.io. You can use email of carame.io as`caramel.io@gmail.com` and for conatct you can use the number `9346965021`
            Do not provide any explanation just output the email.
            ### EMAIL (NO PREAMBLE) No explanations:

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content



if __name__ == '__main__':
    print(os.getenv("GROQ_API_KEY"))




import streamlit as st
import os
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, ScrapeWebsiteTool, SerperDevTool, MDXSearchTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set Streamlit Page Config
st.set_page_config(page_title="AI-Powered Job Application Tailor", layout="wide")

# ---- Streamlit UI ----
st.title("📄 AI-Powered Job Application Tailor")
st.write("Automate and optimize your job application process using AI agents.")

# Input fields
job_url = st.text_input("📞 Job Posting URL")
personal_writeup = st.text_area("📝 Personal Summary", "")

uploaded_file = st.file_uploader("📂 Upload Resume (Markdown Format)", type=["md"])

# Process Resume File
resume_text = ""
if uploaded_file is not None:
    resume_text = uploaded_file.read().decode("utf-8")

# ---- CrewAI Agents and Tasks ----
if st.button("🚀 Generate Tailored Application"):
    if not job_url or not resume_text:
        st.warning("Please enter a job posting URL and upload a resume.")
    else:
        # Define Tools
        scrape_tool = ScrapeWebsiteTool()
        search_tool = SerperDevTool()
        read_resume = FileReadTool(file_path="resume.md")
        semantic_search_resume = MDXSearchTool(mdx="resume.md")

        # Save uploaded resume for processing
        with open("resume.md", "w", encoding="utf-8") as f:
            f.write(resume_text)

        researcher = Agent(
            role="Job Researcher",
            goal="Extract job details from job postings, including job description, qualifications, and skills.",
            tools=[scrape_tool],
            verbose=True,
            backstory="An expert job market researcher who finds the best opportunities."
        )

        profiler = Agent(
            role="Personal Profiler",
            goal="Analyze and extract key details from the applicant's resume to build a strong professional profile.",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
            backstory="A specialist in identifying and refining applicants' key strengths."
        )

        resume_strategist = Agent(
            role="Resume Strategist",
            goal=(
                "Optimize the resume by keeping only relevant or slightly relevant skills, experience, and sections. "
                "Completely remove unrelated content while ensuring that key sections remain intact if they provide value. "
                "Ensure the final resume is ATS-friendly and well-structured."
                "The final output should strictly contain resume and no un necessary explanation of what agents have done"
            ),
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
            backstory="An expert in refining resumes to highlight only the most relevant qualifications."
        )

        interview_preparer = Agent(
            role="Interview Preparer",
            goal="Generate interview questions and talking points based on the tailored resume and job requirements.",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
            backstory="A specialist in preparing candidates for interviews by aligning their experience with job expectations."
        )

        research_task = Task(
            description="Scrape job details from {job_posting_url}, extracting job title, company, description, skills, and qualifications.",
            expected_output="A structured list of job details, including requirements and key skills.",
            agent=researcher
        )

        profile_task = Task(
            description="Analyze the uploaded resume and extract key sections dynamically. Identify strengths and areas that align with the job.",
            expected_output="A structured breakdown of resume sections with analysis of relevance to the job posting.",
            agent=profiler,
            async_execution=True
        )

        resume_strategy_task = Task(
            description=(
                "Compare the job description with the uploaded resume and dynamically restructure the resume. "
                "Identify only the most relevant skills, experience, and projects for the job role. "
                "Extract skills mentioned in project descriptions and add them to the skills section if they align with the job listing. "
                "Completely remove any sections that do not align with job requirements. except for hobbies snd intrests. Do not remove Linkedin links or portfolio and github links."
                "The final resume should be well-structured, ATS-friendly, and formatted correctly."
                "The final resume should not contain any un-necessary work experience or projects that dont allign with the job description."
            ),
            expected_output=(
                "A refined resume that includes only job-relevant details and removes all unrelated content. "
                "The final output should be formatted correctly and in markdown format."
                "Skills section should be updated with skills extracted from project descriptions if they match job requirements."
                "Strictly output only the resume no explanation about the refinements done."
            ),
            output_file="tailored_resume.md",
            context=[research_task, profile_task],
            agent=resume_strategist
        )

        interview_preparation_task = Task(
            description=(
                "Generate a set of interview questions and key talking points tailored to the updated resume and job description. "
                "Ensure the candidate is well-prepared to highlight their relevant experience and skills."
            ),
            expected_output="A document containing interview questions and discussion points relevant to the job application.",
            output_file="interview_materials.md",
            context=[research_task, profile_task, resume_strategy_task],
            agent=interview_preparer
        )

        job_application_crew = Crew(
            agents=[researcher, profiler, resume_strategist, interview_preparer],
            tasks=[research_task, profile_task, resume_strategy_task, interview_preparation_task],
            verbose=True
        )

        st.info("🔍 Running AI agents, please wait...")
        result = job_application_crew.kickoff(inputs={
            "job_posting_url": job_url,
            "personal_writeup": personal_writeup
        })

        st.success("✅ Job application tailored successfully!")

        initial_resume_text = resume_text  # The uploaded resume content

        # Read the tailored resume
        with open("tailored_resume.md", "r", encoding="utf-8") as f:
            tailored_resume_text = f.read()
        
        # Create side-by-side columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Original Resume")
            st.markdown(initial_resume_text, unsafe_allow_html=True)  # Display original resume
        
        with col2:
            st.subheader("✨ Tailored Resume")
            st.markdown(tailored_resume_text, unsafe_allow_html=True)  # Display tailored resume

        st.subheader("🎤 Interview Preparation")
        st.text_area("Generated Interview Questions", result, height=300)


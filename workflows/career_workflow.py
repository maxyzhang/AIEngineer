from agents.career_agent import run as run_career 
from agents.resume_agent import run as run_resume 
from agents.interview_agent import run as run_interview 
from vector_search import search_vector


def run(question):
    knowledge = search_vector(question)

    career_result = run_career(question)

    resume_question = f"""
Based on this career target, suggest resume positioning:
{question}
"""

    resume_result = run_resume(resume_question)

    interview_question = f"""
Based on this career target, suggest interview preparation:
{question}
"""

    interview_result = run_interview(interview_question)

    final_answer = f"""
# Career Workflow Result

## 1. Knowledge Retrieved

{knowledge}

## 2. Career Fit Analysis

{career_result}

## 3. Resume Positioning

{resume_result}

## 4. Interview Preparation

{interview_result}
"""

    return final_answer
from groq import Groq
import os
from typing import List, Dict

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_hypothesis(papers: List[Dict]) -> str:
    if not papers:
        return "No papers selected to generate a hypothesis from."

    context = ""
    for i, paper in enumerate(papers, 1):
        context += f"Paper {i}:\nTitle: {paper.get('title', 'N/A')}\nSummary: {paper.get('summary', 'N/A')}\n\n"

    prompt = f"""
    You are a research assistant. Based on the abstracts of the following papers, identify a potential research gap or a novel hypothesis for future work.
    Think step-by-step:
    1. What are the common themes or methods in these papers?
    2. What are the limitations or unanswered questions mentioned or implied?
    3. Based on the gaps, formulate a concise and testable hypothesis.

    Provided Papers:
    {context}
    
    Research Gap and Hypothesis:
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content
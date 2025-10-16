from groq import Groq
import os
from typing import List, Dict

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def summarize_papers(papers: List[Dict]) -> str:
    if not papers:
        return "No papers selected to summarize."

    context = ""
    for i, paper in enumerate(papers, 1):
        context += f"Paper {i}:\nTitle: {paper.get('title', 'N/A')}\nSummary: {paper.get('summary', 'N/A')}\n\n"

    prompt = f"""
    Based on the following research papers, provide a concise, consolidated summary of the key findings and themes.
    
    {context}
    
    Consolidated Summary:
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content
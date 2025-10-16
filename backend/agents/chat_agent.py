from groq import Groq
import os
from typing import List, Dict

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chat_with_papers(papers: List[Dict], query: str) -> str:
    if not papers:
        return "Please select some papers to chat with."
    if not query:
        return "Please enter a question."

    context = ""
    for i, paper in enumerate(papers, 1):
        context += f"Paper {i}:\nTitle: {paper.get('title', 'N/A')}\nSummary: {paper.get('summary', 'N/A')}\n\n"

    prompt = f"""
    You are a helpful research assistant. Answer the user's question based *only* on the provided context from the research papers.
    If the answer is not found in the context, say "The answer to that question is not found in the selected papers."

    Context from Papers:
    {context}

    User's Question:
    {query}
    
    Answer:
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content
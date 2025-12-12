"""
Summarizer Agent - Generates consolidated summaries of research papers.
"""
import logging
from typing import List, Dict
from core.llm_client import generate_completion, build_context_from_papers

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant specializing in academic paper analysis. 
Provide clear, concise summaries that highlight key findings and themes."""

def summarize_papers(papers: List[Dict]) -> str:
    """
    Generate a consolidated summary of the provided papers.
    
    Args:
        papers: List of paper dicts with 'title' and 'summary' keys
        
    Returns:
        A consolidated summary string
    """
    if not papers:
        return "No papers selected to summarize."

    context = build_context_from_papers(papers)
    
    prompt = f"""Based on the following research papers, provide a concise, consolidated summary of the key findings and themes.

{context}

Consolidated Summary:"""

    try:
        return generate_completion(prompt, SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"Error generating summary: {str(e)}"
"""
Keyword Agent - Extracts key themes and technical concepts from papers.
"""
import logging
from typing import List, Dict
from core.llm_client import generate_completion, build_context_from_papers

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research analyst skilled at identifying key themes, technologies, and concepts in academic literature.
Extract and organize information systematically without making up content not present in the source material."""

def extract_key_themes(papers: List[Dict]) -> str:
    """
    Extract key themes and technical keywords from papers.
    
    Args:
        papers: List of paper dicts with 'title' and 'summary' keys
        
    Returns:
        Organized list of key themes and keywords
    """
    if not papers:
        return "No paper context provided to extract themes from."

    context = build_context_from_papers(papers, label="Document Chunk")
    
    prompt = f"""Based *only* on the following document chunks, identify and extract the top 10-15 most important and recurring technical keywords, concepts, and themes.

- Do not summarize the papers.
- Do not make up themes not in the text.
- Group the results logically (e.g., 'Key Technologies', 'Core Concepts', 'Research Areas').
- Return *only* the list of themes.

CONTEXT:
{context}

KEY THEMES AND KEYWORDS:"""

    try:
        return generate_completion(prompt, SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"Theme extraction failed: {e}")
        return f"Error extracting themes: {str(e)}"
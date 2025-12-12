"""
Shared LLM client for all agents.
Provides a single Groq client instance with error handling.
"""
import logging
from typing import List, Dict, Optional
from groq import Groq
from core.config import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

# Initialize client once
_client: Optional[Groq] = None


def get_client() -> Groq:
    """Get or create the Groq client singleton."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_completion(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Generate a completion using the LLM.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt for context
        
    Returns:
        The generated text response
        
    Raises:
        Exception: If the API call fails
    """
    client = get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise


def generate_completion_stream(prompt: str, system_prompt: Optional[str] = None):
    """
    Generate a streaming completion using the LLM.
    
    Yields tokens as they are generated for real-time display.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt for context
        
    Yields:
        String tokens as they are generated
    """
    client = get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        stream = client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"LLM streaming API call failed: {e}")
        yield f"Error: {str(e)}"


def build_context_from_papers(papers: List[Dict], label: str = "Paper") -> str:
    """
    Build a formatted context string from a list of papers.
    
    Args:
        papers: List of paper dicts with 'title' and 'summary' keys
        label: Label to use for each item (e.g., "Paper", "Document Chunk")
        
    Returns:
        Formatted context string
    """
    context = ""
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')
        context += f"{label} {i}:\nTitle: {title}\nContent: {summary}\n\n"
    return context

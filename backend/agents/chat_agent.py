"""
Chat Agent - RAG-style Q&A over research papers with Corrective RAG.

Implements Corrective RAG pattern: checks if retrieved context is 
relevant before generating, and can refine the response if needed.
"""
import logging
from typing import List, Dict, Generator
from core.llm_client import generate_completion, generate_completion_stream, build_context_from_papers

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions based ONLY on the provided context from research papers.
If the answer is not found in the context, clearly state that the information is not available in the selected papers.
Be specific and cite which paper(s) support your answer when possible."""


def check_context_relevance(context: str, query: str) -> bool:
    """
    Check if retrieved context is relevant to the query.
    
    This is the "corrective" step in Corrective RAG - we verify
    that we have good context before generating.
    
    Args:
        context: The retrieved context text
        query: The user's query
        
    Returns:
        True if context is relevant, False otherwise
    """
    prompt = f"""Assess if the following context contains information that could help answer the query.
Answer with ONLY 'YES' or 'NO'.

Query: {query}

Context (first 1500 chars):
{context[:1500]}

Is this context relevant to answering the query? (YES/NO):"""
    
    try:
        response = generate_completion(prompt).strip().upper()
        is_relevant = 'YES' in response
        logger.info(f"Context relevance check: {'relevant' if is_relevant else 'not relevant'}")
        return is_relevant
    except Exception as e:
        logger.warning(f"Relevance check failed: {e}")
        return True  # Assume relevant on failure


def chat_with_papers(papers: List[Dict], query: str, use_correction: bool = True) -> str:
    """
    Answer a question based on the content of provided papers.
    
    Implements Corrective RAG: checks context relevance before generating.
    
    Args:
        papers: List of paper dicts with 'title' and 'summary' keys
        query: The user's question
        use_correction: Whether to use corrective RAG (context check)
        
    Returns:
        Answer based on paper content
    """
    if not papers:
        return "Please select some papers to chat with."
    if not query or not query.strip():
        return "Please enter a question."

    context = build_context_from_papers(papers)
    
    # Corrective RAG: Check if context is relevant
    context_note = ""
    if use_correction and len(papers) > 0:
        if not check_context_relevance(context, query):
            context_note = "\n\n[Note: The retrieved context may not fully address your question. The answer below is based on the best available information from the selected papers.]"
    
    prompt = f"""You have been given context from exactly {len(papers)} research paper(s).
Answer the user's question based ONLY on these {len(papers)} paper(s) provided below.
Do NOT make up or reference any papers that are not in the context.
If the answer is not found in these papers, say "The answer to that question is not found in the selected papers."

IMPORTANT: Only reference papers that appear in the context below. There are exactly {len(papers)} papers.

Context from Papers:
{context}

User's Question:
{query}

Answer (using only the {len(papers)} papers above):"""

    try:
        response = generate_completion(prompt, SYSTEM_PROMPT)
        return response + context_note
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return f"Error generating response: {str(e)}"


def chat_with_papers_stream(papers: List[Dict], query: str) -> Generator[str, None, None]:
    """
    Streaming version of chat_with_papers for real-time responses.
    
    Args:
        papers: List of paper dicts with 'title' and 'summary' keys
        query: The user's question
        
    Yields:
        Response tokens as they are generated
    """
    if not papers:
        yield "Please select some papers to chat with."
        return
    if not query or not query.strip():
        yield "Please enter a question."
        return

    context = build_context_from_papers(papers)
    
    prompt = f"""Answer the user's question based *only* on the provided context from the research papers.
If the answer is not found in the context, say "The answer to that question is not found in the selected papers."
When possible, mention which paper(s) support your answer.

Context from Papers:
{context}

User's Question:
{query}

Answer:"""

    try:
        for token in generate_completion_stream(prompt, SYSTEM_PROMPT):
            yield token
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        yield f"Error generating response: {str(e)}"
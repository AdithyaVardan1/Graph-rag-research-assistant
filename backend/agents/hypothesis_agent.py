"""
Hypothesis Agent - Identifies research gaps and generates novel hypotheses.
"""
import logging
from typing import List, Dict
from core.llm_client import generate_completion, build_context_from_papers

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research scientist skilled at identifying research gaps and formulating testable hypotheses.
Think critically about what's missing in current research and propose novel directions."""

def generate_hypothesis(papers: List[Dict]) -> str:
    """
    Analyze papers to identify research gaps and generate a hypothesis.
    
    Args:
        papers: List of paper dicts with 'title' and 'summary' keys
        
    Returns:
        Research gap analysis and hypothesis
    """
    if not papers:
        return "No papers selected to generate a hypothesis from."

    context = build_context_from_papers(papers)
    
    prompt = f"""Based on the abstracts of the following papers, identify a potential research gap or a novel hypothesis for future work.

Think step-by-step:
1. What are the common themes or methods in these papers?
2. What are the limitations or unanswered questions mentioned or implied?
3. Based on the gaps, formulate a concise and testable hypothesis.

Provided Papers:
{context}

Research Gap and Hypothesis:"""

    try:
        return generate_completion(prompt, SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        return f"Error generating hypothesis: {str(e)}"
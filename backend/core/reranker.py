"""
Reranker module for improving retrieval accuracy.

Uses LLM-based scoring to rerank retrieved chunks, which is more
accurate than embedding similarity alone for the final top-k results.
"""
import logging
from typing import List, Dict
from core.llm_client import generate_completion

logger = logging.getLogger(__name__)


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rerank retrieved chunks using LLM-based relevance scoring.
    
    This uses the LLM to score relevance of each chunk to the query,
    providing more accurate ranking than pure embedding similarity.
    
    Args:
        query: The user's query
        chunks: List of chunk dicts with 'content', 'title', 'score' keys
        top_k: Number of top results to return
        
    Returns:
        Reranked list of chunks (top_k items)
    """
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return chunks
    
    # Limit candidates to avoid too many LLM calls
    candidates = chunks[:min(10, len(chunks))]
    
    reranked = []
    for chunk in candidates:
        content = chunk.get('content', '')[:600]  # Limit content length
        title = chunk.get('title', 'Unknown')
        
        prompt = f"""Rate how relevant this text passage is to answering the query.
Return ONLY a single number from 0 to 10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = highly relevant and directly answers the query

Query: {query}

Paper Title: {title}
Text Passage: {content}

Relevance Score (0-10):"""

        try:
            response = generate_completion(prompt).strip()
            # Extract first number from response
            score_str = ''.join(c for c in response.split()[0] if c.isdigit() or c == '.')
            score = float(score_str) if score_str else chunk.get('score', 0)
            score = min(10, max(0, score))  # Clamp to 0-10
        except Exception as e:
            logger.warning(f"Reranking failed for chunk: {e}")
            score = chunk.get('score', 0) * 10  # Use original score as fallback
        
        reranked.append({
            **chunk,
            'rerank_score': score,
            'original_score': chunk.get('score', 0)
        })
    
    # Sort by rerank score descending
    reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    logger.info(f"Reranked {len(candidates)} chunks, top score: {reranked[0]['rerank_score'] if reranked else 0}")
    
    return reranked[:top_k]


def batch_rerank_chunks(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Batch reranking - more efficient but slightly less accurate.
    
    Uses a single LLM call to rank multiple chunks at once.
    
    Args:
        query: The user's query
        chunks: List of chunk dicts
        top_k: Number of top results to return
        
    Returns:
        Reranked list of chunks
    """
    if not chunks or len(chunks) <= top_k:
        return chunks[:top_k] if chunks else []
    
    candidates = chunks[:min(8, len(chunks))]
    
    # Build numbered list of chunks
    chunk_list = ""
    for i, chunk in enumerate(candidates):
        content = chunk.get('content', '')[:300]
        chunk_list += f"\n[{i}] {content}\n"
    
    prompt = f"""Given the query, rank these text passages by relevance.
Return ONLY the passage numbers in order from most to least relevant.
Format: comma-separated numbers, e.g., "2,0,3,1"

Query: {query}

Passages:{chunk_list}

Ranking (most to least relevant):"""

    try:
        response = generate_completion(prompt).strip()
        # Parse ranking
        numbers = [int(n.strip()) for n in response.split(',') if n.strip().isdigit()]
        valid_numbers = [n for n in numbers if 0 <= n < len(candidates)]
        
        # Reorder based on ranking
        reranked = []
        seen = set()
        for idx in valid_numbers:
            if idx not in seen:
                reranked.append(candidates[idx])
                seen.add(idx)
        
        # Add any missing chunks at the end
        for i, chunk in enumerate(candidates):
            if i not in seen:
                reranked.append(chunk)
        
        return reranked[:top_k]
        
    except Exception as e:
        logger.warning(f"Batch reranking failed: {e}")
        return chunks[:top_k]

"""
FAISS Vector Store with Hybrid Retrieval.

Combines dense vector search (FAISS) with sparse keyword search (BM25)
for improved retrieval accuracy.
"""
import os
import faiss
import json
import logging
import numpy as np
from typing import List, Dict, Optional
from gravixlayer import GravixLayer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

client = GravixLayer()
MODEL_ID = "nomic-ai/nomic-embed-text:v1.5"


def get_embeddings_from_gravix(texts: List[str]) -> np.ndarray:
    """Gets embeddings from the Gravix Layer API and normalizes them."""
    try:
        response = client.embeddings.create(model=MODEL_ID, input=texts)
        embeddings = np.array([item.embedding for item in response.data])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms!=0)
        
    except Exception as e:
        logger.error(f"Gravix Layer API Error: {e}")
        return np.array([]).reshape(0, 768)


def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    # Lowercase and split on whitespace/punctuation
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Filter very short tokens
    return [t for t in tokens if len(t) > 2]


class FaissStore:
    """
    Vector store combining FAISS (dense) and BM25 (sparse) search.
    
    Provides hybrid retrieval that combines semantic similarity
    with keyword matching for better results.
    """
    
    def __init__(self, index_dir: str = "data/faiss"):
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "papers.index")
        self.meta_path = os.path.join(index_dir, "papers.meta.json")
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Dict] = []
        
        # BM25 components
        self.bm25_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        
        self._load()

    def _load(self):
        """Load existing index and documents if available."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            # Rebuild BM25 corpus from loaded documents
            self._rebuild_bm25()
        else:
            dim = 768
            self.index = faiss.IndexFlatIP(dim)

    def _rebuild_bm25(self):
        """Rebuild BM25 index from loaded documents."""
        self.bm25_corpus = []
        for doc in self.documents:
            tokens = tokenize(doc.get("chunk_text", ""))
            self.bm25_corpus.append(tokens)
        
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)

    def _save(self):
        """Save index and documents to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def reset(self):
        """Clear all data and reinitialize."""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        self.__init__(self.index_dir)

    def add_chunks(self, chunks: List[str], paper_metadata: dict):
        """
        Add document chunks with their embeddings.
        
        Args:
            chunks: List of text chunks
            paper_metadata: Dict with 'id' and 'title' keys
        """
        if not chunks:
            return
        
        # Get embeddings from Gravix
        vecs = get_embeddings_from_gravix(chunks)
        if vecs.size == 0:
            logger.warning(f"No embeddings returned for paper: {paper_metadata.get('title', 'Unknown')}")
            return

        # Add to FAISS index
        self.index.add(vecs.astype("float32"))
        
        # Add to documents and BM25 corpus
        for chunk_text in chunks:
            self.documents.append({
                "paper_id": paper_metadata["id"],
                "paper_title": paper_metadata["title"],
                "chunk_text": chunk_text
            })
            # Add to BM25 corpus
            tokens = tokenize(chunk_text)
            self.bm25_corpus.append(tokens)
        
        # Rebuild BM25 index
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        
        self._save()

    def search_chunks(self, query: str, top_k: int = 5, 
                      paper_ids: List[str] = None) -> List[Dict]:
        """
        Vector-only search (for backward compatibility).
        
        Args:
            query: Search query
            top_k: Number of results to return
            paper_ids: Optional list of paper IDs to filter by
            
        Returns:
            List of result dicts with 'score', 'title', 'content' keys
        """
        if self.index.ntotal == 0:
            return []
            
        # Nomic models perform best with query prefix
        task_prefix = "search_query: "
        qvec = get_embeddings_from_gravix([task_prefix + query])
        if qvec.size == 0:
            return []
        
        if paper_ids:
            # Search more and filter
            search_k = min(self.index.ntotal, 100)
            scores, idxs = self.index.search(qvec.astype("float32"), search_k)
            
            filtered_results = []
            for idx, score in zip(idxs[0], scores[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                doc = self.documents[idx]
                if doc["paper_id"] in paper_ids:
                    filtered_results.append({
                        "score": float(score),
                        "title": doc["paper_title"],
                        "content": doc["chunk_text"],
                        "paper_id": doc["paper_id"]
                    })
                if len(filtered_results) >= top_k:
                    break
            
            return filtered_results
        else:
            scores, idxs = self.index.search(
                qvec.astype("float32"), 
                min(top_k, self.index.ntotal)
            )
            results = []
            for idx, score in zip(idxs[0], scores[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                doc = self.documents[idx]
                results.append({
                    "score": float(score),
                    "title": doc["paper_title"],
                    "content": doc["chunk_text"],
                    "paper_id": doc["paper_id"]
                })
            return results

    def hybrid_search(self, query: str, top_k: int = 5,
                      paper_ids: List[str] = None,
                      alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search combining BM25 (keyword) and vector (semantic) scores.
        
        Args:
            query: Search query
            top_k: Number of results to return
            paper_ids: Optional list of paper IDs to filter by
            alpha: Weight for vector scores (1-alpha for BM25)
                   0.5 = equal weight, 0.7 = prefer semantic
            
        Returns:
            List of result dicts sorted by combined score
        """
        if self.index.ntotal == 0:
            return []
        
        # Get vector search results
        vector_results = self.search_chunks(query, top_k=top_k * 3, paper_ids=paper_ids)
        
        # Get BM25 scores for all documents
        bm25_scores = []
        if self.bm25 and self.bm25_corpus:
            query_tokens = tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1 range
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores = [s / max_bm25 for s in bm25_scores]
        
        # Create a map of document index to vector score
        doc_to_vector_score = {}
        for i, doc in enumerate(self.documents):
            doc_key = (doc["paper_id"], doc["chunk_text"][:100])
            doc_to_vector_score[doc_key] = 0.0
        
        for result in vector_results:
            doc_key = (result.get("paper_id", ""), result["content"][:100])
            doc_to_vector_score[doc_key] = result["score"]
        
        # Combine scores
        combined_results = []
        for i, doc in enumerate(self.documents):
            # Filter by paper_ids if specified
            if paper_ids and doc["paper_id"] not in paper_ids:
                continue
            
            doc_key = (doc["paper_id"], doc["chunk_text"][:100])
            vector_score = doc_to_vector_score.get(doc_key, 0.0)
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
            
            # Skip if both scores are very low
            if vector_score < 0.1 and bm25_score < 0.1:
                continue
            
            # Combine with alpha weighting
            combined_score = alpha * vector_score + (1 - alpha) * bm25_score
            
            combined_results.append({
                "score": combined_score,
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "title": doc["paper_title"],
                "content": doc["chunk_text"],
                "paper_id": doc["paper_id"]
            })
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Hybrid search: {len(combined_results)} candidates, returning top {top_k}")
        
        return combined_results[:top_k]

    def get_chunks_by_paper_ids(self, paper_ids: List[str]) -> List[Dict]:
        """Get all chunks for specified paper IDs."""
        results = []
        for doc in self.documents:
            if doc["paper_id"] in paper_ids:
                results.append({
                    "title": doc["paper_title"],
                    "content": doc["chunk_text"],
                    "paper_id": doc["paper_id"]
                })
        return results
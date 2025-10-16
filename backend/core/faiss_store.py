import os
import faiss
import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Load the model locally
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class FaissStore:
    # --- __init__ and helper methods (restored) ---
    def __init__(self, index_dir: str = "data/faiss"):
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "papers.index")
        self.meta_path = os.path.join(index_dir, "papers.meta.json")
        self.index = None
        self.documents: List[Dict] = []
        self._load()

    def _load(self):
        """Load FAISS index and metadata if they exist, else initialize new index."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        else:
            dim = embedder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)

    def _save(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def reset(self):
        """Clears the index and metadata."""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        self.__init__(os.path.dirname(self.index_path))

    # --- New chunk-based methods ---
    def add_chunks(self, chunks: List[str], paper_metadata: dict):
        """Adds text chunks for a specific paper to the index."""
        if not chunks:
            return
        
        vecs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(vecs.astype("float32"))
        
        for chunk_text in chunks:
            self.documents.append({
                "paper_id": paper_metadata["id"],
                "paper_title": paper_metadata["title"],
                "chunk_text": chunk_text
            })
            
        self._save()

    def search_chunks(self, query: str, top_k: int = 5, paper_ids: List[str] = None) -> List[Dict]:
        """
        Searches for the most relevant chunks. If paper_ids are provided,
        it constrains the search to only those papers.
        """
        if self.index.ntotal == 0:
            return []

        qvec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # If filtering by paper_ids, we need a constrained search strategy
        if paper_ids:
            # Widen the initial search to get more candidates
            search_k = min(self.index.ntotal, 100)
            scores, idxs = self.index.search(qvec.astype("float32"), search_k)
            
            filtered_results = []
            for idx, score in zip(idxs[0], scores[0]):
                doc = self.documents[idx]
                # Only add the chunk if it belongs to a selected paper
                if doc["paper_id"] in paper_ids:
                    filtered_results.append({
                        "score": float(score),
                        "title": doc["paper_title"],
                        "content": doc["chunk_text"]
                    })
                # Stop once we have enough high-quality, filtered results
                if len(filtered_results) >= top_k:
                    break
            
            # Deduplicate and sort the final list
            unique_results = [dict(t) for t in {tuple(d.items()) for d in filtered_results}]
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)

        # Original behavior: search the entire database if no filter is applied
        else:
            scores, idxs = self.index.search(qvec.astype("float32"), min(top_k, self.index.ntotal))
            results = []
            for idx, score in zip(idxs[0], scores[0]):
                doc = self.documents[idx]
                results.append({
                    "score": float(score),
                    "title": doc["paper_title"],
                    "content": doc["chunk_text"]
                })
            unique_results = [dict(t) for t in {tuple(d.items()) for d in results}]
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
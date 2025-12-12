import numpy as np
from typing import Dict
from .faiss_store import FaissStore

class GraphBuilder:
    def __init__(self, faiss_store: FaissStore):
        self.faiss_store = faiss_store

    def build_graph(self) -> Dict:
        """Builds a similarity graph from all documents in the FAISS store."""
        all_data = self.faiss_store.get_all_data()
        if not all_data:
            return {"nodes": [], "edges": []}

        vectors = np.array([item["embedding"] for item in all_data if "embedding" in item and item["embedding"]])
        if vectors.size == 0:
            return {"nodes": [], "edges": []}
            
        similarity_matrix = np.dot(vectors, vectors.T)

        nodes = []
        for item in all_data:
            node_data = item.copy()
            node_data.pop("embedding", None) 
            nodes.append(node_data)

        edges = []
        for i in range(len(all_data)):
            for j in range(i + 1, len(all_data)):
                score = float(similarity_matrix[i, j])
                edges.append({
                    "source": all_data[i]["id"],
                    "target": all_data[j]["id"],
                    "weight": score
                })
        
        return {"nodes": nodes, "edges": edges}
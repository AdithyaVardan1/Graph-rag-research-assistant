import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Our modules
from agents.fetcher_agent import FetcherAgent
from core.faiss_store import FaissStore
from core.graph_builder import GraphBuilder
from agents.summarizer_agent import summarize_papers
from agents.hypothesis_agent import generate_hypothesis
from agents.chat_agent import chat_with_papers
from core.pdf_processor import process_pdf_from_url
from core.faiss_store import embedder
# Import other agents as you build them

# --- Pydantic Models for API validation ---
class BuildRequest(BaseModel):
    topic: str
    max_results: int = 15

class AgentRequest(BaseModel):
    action: str
    selected_papers: List[Dict[str, Any]]
    query: str = None

# --- FastAPI App ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity in hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Core Components ---
# We use a function to get a fresh instance if needed
def get_faiss_store():
    return FaissStore(index_dir="data/faiss")

# --- API Endpoints ---
# backend/main.py

@app.post("/api/build_graph")
async def build_graph(request: BuildRequest):
    try:
        store = get_faiss_store()
        store.reset()

        fetcher = FetcherAgent()
        papers = fetcher.fetch_papers(request.topic, max_results=request.max_results)
        if not papers:
            return {"nodes": [], "edges": []}

        print("Processing PDFs and embedding chunks...")
        all_chunks = []
        for i, paper in enumerate(papers):
            print(f"  Processing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")
            chunks = process_pdf_from_url(paper['link'])
            if chunks:
                store.add_chunks(chunks, paper)
        print("Chunk processing complete.")

        # --- NEW GRAPH BUILDING LOGIC ---
        # 1. Group all stored document vectors by paper ID
        paper_vectors = {}
        for i, doc in enumerate(store.documents):
            paper_id = doc["paper_id"]
            if paper_id not in paper_vectors:
                paper_vectors[paper_id] = []
            # Reconstruct the vector from the FAISS index
            paper_vectors[paper_id].append(store.index.reconstruct(i))

        # 2. Create a single average vector for each paper
        avg_vectors = {}
        paper_id_list = [p['id'] for p in papers]
        for paper_id in paper_id_list:
            if paper_id in paper_vectors and paper_vectors[paper_id]:
                # Stack vectors and calculate the mean along axis 0
                avg_vectors[paper_id] = np.mean(np.vstack(paper_vectors[paper_id]), axis=0)

        # 3. Calculate the similarity matrix based on these deep average vectors
        edges = []
        threshold = 0.5  # Set a threshold for meaningful connections

        for i in range(len(paper_id_list)):
            for j in range(i + 1, len(paper_id_list)):
                id_i = paper_id_list[i]
                id_j = paper_id_list[j]

                if id_i in avg_vectors and id_j in avg_vectors:
                    # Calculate cosine similarity (dot product of normalized vectors)
                    vec_i = avg_vectors[id_i] / np.linalg.norm(avg_vectors[id_i])
                    vec_j = avg_vectors[id_j] / np.linalg.norm(avg_vectors[id_j])
                    score = np.dot(vec_i, vec_j)

                    # 4. Only add an edge if the similarity is significant
                    if score > threshold:
                        edges.append({
                            "source": id_i,
                            "target": id_j,
                            "weight": float(score)
                        })

        return {"nodes": papers, "edges": edges}
        # --- END OF NEW LOGIC ---

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/agent_action")
async def agent_action(request: AgentRequest):
    try:
        store = get_faiss_store()
        selected_paper_ids = [p['id'] for p in request.selected_papers]
        
        # --- START OF CHANGES ---

        if request.action == "chat":
            # First, try to find the best, most relevant chunks for the query
            relevant_chunks = store.search_chunks(request.query, top_k=5, paper_ids=selected_paper_ids)
            
            context_papers = []
            if relevant_chunks:
                # If we find relevant chunks, use them for high-quality context
                print("✅ Found relevant chunks for chat.")
                context_papers = [{"summary": chunk['content'], "title": chunk['title']} for chunk in relevant_chunks]
            else:
                # FALLBACK: If no chunks match, use the main summaries of the selected papers
                print("⚠️ No relevant chunks found. Falling back to paper summaries for chat.")
                context_papers = request.selected_papers
            
            response = chat_with_papers(context_papers, request.query)
            return {"response": response}

        elif request.action == "summarize" or request.action == "hypothesize":
            # For general actions, get a broad set of relevant chunks from the selected papers
            titles_query = " ".join([p['title'] for p in request.selected_papers])
            relevant_chunks = store.search_chunks(titles_query, top_k=10, paper_ids=selected_paper_ids)
            
            context_papers = []
            if relevant_chunks:
                context_papers = [{"summary": chunk['content'], "title": chunk['title']} for chunk in relevant_chunks]
            else:
                # Fallback for summarize/hypothesize as well
                context_papers = request.selected_papers

            if request.action == "summarize":
                response = summarize_papers(context_papers)
            else: # Must be "hypothesize"
                response = generate_hypothesis(context_papers)
            
            return {"response": response}

        else:
            raise HTTPException(status_code=400, detail="Action not supported")

        # --- END OF CHANGES ---
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/health")
def health_check():
    return {"status": "ok"}
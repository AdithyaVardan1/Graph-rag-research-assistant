"""
LitGraph - Main API Server

Enhanced with:
- GraphRAG: Entity/relationship extraction and knowledge graph
- Hybrid Retrieval: BM25 + vector search
- Reranking: LLM-based result reranking
- Corrective RAG: Context relevance checking
- Streaming: Real-time response generation
"""
import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import numpy as np

from core.config import (
    SIMILARITY_THRESHOLD, 
    DEFAULT_MAX_RESULTS, 
    MAX_RESULTS_LIMIT,
    FAISS_DIR
)
from agents.fetcher_agent import FetcherAgent
from core.faiss_store import FaissStore
from core.knowledge_graph import KnowledgeGraph
from core.reranker import rerank_chunks
from agents.summarizer_agent import summarize_papers
from agents.hypothesis_agent import generate_hypothesis
from agents.chat_agent import chat_with_papers, chat_with_papers_stream
from core.pdf_processor import process_pdf_from_url
from agents.keyword_agent import extract_key_themes

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API validation ---
class BuildRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=MAX_RESULTS_LIMIT)
    
    @field_validator('topic')
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Topic cannot be empty or whitespace only')
        return v.strip()

class AgentRequest(BaseModel):
    action: str = Field(..., pattern='^(chat|summarize|hypothesize|extract_themes)$')
    selected_papers: List[Dict[str, Any]] = Field(..., min_length=1)
    query: Optional[str] = None
    session_id: Optional[str] = None
    use_hybrid: bool = True  # Use hybrid search
    use_reranking: bool = True  # Use LLM reranking
    stream: bool = False  # Use streaming response
    
    @field_validator('query')
    @classmethod
    def validate_query_for_chat(cls, v: str, info) -> str:
        return v

# --- Session Storage ---
_session_stores: Dict[str, FaissStore] = {}
_session_kg: Dict[str, KnowledgeGraph] = {}  # Knowledge graphs per session

def get_faiss_store(session_id: Optional[str] = None) -> FaissStore:
    """Get or create a FAISS store for the given session."""
    if session_id and session_id in _session_stores:
        return _session_stores[session_id]
    
    if session_id:
        session_dir = os.path.join(FAISS_DIR, session_id)
    else:
        session_dir = FAISS_DIR
    
    store = FaissStore(index_dir=session_dir)
    
    if session_id:
        _session_stores[session_id] = store
    
    return store

def get_knowledge_graph(session_id: Optional[str] = None) -> KnowledgeGraph:
    """Get or create a Knowledge Graph for the given session."""
    if session_id and session_id in _session_kg:
        return _session_kg[session_id]
    
    kg = KnowledgeGraph()
    if session_id:
        _session_kg[session_id] = kg
    
    return kg

# --- FastAPI App ---
app = FastAPI(
    title="LitGraph",
    description="AI-powered literature graph with GraphRAG for research paper analysis",
    version="2.0.0"
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.post("/api/build_graph")
async def build_graph(request: BuildRequest):
    """
    Build a knowledge graph for the given research topic.
    
    Enhanced with:
    - Paper similarity graph (original)
    - Entity knowledge graph (GraphRAG)
    """
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Building graph for topic: {request.topic}")
    
    try:
        store = get_faiss_store(session_id)
        store.reset()
        
        # Initialize knowledge graph for GraphRAG
        knowledge_graph = KnowledgeGraph()

        fetcher = FetcherAgent()
        logger.info(f"[{session_id}] Fetching papers from arXiv...")
        papers = fetcher.fetch_papers(request.topic, max_results=request.max_results)
        
        if not papers:
            logger.warning(f"[{session_id}] No papers found for topic: {request.topic}")
            return {"nodes": [], "edges": [], "knowledge_graph": {"nodes": [], "edges": []}, "session_id": session_id}

        logger.info(f"[{session_id}] Processing {len(papers)} papers...")
        for i, paper in enumerate(papers):
            logger.info(f"[{session_id}] Processing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # Process PDF chunks for vector store
            chunks = process_pdf_from_url(paper['link'])
            if chunks:
                store.add_chunks(chunks, paper)
            
            # Extract entities and relationships for GraphRAG
            knowledge_graph.add_paper(paper)
        
        # Store knowledge graph in session
        _session_kg[session_id] = knowledge_graph
        
        logger.info(f"[{session_id}] Building similarity graph...")
        
        # Build paper similarity graph (existing logic)
        paper_vectors = {}
        for i, doc in enumerate(store.documents):
            paper_id = doc["paper_id"]
            if paper_id not in paper_vectors:
                paper_vectors[paper_id] = []
            paper_vectors[paper_id].append(store.index.reconstruct(i))

        avg_vectors = {}
        paper_id_list = [p['id'] for p in papers]
        for paper_id in paper_id_list:
            if paper_id in paper_vectors and paper_vectors[paper_id]:
                avg_vectors[paper_id] = np.mean(np.vstack(paper_vectors[paper_id]), axis=0)

        # k-NN approach for paper graph
        K_NEIGHBORS = 3
        
        similarity_matrix = {}
        for i, id_i in enumerate(paper_id_list):
            if id_i not in avg_vectors:
                continue
            vec_i = avg_vectors[id_i] / np.linalg.norm(avg_vectors[id_i])
            similarities = []
            for j, id_j in enumerate(paper_id_list):
                if i != j and id_j in avg_vectors:
                    vec_j = avg_vectors[id_j] / np.linalg.norm(avg_vectors[id_j])
                    score = float(np.dot(vec_i, vec_j))
                    similarities.append((id_j, score))
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarity_matrix[id_i] = similarities[:K_NEIGHBORS]
        
        # Build edges from k-NN
        edge_set = set()
        edges = []
        for source_id, neighbors in similarity_matrix.items():
            for target_id, weight in neighbors:
                edge_key = tuple(sorted([source_id, target_id]))
                if edge_key not in edge_set and weight > 0.3:
                    edge_set.add(edge_key)
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "weight": weight
                    })

        # Get knowledge graph for visualization
        kg_data = knowledge_graph.to_visualization()
        kg_stats = knowledge_graph.get_stats()
        
        logger.info(f"[{session_id}] Graph complete: {len(papers)} papers, {len(edges)} edges, {kg_stats['entity_count']} entities")
        
        return {
            "nodes": papers,
            "edges": edges,
            "knowledge_graph": kg_data,
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"[{session_id}] Build graph failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to build graph: {str(e)}"
        )

@app.post("/api/agent_action")
async def agent_action(request: AgentRequest):
    """
    Execute an agent action on selected papers.
    
    Enhanced with:
    - Hybrid search (BM25 + vector)
    - LLM reranking
    - Knowledge graph context
    """
    logger.info(f"Agent action: {request.action} on {len(request.selected_papers)} papers")
    
    if request.action == "chat" and (not request.query or not request.query.strip()):
        raise HTTPException(status_code=400, detail="Query is required for chat action")
    
    try:
        store = get_faiss_store(request.session_id)
        kg = get_knowledge_graph(request.session_id)
        selected_paper_ids = [p['id'] for p in request.selected_papers]

        if request.action == "chat":
            # Use hybrid search for better retrieval
            if request.use_hybrid:
                relevant_chunks = store.hybrid_search(
                    request.query, 
                    top_k=10, 
                    paper_ids=selected_paper_ids
                )
            else:
                relevant_chunks = store.search_chunks(
                    request.query, 
                    top_k=10, 
                    paper_ids=selected_paper_ids
                )
            
            # Apply reranking for more accurate top results
            if request.use_reranking and relevant_chunks:
                relevant_chunks = rerank_chunks(request.query, relevant_chunks, top_k=5)
            
            # Enhance with knowledge graph context
            related_entities = kg.query_related_entities(request.query, depth=2)
            kg_context = ""
            if related_entities:
                kg_context = f"\n\nRelated concepts from knowledge graph: {', '.join(related_entities[:10])}"
            
            if relevant_chunks:
                logger.info(f"Found {len(relevant_chunks)} relevant chunks for chat")
                context_papers = [
                    {"summary": chunk['content'] + kg_context, "title": chunk['title']} 
                    for chunk in relevant_chunks
                ]
            else:
                logger.info("No chunks found, using paper summaries")
                context_papers = request.selected_papers
            
            # Use streaming if requested
            if request.stream:
                async def generate():
                    for token in chat_with_papers_stream(context_papers, request.query):
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                response = chat_with_papers(context_papers, request.query)
                return {"response": response}

        elif request.action in ["summarize", "hypothesize", "extract_themes"]:
            titles_query = " ".join([p['title'] for p in request.selected_papers])
            
            # Use hybrid search
            if request.use_hybrid:
                relevant_chunks = store.hybrid_search(
                    titles_query, 
                    top_k=15, 
                    paper_ids=selected_paper_ids
                )
            else:
                relevant_chunks = store.search_chunks(
                    titles_query, 
                    top_k=15, 
                    paper_ids=selected_paper_ids
                )

            if relevant_chunks:
                context_papers = [
                    {"summary": chunk['content'], "title": chunk['title']} 
                    for chunk in relevant_chunks
                ]
            else:
                context_papers = request.selected_papers

            if request.action == "summarize":
                response = summarize_papers(context_papers)
            elif request.action == "hypothesize":
                response = generate_hypothesis(context_papers)
            else:
                response = extract_key_themes(context_papers)

            return {"response": response}

        else:
            raise HTTPException(status_code=400, detail="Action not supported")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent action failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent action failed: {str(e)}")

@app.get("/api/knowledge_graph/{session_id}")
async def get_session_knowledge_graph(session_id: str):
    """Get the knowledge graph for a session."""
    kg = _session_kg.get(session_id)
    if not kg:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "graph": kg.to_visualization(),
        "stats": kg.get_stats()
    }

@app.post("/api/query_entities")
async def query_entities(session_id: str, query: str):
    """Query related entities from the knowledge graph."""
    kg = _session_kg.get(session_id)
    if not kg:
        raise HTTPException(status_code=404, detail="Session not found")
    
    related = kg.query_related_entities(query, depth=2)
    papers = kg.get_papers_for_entities(related)
    
    return {
        "related_entities": related,
        "related_paper_ids": papers
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "version": "2.0.0",
        "features": ["graphrag", "hybrid_search", "reranking", "corrective_rag", "streaming"]
    }

# --- Serve Frontend Static Files ---
@app.get("/")
async def serve_index():
    """Serve the main frontend page."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/{filename}")
async def serve_static(filename: str):
    """Serve static files (CSS, JS, etc.)."""
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
# LitGraph

**AI-powered research assistant with GraphRAG** - explore and analyze academic papers through knowledge graphs with advanced retrieval techniques.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-purple)

---

## Features

### Core Capabilities
- 🔍 **Paper Discovery**: Search arXiv for research papers on any topic
- 🕸️ **Knowledge Graph**: Visualize paper relationships based on semantic similarity
- 🧠 **GraphRAG**: Extract entities & relationships, build true knowledge graphs
- 📝 **Smart Summarization**: Get consolidated summaries of selected papers
- 💡 **Hypothesis Generation**: Identify research gaps and novel hypotheses
- 🏷️ **Theme Extraction**: Extract key themes and technical concepts
- 💬 **Paper Chat**: Ask questions about your selected papers using RAG

### Updates (v2.0)

| Feature | Description |
|---------|-------------|
| **GraphRAG** | Entity/relationship extraction using LLM, knowledge graph construction |
| **Hybrid Retrieval** | Combines BM25 keyword search + vector semantic search |
| **Reranking** | LLM-based reranking for more accurate top results |
| **Corrective RAG** | Validates context relevance before generating responses |
| **Streaming** | Real-time response generation (optional) |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/) (free tier available)
- [Gravix Layer API Key](https://gravixlayer.com/) (for embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/AdithyaVardan1/LitGraph.git
cd LitGraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run Locally

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

---

## How It Works

### GraphRAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        Paper Discovery                          │
│  arXiv API → Fetch Papers → Download PDFs → Extract Text        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Dual Indexing                              │
│  ┌───────────────────┐    ┌────────────────────────────────┐   │
│  │   Vector Store    │    │      Knowledge Graph           │   │
│  │   (FAISS + BM25)  │    │   (Entities + Relationships)   │   │
│  └───────────────────┘    └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Retrieval                           │
│  Hybrid Search → Reranking → Corrective Check → Generation     │
└─────────────────────────────────────────────────────────────────┘
```

### Entity Types Extracted
- **CONCEPT**: Core ideas and theories
- **METHOD**: Techniques and approaches
- **ALGORITHM**: Specific algorithms
- **DATASET**: Datasets used or created
- **METRIC**: Evaluation metrics

### Relationship Types
- `USES`: Method uses another concept
- `IMPROVES`: Method improves upon another
- `COMPARES_TO`: Paper compares two methods
- `BASED_ON`: Work builds on previous research
- `APPLIES_TO`: Method applies to a domain

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/build_graph` | POST | Build knowledge graph for a topic |
| `/api/agent_action` | POST | Execute agent action (summarize, chat, etc.) |
| `/api/knowledge_graph/{session_id}` | GET | Get knowledge graph for a session |
| `/api/query_entities` | POST | Query related entities from knowledge graph |
| `/api/health` | GET | Health check with feature list |

---

## Project Structure

```
LitGraph/
├── backend/
│   ├── agents/           # AI agent modules
│   │   ├── chat_agent.py        # RAG Q&A with Corrective RAG
│   │   ├── fetcher_agent.py     # arXiv paper fetching
│   │   ├── hypothesis_agent.py  # Research gap identification
│   │   ├── keyword_agent.py     # Theme extraction
│   │   └── summarizer_agent.py  # Paper summarization
│   ├── core/             # Core functionality
│   │   ├── config.py            # Configuration
│   │   ├── faiss_store.py       # Hybrid vector store (FAISS + BM25)
│   │   ├── knowledge_graph.py   # GraphRAG implementation
│   │   ├── llm_client.py        # LLM client with streaming
│   │   ├── pdf_processor.py     # PDF text extraction
│   │   └── reranker.py          # LLM-based reranking
│   └── main.py           # FastAPI application
├── frontend/
│   ├── index.html        # Main page with graph views
│   ├── script.js         # Frontend logic
│   └── style.css         # Styling
├── requirements.txt
├── vercel.json           # Vercel deployment config
├── TECHNICAL_DEEP_DIVE.md # Detailed technical documentation
└── README.md
```

---

## Usage

1. **Enter a research topic** in the search box (e.g., "transformer attention")
2. **Adjust papers count** using the slider (5-50)
3. **Click "Build Knowledge Graph"** to fetch and process papers
4. **Toggle between views**:
   - 📄 **Papers**: View paper similarity graph
   - 🔗 **Entities**: View extracted entity knowledge graph
5. **Select papers** by clicking nodes
6. **Use agent actions** to analyze:
   - 📝 **Summarize**: Consolidated summary
   - 💡 **Hypothesis**: Research gap analysis
   - 🏷️ **Themes**: Key technical concepts
   - 💬 **Chat**: Ask questions about papers

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.10+ |
| Frontend | Vanilla JS, Cytoscape.js |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | Gravix Layer (Nomic v1.5) |
| Vector Store | FAISS + BM25 |
| Graph | Custom knowledge graph implementation |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | - | Groq API key for LLM |
| `GRAVIX_API_KEY` | Yes | - | Gravix Layer API key for embeddings |
| `LLM_MODEL` | No | `llama-3.3-70b-versatile` | LLM model to use |
| `SIMILARITY_THRESHOLD` | No | `0.5` | Threshold for graph edges |
| `DEFAULT_MAX_RESULTS` | No | `15` | Default papers to fetch |

---

## Deployment

### Vercel (Recommended)

1. Push to GitHub
2. Connect repo to Vercel
3. Set environment variables in Vercel dashboard:
   - `GROQ_API_KEY`
   - `GRAVIX_API_KEY`
4. Deploy!

The `vercel.json` is already configured.

## Documentation

See [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md) for:
- Detailed explanation of every component
- Mathematical foundations (cosine similarity, BM25, k-NN)
- Complete dry run from search to output

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built by [Adithya Vardan](https://github.com/AdithyaVardan1)

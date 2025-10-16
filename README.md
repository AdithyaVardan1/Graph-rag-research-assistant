# Graph RAG Research Assistant

This project is a multi-agent system built on a Graph RAG architecture to accelerate literature reviews and scientific discovery. Instead of a simple list of documents, it generates an interactive knowledge graph that reveals the deep semantic relationships between full-text research papers.

![Graph RAG UI Screenshot](frontend/image_a08068.png)

---

## Core Concepts

The system is designed to overcome the limitations of traditional keyword-based search, which often fails to surface the relational structure within a body of research.

* **Graph RAG Architecture:** Standard RAG retrieves a list of relevant documents. This system goes a step further by constructing a graph where nodes are papers and edges represent their semantic similarity. This allows for intuitive discovery of research clusters and influential "hub" papers.
* **Full-Text Ingestion Pipeline:** The system doesn't rely on shallow abstracts. It ingests the full text of PDFs using `PyMuPDF`, chunks the content with `LangChain`, and embeds each chunk using a Sentence Transformer model.
* **Deep Similarity Graph:** The graph's topology is computed by averaging the full-text chunk vectors for each paper to create a robust semantic representation. Edges are only rendered above a cosine similarity threshold of `0.5` to ensure topologically significant connections.
* **Constrained Context Generation:** The AI agents (Summarizer, Hypothesis Generator, Chat) perform a constrained search on a FAISS vector store, retrieving the most relevant full-text chunks **only** from the user-selected nodes. This provides the LLM with highly relevant, low-noise context for generation.

---

## Technical Architecture

* **Backend:** High-performance **FastAPI** server.
* **Frontend:** A vanilla **HTML, CSS, and JavaScript** single-page application.
* **Visualization:** **Cytoscape.js** for rendering interactive and scalable network graphs.
* **Vector Database:** **FAISS** (Facebook AI Similarity Search) for efficient maximum inner-product search on text embeddings.
* **LLM Inference:** **Groq API** for high-speed generation with Llama 3.
* **Data Processing:** **PyMuPDF** for PDF text extraction and **LangChain** for text chunking.

---

## Setup and Installation

Follow these steps to run the project locally.

### 1. Backend Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Create your environment file
cp .env.example .env 
```
Now, open the `.env` file and add your `GROQ_API_KEY`.

### 2. Run the Servers

You will need two separate terminals.

**Terminal 1: Run the Backend**
```bash
# From the project root directory
uvicorn backend.main:app --reload
```
The backend will be running at `http://127.0.0.1:8000`.

**Terminal 2: Run the Frontend**
```bash
# Navigate to the frontend directory
cd frontend

# Serve the files with Python's built-in server
python -m http.server
```
The frontend will be running at `http://localhost:8000`.

### 3. Usage

Open your browser and navigate to `http://localhost:8000`.

1.  Enter a research topic (e.g., "network security") and click **"Build Knowledge Graph"**.
2.  Select nodes in the graph (hold `Ctrl` or `Cmd` to select multiple).
3.  Use the **Agent Actions** in the sidebar to summarize, generate hypotheses, or chat with the selected papers.

# 📄 Local RAG Chat (LlamaIndex + Ollama)

A privacy-focused, fully local RAG (Retrieval-Augmented Generation) chat application. It allows you to chat with your own PDF, TXT, CSV, and Markdown documents using local LLMs (like Llama 3) and local embeddings (BGE-M3), powered by NVIDIA GPU acceleration.

## 🏗 Architecture

The application is built as a set of Dockerized microservices:

1. **Frontend (`/frontend`)**: A **Streamlit** web interface for chatting and model selection.
2. **Backend (`/backend`)**: A **FastAPI** proxy that orchestrates requests between the UI, the RAG service, and the LLM.
3. **RAG Service (`/rag`)**: A **FastAPI** service using **LlamaIndex** to chunk documents, generate embeddings (HuggingFace/BGE-M3), and retrieve context.
4. **LLM Service**: An **Ollama** container running local models (e.g., `llama3`, `mistral`).

## 🚀 Features

* **100% Local**: No data leaves your machine. API keys are not required.
* **GPU Accelerated**: Uses CUDA for both Embeddings and LLM inference.
* **Dynamic Model Switching**: Auto-detects available Ollama models and lets you switch via the UI.
* **Vector Search**: Uses LlamaIndex `VectorStoreIndex` for fast, accurate retrieval.
* **Context-Aware**: Injects retrieved document snippets into the prompt before sending to the LLM.

---

## 🛠 Prerequisites

* **Docker Desktop** (or Docker Engine + Compose)
* **NVIDIA GPU Drivers** (Recommended for performance. The setup assumes a CUDA-capable GPU).
* *Note: If running CPU-only, remove the `deploy: resources: reservations: devices` section from `docker-compose.yml`.*



---

## 📥 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pigcais.git
cd pigcais

```

### 2. Add Your Documents

Place your knowledge base files (PDF, TXT, CSV, etc.) into the `rag/documents` folder.

```bash
# Example
cp ~/Downloads/employee_handbook.pdf ./rag/documents/

```

### 3. Build and Start the Stack

```bash
docker-compose up -d --build

```

*Note: The first run will take time as it downloads the embedding models (BGE-M3) and builds the Python environments.*

### 4. Download an LLM (Crucial Step)

Ollama starts empty. You need to pull a model (like `llama3`) before chatting.

```bash
docker exec -it ollama_llm ollama pull llama3

```

*(You can also pull `mistral`, `gemma`, etc. The UI will automatically detect them.)*

---

## 🖥 Usage

1. Open your browser and navigate to **[http://localhost:8501](http://localhost:8501)**.
2. **Select Model**: Use the dropdown above the chat input to select `llama3:latest`.
3. **Ask a Question**: Type a query related to the documents you placed in the folder.
* *Example: "What is the policy on remote work mentioned in the handbook?"*


4. **View Response**: The system will retrieve relevant chunks from your documents and generate an answer.

---

## 🔧 Configuration Options

### Changing Chunking Strategy

Edit `rag/main.py` to change how documents are split:

```python
# rag/main.py
CHUNKING_METHOD = "fixed"  # Options: "fixed", "sentence"
CHUNK_SIZE = 512           # Token size
CHUNK_OVERLAP = 50         # Overlap size

```

### Changing the Embedding Model

The system uses `bge-large-zh-v1.5` by default. To change it, update `rag/main.py` and `rag/preload.py`:

```python
# rag/main.py
MODEL_ID = "BAAI/bge-large-zh-v1.5" # or "all-MiniLM-L6-v2", etc.

```

---

## 🐛 Troubleshooting

**1. "Connection Refused" or Models not showing in UI**

* Wait a few moments. The Backend might be trying to connect to Ollama before Ollama is fully ready.
* Refresh the web page.

**2. "Context found: False"**

* Ensure your documents are actually in `rag/documents`.
* Check the logs to see if the RAG service indexed them:
```bash
docker-compose logs -f rag

```


* Look for the line: `Processed X documents into Y nodes`.

**3. GPU Memory Issues**

* If you get CUDA Out of Memory errors, try reducing the `CHUNK_SIZE` in `main.py` or use a smaller LLM (e.g., `llama3:8b` instead of `70b`).

---

## 📁 Project Structure

```text
.
├── docker-compose.yml       # Orchestration
├── frontend/                # Streamlit UI
│   ├── Dockerfile
│   └── app.py
├── backend/                 # FastAPI Proxy
│   ├── Dockerfile
│   └── main.py
└── rag_service/             # LlamaIndex & Vector Store
    ├── Dockerfile
    ├── main.py
    ├── preload.py           # Pre-downloads embedding models
    ├── requirements.txt
    └── documents/           # <--- PUT YOUR FILES HERE

```
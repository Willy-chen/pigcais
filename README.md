# 📄 Local RAG Chat (LlamaIndex + Llama.cpp)

A privacy-focused, fully local RAG (Retrieval-Augmented Generation) chat application. It allows you to chat with your own PDF, TXT, CSV, and Markdown documents using local LLMs (via **llama.cpp**) and local embeddings (BGE-M3), powered by NVIDIA GPU acceleration.

## 🏗 Architecture

The application is built as a set of Dockerized microservices:

1. **Frontend (`/frontend`)**: A **Streamlit** web interface for chat.
2. **Backend (`/backend`)**: A **FastAPI** proxy that orchestrates requests between the UI, the RAG service, and the LLM.
3. **RAG Service (`/rag`)**: A **FastAPI** service using **LlamaIndex** to chunk documents, generate embeddings (HuggingFace/BGE-M3), and retrieve context.
4. **LLM Service**: A **llama.cpp** server container running highly optimized GGUF models with full GPU offloading.

## 🚀 Features

* **100% Local**: No data leaves your machine. No API keys required.
* **High-Performance Inference**: Uses `llama.cpp` server with CUDA (NVIDIA) to run quantized GGUF models efficiently.
* **Large Context Window**: Configured for **8192 token** context (expandable) to handle large document retrievals without truncation.
* **Vector Search**: Uses LlamaIndex `VectorStoreIndex` for accurate retrieval.
* **Context-Aware**: Injects retrieved document snippets into the prompt before sending to the LLM.

---

## 🛠 Prerequisites

* **Docker Desktop** (or Docker Engine + Compose)
* **NVIDIA GPU Drivers** (Required for GPU acceleration).
* **A GGUF Model**: You need to download a model file manually (e.g., Llama-3-8B-Instruct.Q4_K_M.gguf).

---

## 📥 Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/pigcais.git](https://github.com/your-username/pigcais.git)
cd pigcais

```

### 2. Download a Model (GGUF)

Create a `models` directory and place your `.gguf` model file inside it.

```bash
mkdir models
# Download a model (example using wget, or download via browser from HuggingFace)
# Recommended: Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
mv ~/Downloads/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf ./models/

```

*Note: Ensure the filename matches the path defined in your `docker-compose.yml` command.*

### 3. Add Your Documents

Place your knowledge base files (PDF, TXT, CSV, etc.) into the `rag/documents` folder.

```bash
# Example
cp ~/Downloads/employee_handbook.pdf ./rag/documents/

```

### 4. Build and Start the Stack

```bash
docker-compose up -d --build --force-recreate

```

*Note: The first run will download embedding models (BGE-M3) for the RAG service. The LLM service starts immediately if the GGUF file is present.*

---

## 🖥 Usage

1. Open your browser and navigate to **[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)**.
2. **Ask a Question**: Type a query related to the documents you placed in the folder.
* *Example: "What is the policy on remote work mentioned in the handbook?"*


3. **View Response**: The system will retrieve relevant chunks and stream the answer from the local model.

---

## 🔧 Configuration Options

### Adjusting Context Size (Llama.cpp)

To process larger documents, you can increase the context window in `docker-compose.yml` under `llamacpp_service`:

```yaml
command: -m /models/your-model.gguf -c 8192 ...

```

* `-c`: Context size (default is often 2048, set to 8192 or higher for RAG).
* `-np`: Number of parallel slots (set to 1 for maximum context per user).

### Changing Chunking Strategy (RAG)

Edit `rag/main.py` to change how documents are split:

```python
# rag/main.py
CHUNKING_METHOD = "fixed"  # Options: "fixed", "sentence"
CHUNK_SIZE = 512           # Token size
CHUNK_OVERLAP = 50         # Overlap size

```

---

## 🐛 Troubleshooting

**1. "Connection Error: Response ended prematurely"**

* This usually happens if the backend isn't handling the stream termination correctly. Ensure your `backend/main.py` includes the robust error handling for `ChunkedEncodingError`.

**2. Model Truncates / Stops early**

* Your context window is too small. Check `docker-compose.yml` and ensure `llamacpp_service` has `-c 8192` (or higher) in the command.
* Ensure you are running single-slot mode (`-np 1`) if you are the only user, to utilize the full context memory.

**3. "Context found: False"**

* Ensure your documents are in `rag/documents`.
* Check RAG logs: `docker-compose logs -f rag`. Look for "Processed X nodes".

**4. CUDA / GPU Memory Errors**

* If `llama.cpp` fails to start, your model might be too large for your VRAM.
* Try a higher quantization (e.g., Q4_K_M instead of Q8) or reduce the context size (`-c 4096`).

---

## 📁 Project Structure

```text
.
├── docker-compose.yml       # Orchestration (Llama.cpp + RAG + Backend + Frontend)
├── models/                  # <--- PLACE GGUF MODELS HERE
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
    └── documents/           # <--- PUT YOUR KNOWLEDGE BASE FILES HERE

```
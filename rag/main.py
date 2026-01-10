import os
import torch
import logging
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# Memory
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.llms import ChatMessage, MessageRole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Service")

app = FastAPI()

# Config
PERSIST_DIR = "./storage"
DOCS_DIR = "./documents"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Global Objects
index = None
chat_store = None

indexing_status = {
    "is_indexing": False,
    "current": 0,
    "total": 0,
    "message": "Initializing..."
}

# --- Initialization ---

def init_resources():
    global chat_store, index
    
    # 1. Setup Redis Memory
    try:
        chat_store = RedisChatStore(redis_url=REDIS_URL)
        logger.info("Connected to Redis for Chat History.")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Memory will be ephemeral.")
        chat_store = None

    # 2. Setup Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-zh-v1.5", 
        device=device
    )
    Settings.llm = None # We use Backend for LLM

    # 3. Load or Build Index
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            indexing_status["message"] = "Ready (Loaded from disk)"
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            build_empty_index()
    else:
        build_empty_index()

def build_empty_index():
    global index
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents) if documents else VectorStoreIndex([])
    if documents:
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    indexing_status["message"] = "Ready"

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=init_resources).start()

# --- Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: str
    selected_files: Optional[List[str]] = None

class SaveTurnRequest(BaseModel):
    session_id: str
    user_query: str
    ai_response: str

class DocumentRequest(BaseModel):
    filename: str

# --- Endpoints ---

@app.get("/status")
def get_status():
    return indexing_status

@app.post("/add_document")
def add_document(req: DocumentRequest):
    def worker(filename):
        global index
        indexing_status["is_indexing"] = True
        indexing_status["message"] = f"Indexing {filename}..."
        try:
            file_path = os.path.join(DOCS_DIR, filename)
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
            index.insert_nodes(SentenceSplitter().get_nodes_from_documents(docs))
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            indexing_status["message"] = "Ready"
        except Exception as e:
            indexing_status["message"] = f"Error: {e}"
        finally:
            indexing_status["is_indexing"] = False

    threading.Thread(target=worker, args=(req.filename,)).start()
    return {"status": "Processing started"}

@app.post("/construct_prompt")
def construct_prompt(req: QueryRequest):
    """Retrieves context and history, builds a prompt for the LLM."""
    global index, chat_store
    
    if index is None:
        return {"prompt": req.query, "context_found": False}

    # 1. Retrieve Context
    filters = None
    if req.selected_files:
        metadata_filters = [MetadataFilter(key="file_name", value=f) for f in req.selected_files]
        filters = MetadataFilters(filters=metadata_filters, condition="or")

    retriever = index.as_retriever(similarity_top_k=3, filters=filters)
    nodes = retriever.retrieve(req.query)
    context_str = "\n\n".join([n.get_content() for n in nodes]) if nodes else "No relevant documents found."

    # 2. Retrieve History from Redis
    history_str = ""
    if chat_store:
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=2000, 
            chat_store=chat_store, 
            chat_store_key=req.session_id
        )
        # Get recent messages
        msgs = memory.get()
        history_str = "\n".join([f"{m.role.upper()}: {m.content}" for m in msgs])

    # 3. Formulate Prompt
    final_prompt = f"""You are a helpful AI assistant.
    
HISTORY:
{history_str}

CONTEXT:
{context_str}

USER: {req.query}
ASSISTANT:"""
    logger.log(logging.INFO, final_prompt)
    return {"prompt": final_prompt}

@app.post("/save_turn")
def save_turn(req: SaveTurnRequest):
    """Saves the conversation turn to Redis."""
    global chat_store
    if chat_store:
        memory = ChatMemoryBuffer.from_defaults(
            chat_store=chat_store, 
            chat_store_key=req.session_id
        )
        memory.put(ChatMessage(role=MessageRole.USER, content=req.user_query))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=req.ai_response))
        return {"status": "saved"}
    return {"status": "no_store"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
import os
import torch
import threading
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

# --- LlamaIndex Imports ---
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

app = FastAPI()
index = None
PERSIST_DIR = "./storage"
DOCS_DIR = "./documents"

indexing_status = {
    "is_indexing": False,
    "current": 0,
    "total": 0,
    "message": "Idle"
}

class QueryRequest(BaseModel):
    query: str
    selected_files: Optional[List[str]] = None

class DocumentRequest(BaseModel):
    filename: str

def update_status(is_indexing, current, total, message):
    indexing_status["is_indexing"] = is_indexing
    indexing_status["current"] = current
    indexing_status["total"] = total
    indexing_status["message"] = message

def build_index_background():
    global index
    update_status(True, 0, 0, "Initializing...")

    try:
        # 1. Setup Embeddings (if not already done)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Embedding Model in {device}...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-zh-v1.5", 
            device=device
        )
        Settings.llm = None 
        
        # 2. Check for Persistent Index First
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            update_status(True, 0, 0, "Loading existing index from disk...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                index = load_index_from_storage(storage_context)
                update_status(False, 100, 100, "Ready")
                print("Index loaded from storage.")
                return
            except Exception as e:
                print(f"Failed to load storage: {e}. Rebuilding...")

        # 3. Build from scratch
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
            
        update_status(True, 0, 0, "Reading documents...")
        reader = SimpleDirectoryReader(DOCS_DIR, recursive=True)
        documents = reader.load_data()
        
        if not documents:
            index = VectorStoreIndex([])
            update_status(False, 0, 0, "Ready (No documents found)")
            return

        # 4. Process Documents with Progress
        total_docs = len(documents)
        update_status(True, 0, total_docs, "Processing nodes...")
        
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = []
        
        # Manually loop to update progress
        for i, doc in enumerate(documents):
            # Update status every 5 documents or so to reduce overhead
            if i % 5 == 0:
                update_status(True, i, total_docs, f"Indexing document {i+1}/{total_docs}")
            
            nodes.extend(parser.get_nodes_from_documents([doc]))
            
        update_status(True, total_docs, total_docs, "Building vector store...")
        index = VectorStoreIndex(nodes)
        
        update_status(True, total_docs, total_docs, "Saving to disk...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        update_status(False, total_docs, total_docs, "Ready")
        print("Index built and saved.")

    except Exception as e:
        update_status(False, 0, 0, f"Error: {str(e)}")
        print(f"Indexing failed: {e}")

@app.on_event("startup")
async def startup_event():
    # Start indexing in a separate thread so the API starts immediately
    thread = threading.Thread(target=build_index_background)
    thread.start()

@app.get("/status")
def get_status():
    return indexing_status

def add_document_worker(filename: str):
    global index
    file_path = os.path.join(DOCS_DIR, filename)
    
    try:
        # 1. Update Status to "Busy"
        update_status(True, 0, 1, f"Reading {filename}...")
        
        # 2. Load & Chunk
        new_docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        new_nodes = parser.get_nodes_from_documents(new_docs)
        
        # 3. Update Status to "Indexing"
        update_status(True, 0, len(new_nodes), f"Indexing {filename}...")
        
        # 4. Insert & Persist (The heavy part)
        index.insert_nodes(new_nodes)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        # 5. Reset Status
        update_status(False, 100, 100, "Ready")
        print(f"Successfully added {filename}")
        
    except Exception as e:
        print(f"Error adding document: {e}")
        update_status(False, 0, 0, f"Error: {str(e)}")

@app.post("/add_document")
def add_document(req: DocumentRequest):
    """
    Returns immediately and runs indexing in the background.
    """
    global index
    if not index:
         # If system is initializing, queueing is complex, so we just reject or trigger build
         return {"status": "System is initializing, please wait."}

    file_path = os.path.join(DOCS_DIR, req.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # --- KEY CHANGE: Run in background thread ---
    thread = threading.Thread(target=add_document_worker, args=(req.filename,))
    thread.start()
    
    # Return immediately so Backend doesn't timeout
    return {"status": "Indexing started in background"}

@app.post("/construct_prompt")
def construct_prompt(req: QueryRequest):
    """
    Retrieves documents via LlamaIndex and builds the prompt string.
    """
    global index
    if index is None:
        return {"prompt": req.query, "context_found": False, "note": "System is still indexing..."}
    
    # --- Dynamic Filtering ---
    filters = None
    if req.selected_files:
        # Create a filter that matches ANY of the selected filenames
        # LlamaIndex 'file_name' metadata is set automatically by SimpleDirectoryReader
        metadata_filters = [
            MetadataFilter(key="file_name", value=file_name) 
            for file_name in req.selected_files
        ]
        # logic="OR" means: retrieve if file_name is A OR file_name is B
        filters = MetadataFilters(filters=metadata_filters, condition="or")

    # --- Create Retriever with Filters ---
    # We instantiate the retriever per-request to apply different filters
    retriever = index.as_retriever(
        similarity_top_k=3, 
        filters=filters
    )
    
    results = retriever.retrieve(req.query)

    if not results:
         return {"prompt": req.query, "context_found": False}
    
    # --- Format Context ---
    context_list = []
    for node_with_score in results:
        # LlamaIndex stores the filename in metadata 'file_name' by default
        source = node_with_score.metadata.get("file_name", "unknown")
        text = node_with_score.get_content().strip()
        context_list.append(f"[Source: {source}]\n{text}")
    
    context_text = "\n\n".join(context_list)
    
    # --- Build Prompt ---
    final_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer is not in the context, say so, but try to be helpful.

CONTEXT INFORMATION:
---------------------
{context_text}
---------------------

USER QUESTION:
{req.query}
"""
    print("prompt: ", final_prompt, flush=True)
    return {"prompt": final_prompt, "context_found": True}
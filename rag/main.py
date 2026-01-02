import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# --- LlamaIndex Imports ---
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

app = FastAPI()

# Global variables
retriever = None

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    global retriever
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"RAG Service (LlamaIndex) starting on: {device.upper()}")

    # 1. Setup Embeddings
    # Using BAAI/bge-m3 or all-MiniLM-L6-v2 depending on preference
    print("Loading Embedding Model...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="all-MiniLM-L6-v2", 
        device=device
    )
    Settings.llm = None  # We don't use the LLM here, only for prompt construction

    # 2. Load Documents
    docs_dir = "./documents"
    if not os.path.exists(docs_dir):
        print(f"Warning: {docs_dir} not found.")
        return

    print("Loading documents from disk...")
    # SimpleDirectoryReader automatically handles txt, pdf, csv, etc.
    documents = SimpleDirectoryReader(docs_dir, recursive=True).load_data()
    
    if not documents:
        print("No documents found.")
        return

    # 3. Chunking (Node Parsing)
    # Similar to RecursiveCharacterTextSplitter: chunk_size=1000, overlap=100
    parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Processed {len(documents)} documents into {len(nodes)} nodes.")

    # 4. Build Index
    index = VectorStoreIndex(nodes)
    
    # 5. Create Retriever
    # This acts as the vector search engine
    retriever = index.as_retriever(similarity_top_k=3)
    print("RAG System Ready.")

@app.post("/construct_prompt")
def construct_prompt(req: QueryRequest):
    """
    Retrieves documents via LlamaIndex and builds the prompt string.
    """
    if not retriever:
        return {"prompt": req.query, "context_found": False}
    
    # --- Retrieve ---
    results = retriever.retrieve(req.query)
    
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
    return {"prompt": final_prompt, "context_found": True}
import os
import json
import shutil
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

RAG_URL = os.getenv("RAG_URL", "http://rag:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama_service:11434")
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://llamacpp_service:8080")
DOCS_DIR = "/app/documents"

class ChatRequest(BaseModel):
    message: str
    model: str
    selected_files: Optional[List[str]] = None
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def get_models():
    """
    Returns a combined list of models:
    1. The 'Production' Llama.cpp model (if running).
    2. All available 'Test' models from Ollama.
    """
    model_list = []

    # 1. Check Llama.cpp (Production/Parallel)
    try:
        # A simple health check to see if the server is up
        r = requests.get(f"{LLAMACPP_URL}/health", timeout=0.5)
        if r.status_code == 200:
            # We give this a special name so the user knows it's the fast one
            model_list.append("[Production] Llama.cpp")
    except:
        pass # It might be down or loading, that's fine

    # 2. Check Ollama (Test/Dev)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1)
        if r.status_code == 200:
            data = r.json()
            for m in data.get('models', []):
                # Prefix to distinguish them
                model_list.append(f"[Test] {m['name']}")
    except:
        pass

    if not model_list:
        return {"models": ["Error: No models available"]}

    return {"models": model_list}

@app.get("/documents")
def list_documents():
    """List all files in the documents directory."""
    if not os.path.exists(DOCS_DIR):
        return []
    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    return {"files": files}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a file and incrementally add it to the RAG index."""
    try:
        os.makedirs(DOCS_DIR, exist_ok=True)
        file_path = os.path.join(DOCS_DIR, file.filename)
        
        # 1. Save File to Disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Trigger Incremental Indexing
        try:
            # We send the filename to the RAG service so it knows exactly what to add
            payload = {"filename": file.filename}
            requests.post(f"{RAG_URL}/add_document", json=payload, timeout=60)
        except Exception as e:
            print(f"Warning: Failed to add document to index: {e}")

        return {"filename": file.filename, "status": "uploaded and indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a file."""
    file_path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted", "filename": filename}
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/indexing_status")
def get_indexing_status():
    """Proxy status check to RAG service"""
    try:
        # Timeout is short because we want quick UI updates
        res = requests.get(f"{RAG_URL}/status", timeout=1)
        return res.json()
    except:
        return {"is_indexing": False, "message": "RAG Service Unreachable"}

def stream_llamacpp(prompt):
    url = f"{LLAMACPP_URL}/v1/chat/completions"
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_tokens": 8192, # Increased to match your new capacity
        "temperature": 0.7
    }

    try:
        # We use a session to allow for better connection handling
        with requests.Session() as session:
            with session.post(url, json=payload, stream=True) as r:
                r.raise_for_status()
                
                try:
                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode('utf-8').strip()
                            if decoded.startswith("data: "):
                                data_str = decoded[6:] 
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    pass
                except requests.exceptions.ChunkedEncodingError:
                    # Llama.cpp sometimes closes the stream abruptly after the last token.
                    # This is not a real error if we already got data.
                    pass
                except Exception as inner_e:
                    # Only yield error if it's NOT a disconnect at the end
                    yield f"\n[Stream Error: {str(inner_e)}]"

    except Exception as e:
        yield f"Connection Error: {str(e)}"

def stream_ollama(prompt, model_tag):
    """Handler for Ollama"""
    # Remove the "[Test] " prefix to get the real model name
    real_name = model_tag.replace("[Test] ", "")
    
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": real_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                except:
                    pass

@app.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest):
    final_prompt = req.message
    # 1. RAG Retrieval
    try:
        rag_payload = {
            "query": req.message,
            "selected_files": req.selected_files # Pass the list from Frontend
        }
        rag_res = requests.post(f"{RAG_URL}/construct_prompt", json=rag_payload)
        
        if rag_res.status_code == 200:
            final_prompt = rag_res.json().get("prompt", req.message)
    except Exception as e:
        print(f"RAG Error: {e}")

    # (Rest of routing logic remains the same)
    if "[Production]" in req.model:
        return StreamingResponse(stream_llamacpp(final_prompt), media_type="text/plain")
    else:
        return StreamingResponse(stream_ollama(final_prompt, req.model), media_type="text/plain")
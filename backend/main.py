import os
import json
import shutil
import logging
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Service URLs
RAG_URL = os.getenv("RAG_URL", "http://localhost:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://localhost:8080")
DOCS_DIR = "./documents"

class ChatRequest(BaseModel):
    message: str
    model: str
    session_id: str  # <--- Required for memory
    selected_files: Optional[List[str]] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def get_models():
    """Combines Llama.cpp and Ollama models."""
    model_list = []
    
    # Check Llama.cpp
    try:
        r = requests.get(f"{LLAMACPP_URL}/health", timeout=0.5)
        if r.status_code == 200:
            model_list.append("[Production] Llama.cpp")
    except:
        pass

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1)
        if r.status_code == 200:
            data = r.json()
            for m in data.get('models', []):
                model_list.append(f"[Test] {m['name']}")
    except:
        pass

    return {"models": model_list if model_list else ["Error: No models available"]}

@app.get("/documents")
def list_documents():
    if not os.path.exists(DOCS_DIR):
        return {"files": []}
    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    return {"files": files}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs(DOCS_DIR, exist_ok=True)
        file_path = os.path.join(DOCS_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger RAG Indexing
        try:
            payload = {"filename": file.filename}
            requests.post(f"{RAG_URL}/add_document", json=payload, timeout=60)
        except Exception as e:
            logger.error(f"RAG Indexing trigger failed: {e}")

        return {"filename": file.filename, "status": "uploaded and indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexing_status")
def get_indexing_status():
    try:
        res = requests.get(f"{RAG_URL}/status", timeout=1)
        return res.json()
    except:
        return {"is_indexing": False, "message": "RAG Service Unreachable"}

# --- Streaming Generators ---

def stream_llamacpp(prompt):
    url = f"{LLAMACPP_URL}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.7
    }

    try:
        with requests.Session() as session:
            with session.post(url, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8').strip()
                        if decoded.startswith("data: "):
                            data_str = decoded[6:] 
                            if data_str == "[DONE]": break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0]["delta"].get("content", "")
                                    if content: yield content
                            except: pass
    except Exception as e:
        yield f"Error: {str(e)}"

def stream_ollama(prompt, model_tag):
    real_name = model_tag.replace("[Test] ", "")
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": real_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    try:
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content: yield content
                    except: pass
    except Exception as e:
        yield f"Error: {str(e)}"

# --- Main Chat Endpoint ---

@app.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest):
    final_prompt = req.message
    
    # 1. Construct Prompt (History + Context) via RAG Service
    try:
        rag_payload = {
            "query": req.message,
            "session_id": req.session_id,
            "selected_files": req.selected_files
        }
        rag_res = requests.post(f"{RAG_URL}/construct_prompt", json=rag_payload)
        
        if rag_res.status_code == 200:
            data = rag_res.json()
            final_prompt = data.get("prompt", req.message)
            logger.info("Prompt constructed with RAG context.")
    except Exception as e:
        logger.error(f"RAG Error: {e}")

    # 2. Select Generator
    if "[Production]" in req.model:
        generator = stream_llamacpp(final_prompt)
    else:
        generator = stream_ollama(final_prompt, req.model)

    # 3. Stream & Accumulate for Memory
    async def response_wrapper():
        full_response = ""
        for chunk in generator:
            full_response += chunk
            yield chunk
        
        # 4. Save Turn to Memory (After stream finishes)
        try:
            save_payload = {
                "session_id": req.session_id,
                "user_query": req.message,
                "ai_response": full_response
            }
            requests.post(f"{RAG_URL}/save_turn", json=save_payload, timeout=2)
            logger.info(f"Memory saved for session {req.session_id}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    return StreamingResponse(response_wrapper(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
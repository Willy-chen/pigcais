import os
import requests
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

RAG_URL = os.getenv("RAG_URL", "http://rag:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama_service:11434")
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://llamacpp_service:8080")

class ChatRequest(BaseModel):
    message: str
    model: str
    
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
    # 1. RAG Retrieval
    final_prompt = req.message
    try:
        rag_res = requests.post(f"{RAG_URL}/construct_prompt", json={"query": req.message})
        if rag_res.status_code == 200:
            final_prompt = rag_res.json().get("prompt", req.message)
    except Exception as e:
        print(f"RAG Error: {e}")

    # 2. Routing Logic
    if "[Production]" in req.model:
        return StreamingResponse(stream_llamacpp(final_prompt), media_type="text/plain")
    else:
        # Default to Ollama for everything else
        return StreamingResponse(stream_ollama(final_prompt, req.model), media_type="text/plain")
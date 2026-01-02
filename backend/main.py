import os
import requests
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

RAG_URL = os.getenv("RAG_URL", "http://localhost:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

class ChatRequest(BaseModel):
    message: str
    model: str = "llama3"

# --- NEW ENDPOINT: List Available Models ---
@app.get("/models")
def get_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            data = response.json()
            # Extract just the model names (e.g., "llama3:latest")
            model_names = [model["name"] for model in data.get("models", [])]
            return {"models": model_names}
        else:
            return {"models": [], "error": "Failed to fetch from Ollama"}
    except Exception as e:
        return {"models": [], "error": str(e)}

# --- Existing Streaming Logic ---
def stream_generator(model: str, prompt: str):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_data = json.loads(decoded_line)
                        token = json_data.get("response", "")
                        if token: yield token
                        if json_data.get("done", False): break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"Error streaming from Ollama: {str(e)}"

@app.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest):
    # 1. Get Constructed Prompt from RAG
    try:
        rag_res = requests.post(f"{RAG_URL}/construct_prompt", json={"query": req.message})
        if rag_res.status_code == 200:
            final_prompt = rag_res.json().get("prompt", req.message)
        else:
            final_prompt = req.message
    except:
        final_prompt = req.message

    # 2. Return Streaming Response
    return StreamingResponse(
        stream_generator(req.model, final_prompt), 
        media_type="text/plain"
    )
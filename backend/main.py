import os
import json
import shutil
import logging
import requests
from typing import List, Optional

import jwt
import database
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Configuration ---
SECRET_KEY = "SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 days
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Service URLs
RAG_URL = os.getenv("RAG_URL", "http://localhost:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://localhost:8080")
DOCS_DIR = "./documents"

class ChatRequest(BaseModel):
    message: str
    model: str
    session_id: str 
    selected_files: Optional[List[str]] = None

class UserAuth(BaseModel):
    username: str
    password: str

class NewSession(BaseModel):
    title: str

class UrlRequest(BaseModel):
    url: str

@app.on_event("startup")
def startup():
    # Wait loop logic could be added here if DB takes time to start
    try:
        database.init_db()
    except Exception as e:
        logger.error(f"DB Init Failed: {e}")

# --- Authentication Helpers ---

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None: 
            raise ValueError("Token missing 'sub' (User ID)")
        return int(user_id)
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid Token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Unexpected Auth Error: {e}") # <--- THIS WILL SHOW IN LOGS
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- NEW: Authentication Endpoints ---

@app.post("/auth/register")
def register(user: UserAuth):
    """Register a new user."""
    if database.create_user(user.username, user.password):
        return {"status": "User created successfully"}
    raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/auth/login")
def login(user: UserAuth):
    """Login and return JWT token."""
    user_id = database.verify_user(user.username, user.password)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # FIX: Convert user_id to string for the JWT 'sub' claim
    access_token = create_access_token(data={"sub": str(user_id)})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Session Management ---

@app.get("/sessions")
def get_sessions(user_id: int = Depends(get_current_user_id)):
    sessions = database.get_user_sessions(user_id)
    # Convert UUIDs to strings for JSON serialization
    for s in sessions:
        s['id'] = str(s['id'])
    return {"sessions": sessions}

@app.post("/sessions")
def create_session(s: NewSession, user_id: int = Depends(get_current_user_id)):
    session_id = database.create_session(user_id, s.title)
    return {"session_id": session_id, "title": s.title}

@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str, user_id: int = Depends(get_current_user_id)):
    """Delete a chat session and all its messages."""
    success = database.delete_session(session_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or access denied")
    return {"status": "deleted", "session_id": session_id}

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, user_id: int = Depends(get_current_user_id)):
    messages = database.get_session_messages(session_id)
    return {"messages": messages}

# --- System & Model Endpoints ---

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

@app.post("/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Save temp file
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        import audio_inference
        import train_xgb
        
        # If model doesn't exist, train it!
        if getattr(audio_inference, 'XGB_MODEL_PATH', None):
            if not os.path.exists(audio_inference.XGB_MODEL_PATH) and not os.path.exists("ultimate_xgb.json"):
                # Run the train_and_save to create the model json since it doesn't exist
                logger.info("XGB model not found, training on the fly...")
                train_xgb.train_and_save()
                
        processor = audio_inference.get_processor()
        
        async def event_generator():
            try:
                for chunk in processor.predict(temp_path):
                    yield json.dumps(chunk) + "\n"
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        return StreamingResponse(event_generator(), media_type="application/x-ndjson")
        
    except Exception as e:
        logger.error(f"Audio Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_url")
async def analyze_url(req: UrlRequest):
    try:
        # Download remote file
        filename = req.url.split("/")[-1].split("?")[0]
        if not filename.lower().endswith(".wav"):
            filename += ".wav"
        temp_path = f"/tmp/{filename}"
        
        logger.info(f"Downloading remote audio: {req.url}")
        with requests.get(req.url, stream=True) as r:
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        import audio_inference
        import train_xgb
        
        # If model doesn't exist, train it!
        if getattr(audio_inference, 'XGB_MODEL_PATH', None):
            if not os.path.exists(audio_inference.XGB_MODEL_PATH) and not os.path.exists("ultimate_xgb.json"):
                logger.info("XGB model not found, training on the fly...")
                train_xgb.train_and_save()
                
        processor = audio_inference.get_processor()
        
        async def event_generator():
            try:
                for chunk in processor.predict(temp_path):
                    yield json.dumps(chunk) + "\n"
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        return StreamingResponse(event_generator(), media_type="application/x-ndjson")
        
    except Exception as e:
        logger.error(f"Remote Audio Analysis failed: {e}")
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
    
    # 1. Save User Message to Postgres
    try:
        database.add_message(req.session_id, "user", req.message)
    except Exception as e:
        logger.error(f"Database Error: {e}")
        raise HTTPException(status_code=404, detail="Session not found. Please create a new chat.")
    
    final_prompt = req.message
    
    # 2. Construct Prompt (History + Context) via RAG Service
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

    # 3. Select Generator
    if "[Production]" in req.model:
        generator = stream_llamacpp(final_prompt)
    else:
        generator = stream_ollama(final_prompt, req.model)

    # 4. Stream & Accumulate for Memory
    async def response_wrapper():
        full_response = ""
        for chunk in generator:
            full_response += chunk
            yield chunk
        
        # 5. Save AI Response to Database (CRITICAL FIX: Ensure history is saved)
        try:
            database.add_message(req.session_id, "assistant", full_response)
        except Exception as e:
            logger.error(f"Failed to save assistant message to DB: {e}")

        # 6. Save Turn to RAG Memory (Redis/Vector)
        try:
            save_payload = {
                "session_id": req.session_id,
                "user_query": req.message,
                "ai_response": full_response
            }
            requests.post(f"{RAG_URL}/save_turn", json=save_payload, timeout=2)
        except Exception as e:
            logger.error(f"Failed to save RAG memory: {e}")

    return StreamingResponse(response_wrapper(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
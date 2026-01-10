import streamlit as st
import requests
import os
import uuid
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat", layout="wide")

# --- Fetch Available Models ---
@st.cache_data(ttl=60)
def fetch_models():
    try:
        response = requests.get(f"{BACKEND_URL}/models")
        if response.status_code == 200:
            return response.json().get("models", ["llama3"])
        return ["llama3"] 
    except:
        return ["llama3"]

available_models = fetch_models()

# --- Session State ---
if "sessions" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.sessions = {new_id: {'title': "New Chat", 'messages': []}}
    st.session_state.current_session = new_id

if "current_session" not in st.session_state:
    st.session_state.current_session = list(st.session_state.sessions.keys())[0]

# --- Sidebar: Upload & Status ---
with st.sidebar:
    st.header("📂 Document Library")
    status_placeholder = st.empty()

    # Indexing Status
    try:
        # Increased timeout to 5s because embedding takes CPU power
        status_res = requests.get(f"{BACKEND_URL}/indexing_status", timeout=5)
        
        if status_res.status_code == 200:
            status_data = status_res.json()
            
            if status_data.get("is_indexing"):
                # Calculate progress
                curr = status_data.get("current", 0)
                tot = status_data.get("total", 1)
                if tot == 0: tot = 1
                prog_val = min(curr / tot, 1.0)
                
                # Show Progress Bar
                status_placeholder.progress(prog_val, text=f"{status_data.get('message')}")
                
                # Wait longer before refreshing to reduce load
                time.sleep(2) 
                st.rerun()
            else:
                status_placeholder.success("System Ready")
        else:
            status_placeholder.warning(f"Status Unknown ({status_res.status_code})")
            
    except requests.exceptions.ConnectionError:
        # Only show error if it persists (you could add a counter here)
        status_placeholder.error("Backend Offline")
    except requests.exceptions.Timeout:
        status_placeholder.warning("Backend Busy")
    except Exception as e:
        status_placeholder.error(f"Error: {e}")

    # Upload
    with st.expander("Upload Documents"):
        uploaded_file = st.file_uploader("Select file", type=["txt", "pdf", "md", "csv", "json", "docx"])
        if uploaded_file:
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    res = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success("Uploaded!")
                        time.sleep(0.5)
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # File Selector
    st.divider()
    try:
        res = requests.get(f"{BACKEND_URL}/documents")
        file_list = res.json().get("files", []) if res.status_code == 200 else []
    except:
        file_list = []

    selected_files = []
    if file_list:
        with st.form("file_selector"):
            st.write(f"Available Files ({len(file_list)})")
            for f in file_list:
                if st.checkbox(f, value=True, key=f):
                    selected_files.append(f)
            st.form_submit_button("Update Context")

# --- Sidebar: Chats ---
with st.sidebar:
    st.divider()
    st.subheader("Chats")
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = {'title': "New Chat", 'messages': []}
        st.session_state.current_session = new_id
        st.rerun()
    
    for sid, sdata in st.session_state.sessions.items():
        label = sdata['title']
        if sid == st.session_state.current_session:
            label = f"🟢 {label}"
        if st.button(label, key=sid):
            st.session_state.current_session = sid
            st.rerun()

# --- Main Chat ---
current_id = st.session_state.current_session
current_messages = st.session_state.sessions[current_id]['messages']

st.title("📄 DocuChat RAG")
st.markdown(f"**Session Docs:** {len(selected_files)} selected")

# Display History
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Model Select
col1, _ = st.columns([1, 4])
with col1:
    selected_model = st.selectbox("Model", available_models, index=0)

# Input
if prompt := st.chat_input("Ask a question..."):
    # Title update
    if not current_messages:
        st.session_state.sessions[current_id]['title'] = prompt[:20]

    # User message UI
    st.session_state.sessions[current_id]['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Send session_id for memory handling
            payload = {
                "message": prompt, 
                "model": selected_model, 
                "selected_files": selected_files,
                "session_id": current_id 
            }
            
            with requests.post(f"{BACKEND_URL}/chat_stream", json=payload, stream=True) as r:
                if r.status_code == 200:
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            text_chunk = chunk.decode("utf-8")
                            full_response += text_chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.error(f"Error: {r.text}")
        except Exception as e:
            message_placeholder.error(f"Connection Error: {e}")
            
        st.session_state.sessions[current_id]['messages'].append({"role": "assistant", "content": full_response})
import streamlit as st
import requests
import os
import time

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="RAG Chat", layout="wide")

# --- 1. Session State Initialization ---
if "token" not in st.session_state:
    st.session_state.token = None

if "current_session" not in st.session_state:
    st.session_state.current_session = None

# --- 2. Auth Helpers ---
def login_user(username, password):
    try:
        res = requests.post(f"{BACKEND_URL}/auth/login", json={"username": username, "password": password})
        if res.status_code == 200:
            return res.json()["access_token"]
        return None
    except:
        return None

def register_user(username, password):
    try:
        res = requests.post(f"{BACKEND_URL}/auth/register", json={"username": username, "password": password})
        return res.status_code == 200
    except:
        return False

# --- 3. Login Screen ---
if not st.session_state.token:
    st.title("🔐 Login to DocuChat")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                token = login_user(u, p)
                if token:
                    st.session_state.token = token
                    st.success("Logged in")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("reg"):
            u = st.text_input("New User")
            p = st.text_input("New Pass", type="password")
            if st.form_submit_button("Register"):
                if register_user(u, p):
                    st.success("Created! Please login.")
                else:
                    st.error("Error creating user")
    st.stop()

# --- 4. Main App (Logged In) ---
auth_headers = {"Authorization": f"Bearer {st.session_state.token}"}

# --- Sidebar ---
with st.sidebar:
    st.write("👤 Logged In")
    if st.button("Logout"):
        st.session_state.token = None
        st.session_state.current_session = None
        st.rerun()
    
    st.divider()

    # --- Document Upload ---
    with st.expander("📂 Upload Documents"):
        uploaded_file = st.file_uploader("Select file", type=["txt", "pdf", "md"])
        if uploaded_file:
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    res = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success("Uploaded!")
                    else:
                        st.error("Upload failed")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- File Selector ---
    try:
        res = requests.get(f"{BACKEND_URL}/documents")
        all_files = res.json().get("files", []) if res.status_code == 200 else []
    except:
        all_files = []
    
    selected_files = []
    if all_files:
        st.divider()
        st.write("Select Context:")
        with st.form("files"):
            for f in all_files:
                if st.checkbox(f, value=True, key=f):
                    selected_files.append(f)
            st.form_submit_button("Update Context")

    # --- Chat Management ---
    st.divider()
    st.header("Chat Rooms")

    # Create New Chat (API CALL)
    if st.button("➕ New Chat", use_container_width=True):
        res = requests.post(f"{BACKEND_URL}/sessions", json={"title": "New Chat"}, headers=auth_headers)
        if res.status_code == 200:
            st.session_state.current_session = res.json()["session_id"]
            st.rerun()

    # Load Sessions (API CALL)
    try:
        res = requests.get(f"{BACKEND_URL}/sessions", headers=auth_headers)
        sessions = res.json().get("sessions", [])
    except:
        sessions = []

    # Auto-Select first session if none selected
    if not st.session_state.current_session and sessions:
        st.session_state.current_session = str(sessions[0]["id"])
    elif not sessions and not st.session_state.current_session:
        # If NO sessions exist at all, force create one
        res = requests.post(f"{BACKEND_URL}/sessions", json={"title": "New Chat"}, headers=auth_headers)
        if res.status_code == 200:
            st.session_state.current_session = res.json()["session_id"]
            st.rerun()

    # Render Session List
    for sess in sessions:
        sid = str(sess["id"])
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            label = sess["title"]
            if st.session_state.current_session == sid:
                label = f"🟢 {label}"
            if st.button(label, key=f"s_{sid}", use_container_width=True):
                st.session_state.current_session = sid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"d_{sid}"):
                requests.delete(f"{BACKEND_URL}/sessions/{sid}", headers=auth_headers)
                if st.session_state.current_session == sid:
                    st.session_state.current_session = None
                st.rerun()

# --- Main Chat Area ---
if st.session_state.current_session:
    sess_id = st.session_state.current_session

    # 1. Load History (API CALL)
    try:
        res = requests.get(f"{BACKEND_URL}/sessions/{sess_id}/messages", headers=auth_headers)
        messages = res.json().get("messages", [])
    except:
        messages = []

    st.title("📄 DocuChat RAG")
    
    # 2. Display Messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. Chat Input
    if prompt := st.chat_input("Ask something..."):
        # Optimistic UI Update
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Stream Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            payload = {
                "message": prompt,
                "model": "llama3",
                "session_id": sess_id,
                "selected_files": selected_files
            }
            
            try:
                # Use requests.post with stream=True
                with requests.post(f"{BACKEND_URL}/chat_stream", json=payload, stream=True) as r:
                    if r.status_code == 200:
                        for chunk in r.iter_content(chunk_size=None):
                            if chunk:
                                text = chunk.decode("utf-8")
                                full_response += text
                                placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        
                        # (Optional) Update Title after first message?
                        # You can add logic here to PATCH the session title if len(messages) == 0
                    elif r.status_code == 404:
                         placeholder.error("Session expired. Please create a new chat.")
                    else:
                        placeholder.error(f"Error: {r.status_code}")
            except Exception as e:
                placeholder.error(f"Connection Error: {e}")
import streamlit as st
import requests
import os
import uuid
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat", layout="wide")

# --- Fetch Available Models ---
@st.cache_data(ttl=60) # Cache for 60 seconds so we don't spam the backend
def fetch_models():
    try:
        response = requests.get(f"{BACKEND_URL}/models")
        if response.status_code == 200:
            return response.json().get("models", ["llama3"])
        return ["llama3"] # Fallback
    except:
        return ["llama3"] # Fallback

available_models = fetch_models()

# --- Session State ---
if "sessions" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.sessions = {new_id: {'title': "New Chat", 'messages': []}}
    st.session_state.current_session = new_id

if "current_session" not in st.session_state:
    st.session_state.current_session = list(st.session_state.sessions.keys())[0]

with st.sidebar:
    st.header("📂 Document Library")
    st.subheader("System Status")
    status_placeholder = st.empty()
    try:
        status_res = requests.get(f"{BACKEND_URL}/indexing_status")
        if status_res.status_code == 200:
            status_data = status_res.json()
            
            if status_data.get("is_indexing"):
                current = status_data.get("current", 0)
                total = status_data.get("total", 1) # avoid div/0
                if total == 0: total = 1
                
                progress = min(current / total, 1.0)
                status_placeholder.progress(progress, text=status_data.get("message"))
                
                # Auto-refresh if still indexing
                time.sleep(1)
                st.rerun()
            else:
                status_placeholder.success(f"System Ready: {status_data.get('message')}")
    except Exception as e:
        status_placeholder.error(f"Connection Error: {e}")

    # 1. Upload Section
    with st.expander("Upload Documents", expanded=True):
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "md", "csv", "json", "docx"])
        if uploaded_file:
            # 1. Show a spinner while sending to backend
            with st.spinner("Uploading file..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"File '{uploaded_file.name}' uploaded!")
                        
                        # 2. FORCE a rerun so the Status Bar picks up the "Busy" state
                        time.sleep(0.5) 
                        st.rerun()
                    else:
                        st.error("Upload failed.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    
    # 2. File List & Selection
    st.divider()
    st.subheader("Available Files")

    # Fetch files from backend
    try:
        res = requests.get(f"{BACKEND_URL}/documents")
        if res.status_code == 200:
            file_list = res.json().get("files", [])
        else:
            file_list = []
    except:
        st.error("Could not connect to Backend.")
        file_list = []

    # Count
    st.metric("Total Documents", len(file_list))

    # Selection Form
    selected_files = []
    if file_list:
        with st.form("file_selector"):
            st.write("Select files for chat context:")
            for f in file_list:
                # Checkbox for each file
                if st.checkbox(f, value=True, key=f):
                    selected_files.append(f)
            
            # Action buttons
            submitted = st.form_submit_button("Update Context") 
    else:
        st.info("No documents found. Upload one above!")

    if st.button("🔄 Refresh Library"):
        st.rerun()

# --- Sidebar: Chat Rooms ---
with st.sidebar:
    st.header("Chat Rooms")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = {'title': "New Chat", 'messages': []}
        st.session_state.current_session = new_id
        st.rerun()
    
    st.divider()
    
    for session_id, session_data in st.session_state.sessions.items():
        label = session_data['title']
        if session_id == st.session_state.current_session:
            label = f"🟢 {label}"
        if st.button(label, key=session_id, use_container_width=True):
            st.session_state.current_session = session_id
            st.rerun()

# --- Main Chat Area ---
current_id = st.session_state.current_session
current_messages = st.session_state.sessions[current_id]['messages']

st.title("📄 DocuChat RAG")
st.markdown(f"**Session Docs:** {len(selected_files)} selected")

# Display History
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input Area & Model Select ---
# Layout: [Model Selector (20%)] [Chat Input (80%)]
# Note: Streamlit's chat_input is fixed at bottom, so we place the selectbox just above it
# or use columns if we want them side-by-side (but chat_input can't be in a column easily).
# Best Approach: Put Model Selector in a clean row right above the chat input area.

col1, col2 = st.columns([1, 4])
with col1:
    selected_model = st.selectbox(
        "Select Model", 
        available_models, 
        index=0 if available_models else None,
        label_visibility="collapsed"
    )

# Chat Input
if prompt := st.chat_input("Ask about your documents..."):
    # 1. Update Title if needed
    if len(current_messages) == 0:
        short_title = prompt[:20] + "..." if len(prompt) > 20 else prompt
        st.session_state.sessions[current_id]['title'] = short_title

    # 2. Display User Message
    st.session_state.sessions[current_id]['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Stream Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            payload = {"message": prompt, "model": selected_model, "selected_files": selected_files}
            with requests.post(
                f"{BACKEND_URL}/chat_stream", 
                json=payload, 
                stream=True
            ) as r:
                if r.status_code == 200:
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            text_chunk = chunk.decode("utf-8")
                            full_response += text_chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    full_response = f"Error: {r.status_code} - {r.text}"
                    message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Connection Error: {e}"
            message_placeholder.markdown(full_response)
            
        st.session_state.sessions[current_id]['messages'].append({"role": "assistant", "content": full_response})
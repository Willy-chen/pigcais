import streamlit as st
import extra_streamlit_components as stx
import requests
import os
import time
from datetime import datetime, timedelta

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="RAG Chat", layout="wide")

# --- 1. Session State Initialization ---
if "token" not in st.session_state:
    st.session_state.token = None

if "current_session" not in st.session_state:
    st.session_state.current_session = None

if "audio_history" not in st.session_state:
    st.session_state.audio_history = []

# --- Cookie Manager for Persistence ---
cookie_manager = stx.CookieManager()

# Try to recover token from cookies if not in session state
if not st.session_state.get("token"):
    saved_token = cookie_manager.get(cookie="auth_token")
    if saved_token:
        st.session_state.token = saved_token

# --- 2. Utils ---
def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def run_audio_analysis(endpoint, files=None, json_data=None, original_filename="audio.wav"):
    """Generic function to run audio analysis and update UI/History"""
    try:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        status_text.text("Connecting to backend...")
        
        # Use a session for streaming
        with requests.post(f"{BACKEND_URL}/{endpoint}", files=files, json=json_data, stream=True) as r:
            if r.status_code == 200:
                import json
                final_data = None
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("status") == "progress":
                            val = min(1.0, max(0.0, data.get("progress", 0.0)))
                            progress_bar.progress(val)
                            status_text.text(data.get("message", ""))
                        elif data.get("status") == "success":
                            final_data = data
                
                if final_data:
                    progress_bar.empty()
                    status_text.text("Analysis Complete!")
                    
                    probs = final_data["probabilities"]
                    pred = final_data["prediction"]
                    
                    st.success(f"**Final Prediction:** {pred.title()} (Averaged over {final_data.get('segments_analyzed', 1)} segments)")
                    
                    import plotly.express as px
                    import pandas as pd
                    
                    df_probs = pd.DataFrame({
                        "Class": ["No Breathing", "Normal Breathing", "Abnormal Breathing"],
                        "Probability": [
                            probs.get("no-breathing", 0), 
                            probs.get("normal breathing", 0), 
                            probs.get("abnormal breathing", 0)
                        ]
                    })
                    
                    fig = px.pie(df_probs, values='Probability', names='Class', title='Overall Breathing Class Probabilities', color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Store audio bytes if available (from files)
                    audio_bytes = None
                    if files and 'file' in files:
                        audio_bytes = files['file'][1].getvalue()
                    
                    # Save to history
                    st.session_state.audio_history.append({
                        "filename": original_filename,
                        "prediction": pred,
                        "probabilities": probs,
                        "segments_analyzed": final_data.get("segments_analyzed", 1),
                        "detailed_segments": final_data.get("detailed_segments", []),
                        "envelope": final_data.get("envelope", []),
                        "duration": final_data.get("duration", 0.0),
                        "spectrogram": final_data.get("spectrogram", []),
                        "audio_bytes": audio_bytes
                    })
                else:
                    st.error("Analysis completed but no results were returned.")
            else:
                progress_bar.empty()
                st.error(f"Backend Error: {r.status_code} - {r.text}")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Connection Error: {str(e)}")

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
                    # Save to cookie for 30 days
                    cookie_manager.set("auth_token", token, expires_at=datetime.now() + timedelta(days=30))
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
    if st.button("🚪 Logout"):
        cookie_manager.delete("auth_token")
        st.session_state.token = None
        st.session_state.current_session = None
        st.rerun()
    
    st.divider()

    # --- Document Upload ---
    with st.expander("📂 Upload Documents"):
        uploaded_file = st.file_uploader("Select file", type=["txt", "pdf", "md", "csv", "json"])
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
    if st.button("➕ New Chat", width='stretch'):
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
            if st.button(label, key=f"s_{sid}", width='stretch'):
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

    st.title("📄 DocuChat & Audio Analysis")
    
    tab_chat, tab_audio = st.tabs(["💬 Chat", "🎙️ Audio Analysis"])
    
    with tab_chat:
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
                            
                        elif r.status_code == 404:
                             placeholder.error("Session expired. Please create a new chat.")
                        else:
                            placeholder.error(f"Error: {r.status_code}")
                except Exception as e:
                    placeholder.error(f"Connection Error: {e}")

    with tab_audio:
        st.header("Breathing Analysis Dashboard")
        st.info("Analyze pig breathing patterns using AST + XGBoost models.")
        
        # Microphone Device Info & Troubleshooting
        import streamlit.components.v1 as components
        with st.expander("🛠️ Microphone Settings & Device List"):
            st.info("Browsers usually use your **Default System Microphone**. To switch devices, use the 🔒 (Lock) icon in your browser address bar or your OS sound settings.")
            if st.button("List Available Microphones"):
                components.html("""
                    <script>
                        (async () => {
                            try {
                                // Requesting permission first ensures labels are visible
                                await navigator.mediaDevices.getUserMedia({ audio: true });
                                const devices = await navigator.mediaDevices.enumerateDevices();
                                const mics = devices.filter(d => d.kind === 'audioinput');
                                const list = mics.map((m, i) => `${i+1}. ${m.label || 'Unnamed Mic'}`).join('\\n');
                                alert("Detected Microphones:\\n" + list + "\\n\\nTo change the 'Active' mic, please update your browser settings at the top-left of the URL bar (🔒 icon).");
                            } catch (e) {
                                alert("Could not list devices: " + e.message);
                            }
                        })();
                    </script>
                """, height=0)
        
        upload_mode = st.radio("Choose Input Method", ["File Upload", "Microphone", "Cloud Storage (Firebase)"], horizontal=True)
        
        if upload_mode == "File Upload":
            audio_file = st.file_uploader("Upload Audio (WAV)", type=["wav"])
            if audio_file:
                st.audio(audio_file)
                if st.button("🔍 Analyze Uploaded File", use_container_width=True):
                    files = {"file": (audio_file.name, audio_file, "audio/wav")}
                    run_audio_analysis("analyze_audio", files=files, original_filename=audio_file.name)
        
        elif upload_mode == "Microphone":
            try:
                mic_audio = st.audio_input("Record Audio")
                if mic_audio is not None:
                    st.write(f"✅ Recording captured: {mic_audio.name} ({len(mic_audio.getvalue())} bytes)")
                    st.audio(mic_audio)
                    if st.button("🔍 Analyze Recording", use_container_width=True):
                        # Use the actual MIME type from the widget
                        files = {"file": ("recording.wav", mic_audio, mic_audio.type)}
                        run_audio_analysis("analyze_audio", files=files, original_filename="Microphone Recording")
            except Exception as e:
                st.error(f"Mic Widget Error: {str(e)}")
                    
        elif upload_mode == "Cloud Storage (Firebase)":
            fb_url = st.text_input("Enter Firebase/Direct WAV URL", placeholder="https://firebasestorage.googleapis.com/...")
            if fb_url:
                if st.button("🔍 Analyze Remote File", use_container_width=True):
                    run_audio_analysis("analyze_url", json_data={"url": fb_url}, original_filename=f"Cloud: {fb_url.split('/')[-1].split('?')[0]}")
                    
        # History UI
        if st.session_state.audio_history:
            st.divider()
            st.header("🕒 Prediction History")
            for idx, record in enumerate(reversed(st.session_state.audio_history)):
                with st.expander(f"📁 **{record['filename']}**  |  {record['prediction'].title()}"):
                    import plotly.express as px
                    import pandas as pd
                    
                    hist_probs = record['probabilities']
                    df_probs = pd.DataFrame({
                        "Class": ["No Breathing", "Normal Breathing", "Abnormal Breathing"],
                        "Probability": [
                            hist_probs.get("no-breathing", 0), 
                            hist_probs.get("normal breathing", 0), 
                            hist_probs.get("abnormal breathing", 0)
                        ]
                    })
                    
                    fig = px.pie(df_probs, values='Probability', names='Class', title='Overall Probabilities', color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig, width='stretch', key=f"pie_{idx}")
                    
                    if record.get("audio_bytes"):
                        st.audio(record["audio_bytes"], format="audio/wav")
                    
                    if record.get('detailed_segments'):
                        view_mode = st.radio("View Mode", ["Waveform", "Spectrogram"], key=f"view_{idx}", horizontal=True)
                        
                        import plotly.graph_objects as go
                        import numpy as np
                        
                        duration = record['duration']
                        
                        if view_mode == "Waveform" and record.get('envelope'):
                            envelope = record['envelope']
                            time_axis = [i * duration / max(1, len(envelope)-1) for i in range(len(envelope))]
                            fig_main = go.Figure()
                            fig_main.add_trace(go.Scatter(
                                x=time_axis, y=envelope, mode='lines', 
                                name='Waveform', line=dict(color='gray', width=1.5),
                                showlegend=False, hoverinfo='skip'
                            ))
                            max_val = max(envelope) if envelope else 1.0
                        elif view_mode == "Spectrogram" and record.get('spectrogram'):
                            spectrogram = np.array(record['spectrogram'])
                            fig_main = go.Figure()
                            fig_main.add_trace(go.Heatmap(
                                z=spectrogram,
                                x=np.linspace(0, duration, spectrogram.shape[1]),
                                y=np.linspace(0, 8000, spectrogram.shape[0]), # 16kHz SR -> 8kHz Max Freq
                                colorscale='Viridis',
                                showscale=False,
                                name='Spectrogram'
                            ))
                            max_val = 8000
                        else:
                            st.warning("No visualization data available.")
                            continue

                        # Overlay colored semi-transparent blocks (Sync with hover)
                        for i, seg in enumerate(record['detailed_segments']):
                            p = seg['probabilities']
                            if p.get('abnormal breathing', 0) >= 0.25:
                                color = 'rgba(255, 0, 0, 0.4)'
                                label = 'Abnormal Breathing'
                            elif p.get('normal breathing', 0) >= 0.70:
                                color = 'rgba(0, 255, 0, 0.4)'
                                label = 'Normal Breathing'
                            else:
                                color = 'rgba(100, 100, 100, 0.2)'
                                label = 'No Breathing'
                                
                            hover_text = (
                                f"<b>{label}</b><br>"
                                f"Time: {format_seconds(seg['start_time'])} - {format_seconds(seg['end_time'])}<br>"
                                f"Normal: {p.get('normal breathing',0):.1%}<br>"
                                f"Abnormal: {p.get('abnormal breathing',0):.1%}<br>"
                                f"None: {p.get('no-breathing',0):.1%}"
                            )
                            
                            # Range should match the plot's Y axis
                            y_range = [0, max_val]
                            fig_main.add_trace(go.Scatter(
                                x=[seg['start_time'], seg['end_time'], seg['end_time'], seg['start_time']],
                                y=[y_range[0], y_range[0], y_range[1], y_range[1]],
                                fill='toself', fillcolor=color, line=dict(width=0),
                                name=label, hoverinfo='text', text=hover_text,
                                showlegend=False
                            ))
                            
                        # Custom X-axis ticks for HH:MM:SS
                        num_ticks = 5
                        tick_vals = [i * duration / (num_ticks - 1) for i in range(num_ticks)]
                        tick_text = [format_seconds(v) for v in tick_vals]

                        fig_main.update_layout(
                            xaxis_title="Time (HH:MM:SS)",
                            yaxis_title="Amplitude" if view_mode == "Waveform" else "Frequency (Hz)",
                            hovermode="closest",
                            xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text),
                            yaxis=dict(range=[0, max_val * 1.05]),
                            margin=dict(l=0, r=0, t=10, b=10)
                        )
                        st.plotly_chart(fig_main, width='stretch', key=f"graph_{idx}")
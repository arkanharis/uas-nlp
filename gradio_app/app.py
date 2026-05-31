import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
import base64
from datetime import datetime

API_BASE_URL = "http://localhost:8000"
HEALTH_CHECK_TIMEOUT = 5
VOICE_CHAT_TIMEOUT = 900  # 15 menit untuk pipeline lengkap (STT: max 300s + LLM: 30s + TTS: 60s + overhead)

def check_server_health():
    """
    Check apakah FastAPI server dan semua services ready
    Returns: status_display (simple format: "STT ✓ | LLM ✓ | TTS ✓")
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            services = data.get('services', {})
            
            # Extract status untuk setiap service
            stt_status = "✅" if services.get('stt', {}).get('status') == 'ready' else "❌"
            llm_status = "✅" if services.get('llm', {}).get('status') == 'ready' else "❌"
            tts_status = "✅" if services.get('tts', {}).get('status') == 'ready' else "❌"
            prep_status = "✅" if services.get('preprocessing', {}).get('status') == 'ready' else "❌"
            
            # Format kompak
            status_display = f"STT {stt_status} | LLM {llm_status} | TTS {tts_status} | Preprocessing {prep_status}"
            return status_display
        else:
            return f"Server Error (Status: {response.status_code})"
    
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to server"
    except requests.exceptions.Timeout:
        return "⏱️ Server timeout"
    except Exception as e:
        return f"❌ Error: {type(e).__name__}"

def voice_chat(audio, mode="preserve"):
    """
    Process audio through voice chat pipeline
    
    Args:
        audio: Audio tuple (sr, audio_data)
        mode: "preserve" | "normalized" | "translate"
    
    Returns:
        tuple: (output_audio_path, transcript, llm_response, latency_info)
    """
    if audio is None:
        return None, "❌ No audio provided", "", "N/A"
    
    sr, audio_data = audio

    # Simpan sebagai .wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    try:
        # Kirim ke endpoint FastAPI dengan mode parameter
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            params = {"mode": mode}
            response = requests.post(
                f"{API_BASE_URL}/voice-chat",
                files=files,
                params=params,
                timeout=VOICE_CHAT_TIMEOUT
            )

        if response.status_code == 200:
            data = response.json()
            
            # Ambil audio_base64 dari response (backend encode sebagai base64)
            audio_base64 = data.get("audio_base64", None)
            
            if not audio_base64:
                return None, "❌ Audio file not generated", "", "N/A"
            
            # Decode base64 ke binary dan save ke temp file
            try:
                audio_bytes = base64.b64decode(audio_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    tmpfile.write(audio_bytes)
                    audio_file_path = tmpfile.name
                print(f"[GRADIO] ✅ Audio saved to {audio_file_path} ({len(audio_bytes)} bytes)")
            except Exception as e:
                return None, f"❌ Failed to decode audio: {str(e)}", "", "N/A"
            
            # Extract informasi
            transcript = data.get("transcript", "N/A")
            llm_response = data.get("llm_response", "N/A")
            preprocessing = data.get("preprocessing", {})
            latency = data.get("latency", {})
            
            # Format latency info
            latency_text = f"""
**Latency Breakdown:**
- STT: {latency.get('stt_ms', 0):.2f}ms
- LLM: {latency.get('llm_ms', 0):.2f}ms
- TTS: {latency.get('tts_ms', 0):.2f}ms
- Total: {latency.get('total_ms', 0):.2f}ms ({latency.get('total_ms', 0)/1000:.2f}s)
            """
            
            # Format preprocessing info
            preprocessing_text = f"""
**Preprocessing Analysis:**
- Detected Languages: {', '.join(preprocessing.get('detected_languages', []))}
- Code-Switch Segments: {preprocessing.get('code_switch_count', 0)}
- Mode Applied: {mode}
            """
            
            # Format transcript dengan preprocessing info
            transcript_full = f"""
**Original Transcript:**
{transcript}

{preprocessing_text}
            """
            
            return audio_file_path, transcript_full, llm_response, latency_text
        
        elif response.status_code == 500:
            error_data = response.json()
            error_msg = error_data.get("detail", "Server error")
            return None, f"❌ Server Error: {error_msg}", "", "N/A"
        else:
            return None, f"❌ Error {response.status_code}", "", "N/A"
    
    except requests.exceptions.Timeout:
        return None, "❌ Request Timeout - Pipeline took too long", "", "N/A"
    except requests.exceptions.ConnectionError:
        return None, "❌ Connection Error - Cannot reach server", "", "N/A"
    except Exception as e:
        return None, f"❌ Error: {str(e)}", "", "N/A"

# ===== GRADIO UI =====
with gr.Blocks(title="🎙️ Multilingual Speech-to-Speech Chatbot", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("# 🎙️ Multilingual Speech-to-Speech Chatbot")
    gr.Markdown("*Supports: Indonesian 🇮🇩 | English 🇬🇧 | Arabic 🇸🇦 | Code-Switching*")
    
    # Health Check Section
    gr.Markdown("## 📊 System Status")
    with gr.Row():
        server_status = gr.Textbox(label="Server", interactive=False, scale=2)
        check_btn = gr.Button("🔄 Check", scale=1)
    
    # Main Chat Section
    gr.Markdown("## 💬 Voice Chat")
    with gr.Row():
        with gr.Column():
            # Mode selection
            mode_select = gr.Radio(
                choices=["preserve", "normalized", "translate"],
                value="preserve",
                label="📝 Processing Mode",            
            )
            
            # Audio input
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                label="🎤 Record Your Question",
                interactive=True
            )
            
            # Submit button
            submit_btn = gr.Button("🚀 Submit & Process", variant="primary", size="lg")
        
        with gr.Column():
            # Audio output
            audio_output = gr.Audio(
                type="filepath",
                label="🔊 Bot Response (Audio)",
                interactive=False
            )
    
    # Results Section
    gr.Markdown("## 📋 Results")
    with gr.Tabs():
        # Tab 1: Transcript & Preprocessing
        with gr.Tab(label="📝 Transcript & Analysis"):
            transcript_display = gr.Textbox(
                label="Transcript with Preprocessing Analysis",
                interactive=False,
                lines=8
            )
        
        # Tab 2: LLM Response
        with gr.Tab(label="🤖 LLM Response"):
            llm_response_display = gr.Textbox(
                label="Model Response",
                interactive=False,
                lines=8
            )
        
        # Tab 3: Latency
        with gr.Tab(label="⏱️ Latency"):
            latency_display = gr.Textbox(
                label="Latency Breakdown",
                interactive=False,
                lines=8
            )
    
    
    # Event handlers
    def update_health_status():
        """Update health check info"""
        status_display = check_server_health()
        return status_display
    
    check_btn.click(
        fn=update_health_status,
        outputs=[server_status]
    )
    
    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, mode_select],
        outputs=[audio_output, transcript_display, llm_response_display, latency_display]
    )
    
    # Auto health check on load
    demo.load(update_health_status, outputs=[server_status])

demo.launch(share=False)


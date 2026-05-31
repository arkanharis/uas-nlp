"""
FastAPI server untuk menghubungkan STT, LLM, dan TTS dalam satu aplikasi voice chat.
"""

import tempfile
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
import json

from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech
from app.preprocessing import preprocess_text

# Inisialisasi FastAPI app
app = FastAPI(
    title="Voice Chat API",
    description="API untuk voice chat dengan STT, LLM, dan TTS",
    version="1.0.0"
)

# Konfigurasi CORS untuk mengizinkan request dari aplikasi Gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Health check endpoint untuk verifikasi server berjalan
    """
    return JSONResponse(content={
        "status": "ok",
        "message": "Voice Chat API is running",
        "endpoints": {
            "voice_chat": "/voice-chat",
            "health": "/health"
        }
    })

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Comprehensive health check endpoint untuk verifikasi semua services
    """
    import time
    from datetime import datetime
    
    try:
        # Check STT availability
        stt_status = "ready"
        try:
            from app.stt import WHISPER_BINARY, WHISPER_MODEL_PATH
            if not os.path.exists(WHISPER_BINARY) or not os.path.exists(WHISPER_MODEL_PATH):
                stt_status = "missing_files"
        except:
            stt_status = "error"
        
        # Check LLM availability
        llm_status = "ready"
        try:
            from app.llm import GOOGLE_API_KEY, MODEL
            if not GOOGLE_API_KEY:
                llm_status = "missing_key"
        except:
            llm_status = "error"
        
        # Check TTS availability
        tts_status = "ready"
        try:
            from app.tts import COQUI_MODEL_PATH, COQUI_CONFIG_PATH, COQUI_SPEAKERS_PATH
            missing_files = []
            if not os.path.exists(COQUI_MODEL_PATH):
                missing_files.append("model")
            if not os.path.exists(COQUI_CONFIG_PATH):
                missing_files.append("config")
            if not os.path.exists(COQUI_SPEAKERS_PATH):
                missing_files.append("speakers")
            
            if missing_files:
                tts_status = f"missing_files ({', '.join(missing_files)})"
        except:
            tts_status = "error"
        
        # Check Preprocessing availability
        preprocessing_status = "ready"
        try:
            from app.preprocessing import preprocess_text
        except:
            preprocessing_status = "error"
        
        return JSONResponse(content={
            "status": "operational" if all(s == "ready" for s in [stt_status, llm_status, tts_status, preprocessing_status]) else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "stt": {
                    "status": stt_status,
                    "model": "whisper.cpp (ggml-base.bin)",
                    "engine": "Whisper"
                },
                "llm": {
                    "status": llm_status,
                    "model": "gemma-4-31b-it",
                    "engine": "Google Gemini API"
                },
                "tts": {
                    "status": tts_status,
                    "model": "Coqui TTS (Indonesian)",
                    "engine": "Coqui"
                },
                "preprocessing": {
                    "status": preprocessing_status,
                    "model": "langdetect + regex patterns",
                    "engine": "Custom"
                }
            },
            "capabilities": {
                "multilingual": True,
                "code_switching": True,
                "modes": ["preserve", "normalized", "translate"]
            },
            "message": "All services operational" if stt_status == llm_status == tts_status == preprocessing_status == "ready" else "Some services may have issues"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "message": str(e),
                "services": {
                    "stt": {"status": "unknown"},
                    "llm": {"status": "unknown"},
                    "tts": {"status": "unknown"},
                    "preprocessing": {"status": "unknown"}
                }
            }
        )

@app.post("/voice-chat", tags=["Voice Chat"])
async def voice_chat(
    file: UploadFile = File(...),
    mode: str = "preserve"
):
    """
    Endpoint utama untuk voice chat dengan preprocessing:
    1. Menerima file audio dari user (STT)
    2. [NEW] Preprocessing: deteksi bahasa, code-switching detection
    3. Mengirim teks ke LLM untuk mendapatkan respons
    4. Mengkonversi respons teks ke audio (TTS)
    5. Mengembalikan file audio hasil TTS
    
    Args:
        file (UploadFile): File audio dari user (format .wav, .mp3, dll)
        mode (str): Mode preprocessing
            - 'preserve' (default): Pertahankan pola code-switching asli
            - 'normalized': Normalisasi ke satu bahasa
            - 'translate': Terjemahkan (opsional)
    
    Returns:
        dict: JSON response dengan audio, transcript, LLM response, dan metrics
    """
    import time
    
    try:
        start_time = time.time()
        
        # Baca file audio yang diunggah
        audio_bytes = await file.read()
        
        # Dapatkan ekstensi file
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        
        # Step 1: Transkripsi audio ke teks menggunakan STT
        print(f"[STT] Transcribing audio file: {file.filename}")
        stt_start = time.time()
        user_text = transcribe_speech_to_text(audio_bytes, file_ext)
        stt_latency = (time.time() - stt_start) * 1000
        
        if user_text.startswith("[ERROR]"):
            return {
                "error": "STT failed",
                "message": user_text,
                "latency": {"stt_ms": stt_latency}
            }
        
        print(f"[STT] Transcribed text: {user_text}")
        
        # Step 2: [NEW] Preprocessing - Language detection & code-switching
        print(f"[PREPROCESSING] Analyzing: {user_text[:50]}...")
        preprocess_start = time.time()
        preprocessing_result = preprocess_text(user_text, mode=mode)
        preprocess_latency = (time.time() - preprocess_start) * 1000
        
        print(f"[PREPROCESSING] Detected language: {preprocessing_result.detected_language}")
        print(f"[PREPROCESSING] Has code-switching: {preprocessing_result.has_code_switching}")
        print(f"[PREPROCESSING] Language distribution: {preprocessing_result.language_distribution}")
        
        # Dapatkan teks sesuai mode untuk dikirim ke LLM
        text_for_llm = preprocessing_result.normalized if mode == "normalized" else user_text
        
        # Step 3: Generate respons menggunakan LLM
        print(f"[LLM] Generating response for: {text_for_llm[:50]}...")
        llm_start = time.time()
        llm_response = generate_response(text_for_llm)
        llm_latency = (time.time() - llm_start) * 1000
        
        if llm_response.startswith("[ERROR]"):
            return {
                "error": "LLM failed",
                "message": llm_response,
                "transcript": user_text,
                "latency": {
                    "stt_ms": round(stt_latency, 2),
                    "llm_ms": round(llm_latency, 2)
                }
            }
        
        print(f"[LLM] Response: {llm_response[:50]}...")
        
        # Step 4: Konversi respons teks ke audio menggunakan TTS
        print(f"[TTS] Converting response to speech")
        tts_start = time.time()
        audio_file_path = transcribe_text_to_speech(llm_response)
        tts_latency = (time.time() - tts_start) * 1000
        
        if audio_file_path.startswith("[ERROR]"):
            return {
                "error": "TTS failed",
                "message": audio_file_path,
                "transcript": user_text,
                "llm_response": llm_response,
                "latency": {
                    "stt_ms": round(stt_latency, 2),
                    "llm_ms": round(llm_latency, 2),
                    "tts_ms": round(tts_latency, 2)
                }
            }
        
        print(f"[TTS] Audio file saved to: {audio_file_path}")
        
        # Read audio file dan encode ke base64
        audio_base64 = None
        try:
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            print(f"[TTS] ✅ Audio encoded to base64 ({len(audio_base64)} chars)")
        except Exception as e:
            print(f"[TTS] ⚠️ Failed to encode audio: {str(e)}")
        
        total_latency = (time.time() - start_time) * 1000
        
        # Return JSON response dengan semua informasi
        response_data = {
            "status": "success",
            "transcript": user_text,
            "llm_response": llm_response,
            "audio_file": audio_file_path,
            "audio_base64": audio_base64,
            "preprocessing": {
                "detected_languages": preprocessing_result.language_distribution,
                "code_switch_count": len([s for s in preprocessing_result.segments if s.language != preprocessing_result.detected_language]),
                "mode": mode
            },
            "latency": {
                "stt_ms": round(stt_latency, 2),
                "llm_ms": round(llm_latency, 2),
                "tts_ms": round(tts_latency, 2),
                "total_ms": round(total_latency, 2)
            }
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        print(f"[ERROR] Voice chat endpoint failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Unexpected error",
                "message": str(e)
            }
        )

@app.post("/transcribe", tags=["STT"])
async def transcribe(file: UploadFile = File(...)):
    """
    Endpoint untuk transcripsi audio ke teks (STT only)
    
    Args:
        file (UploadFile): File audio dari user
    
    Returns:
        dict: Teks hasil transkripsi
    """
    try:
        audio_bytes = await file.read()
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        text = transcribe_speech_to_text(audio_bytes, file_ext)
        
        if text.startswith("[ERROR]"):
            return JSONResponse(
                status_code=400,
                content={"error": text}
            )
        
        return JSONResponse(content={"transcription": text})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/chat", tags=["LLM"])
async def chat(prompt: str):
    """
    Endpoint untuk chat dengan LLM (text to text)
    
    Args:
        prompt (str): Pertanyaan atau pesan dari user
    
    Returns:
        dict: Respons dari LLM
    """
    try:
        response = generate_response(prompt)
        
        if response.startswith("[ERROR]"):
            return JSONResponse(
                status_code=400,
                content={"error": response}
            )
        
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/synthesize", tags=["TTS"])
async def synthesize(text: str):
    """
    Endpoint untuk mengkonversi teks ke audio (TTS only)
    
    Args:
        text (str): Teks yang akan dikonversi ke audio
    
    Returns:
        FileResponse: File audio hasil TTS
    """
    try:
        audio_file_path = transcribe_text_to_speech(text)
        
        if audio_file_path.startswith("[ERROR]"):
            return JSONResponse(
                status_code=400,
                content={"error": audio_file_path}
            )
        
        return FileResponse(
            path=audio_file_path,
            media_type="audio/wav",
            filename="synthesis.wav"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/preprocess", tags=["Preprocessing"])
async def preprocess(text: str, mode: str = "preserve"):
    """
    Endpoint untuk analisis preprocessing: deteksi bahasa dan code-switching
    
    Args:
        text (str): Teks untuk dianalisis
        mode (str): Mode preprocessing
            - 'preserve' (default): Pertahankan pola code-switching
            - 'normalized': Normalisasi ke bahasa dominan
            - 'translate': Persiapkan untuk translasi
    
    Returns:
        dict: Hasil analisis preprocessing dengan metadata lengkap
    """
    try:
        result = preprocess_text(text, mode=mode)
        
        if result.error:
            return JSONResponse(
                status_code=400,
                content={"error": result.error}
            )
        
        return JSONResponse(content={
            "original": result.original,
            "normalized": result.normalized,
            "preserve": result.preserve,
            "detected_language": result.detected_language,
            "has_code_switching": result.has_code_switching,
            "language_distribution": result.language_distribution,
            "segments": [
                {
                    "text": seg.text,
                    "language": seg.language,
                    "confidence": seg.confidence,
                    "position": f"{seg.start_pos}-{seg.end_pos}"
                }
                for seg in result.segments
            ],
            "mode": result.mode
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    
    # Jalankan server FastAPI
    # Akses dokumentasi API di http://localhost:8000/docs
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

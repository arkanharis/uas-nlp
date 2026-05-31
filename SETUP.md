# 🎙️ Voice Chatbot - Panduan Menjalankan Aplikasi

Panduan lengkap untuk menjalankan aplikasi Voice Chatbot dengan STT, LLM, dan TTS.

## 📋 Prasyarat

Pastikan Anda sudah memiliki:
- Python 3.8+
- Virtual environment sudah diaktifkan
- Semua dependencies di `requirements.txt` sudah terinstall
- API Key Gemini sudah tersimpan di file `.env`

## 🚀 Cara Menjalankan Aplikasi

### 1. Aktifkan Virtual Environment

**Windows (PowerShell):**
```powershell
.\env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\env\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source env/bin/activate
```

### 2. Jalankan FastAPI Server

Buka terminal dan navigasi ke folder `app`:
```bash
cd app
python main.py
```

Server akan berjalan di `http://localhost:8000`

**Dokumentasi API:** Buka browser dan akses `http://localhost:8000/docs`

### 3. Jalankan Aplikasi Gradio (Di Terminal Baru)

Buka terminal baru, aktifkan virtual environment, kemudian:
```bash
cd gradio_app
python app.py
```

Aplikasi akan membuka di browser secara otomatis, biasanya di `http://127.0.0.1:7860`

## 🔧 Konfigurasi

### File `.env`
Pastikan file `.env` di root project berisi:
```
GEMINI_API_KEY=your_api_key_here
```

### Model Files
Pastikan sudah tersedia:
- **STT Model:** `app/whisper.cpp/models/ggml-large-v3-turbo.bin`
- **TTS Model:** `app/coqui_utils/checkpoint_1260000-inference.pth`
- **TTS Config:** `app/coqui_utils/config.json`
- **TTS Speakers:** `app/coqui_utils/speakers.pth`

## 📡 API Endpoints

### 1. **Voice Chat** (Main Endpoint)
```
POST /voice-chat
Content-Type: multipart/form-data

Body:
- file: <audio_file> (format: .wav, .mp3, dll)

Returns: Audio file (response.wav)
```

### 2. **Transcribe Only** (STT)
```
POST /transcribe
Content-Type: multipart/form-data

Body:
- file: <audio_file>

Returns: {"transcription": "teks hasil transkripsi"}
```

### 3. **Chat Only** (LLM)
```
POST /chat
Content-Type: application/x-www-form-urlencoded

Body:
- prompt: <text_query>

Returns: {"response": "respons dari LLM"}
```

### 4. **Synthesize Only** (TTS)
```
POST /synthesize
Content-Type: application/x-www-form-urlencoded

Body:
- text: <text_to_convert>

Returns: Audio file (synthesis.wav)
```

## 🐛 Troubleshooting

### Error: "GOOGLE_API_KEY not found"
- Pastikan file `.env` ada di root project
- Verifikasi format: `GEMINI_API_KEY=your_key_here`
- Restart FastAPI server setelah update `.env`

### Error: "Whisper binary not found"
- Pastikan `app/whisper.cpp/build/bin/whisper` ada
- Jika tidak ada, compile whisper.cpp terlebih dahulu atau download binary

### Error: "Model file not found"
- Verifikasi path di `app/stt.py` dan `app/tts.py`
- Download model files yang hilang dari:
  - STT: https://huggingface.co/ggerganov/whisper.cpp
  - TTS: Coqui TTS models

### Error: "tts command not found"
- Install Coqui TTS: `pip install coqui-tts`
- Verifikasi instalasi: `which tts` (Linux/Mac) atau `where tts` (Windows)

### Koneksi antar aplikasi gagal
- Pastikan FastAPI server sudah running di `localhost:8000`
- Cek CORS settings di `app/main.py`
- Jika masih error, disable browser cache atau gunakan incognito mode

## 📝 Logging

Aplikasi akan menampilkan log di terminal:
- `[STT]` - Pesan dari Speech-to-Text
- `[LLM]` - Pesan dari Language Model
- `[TTS]` - Pesan dari Text-to-Speech
- `[ERROR]` - Pesan error

## 📊 Alur Kerja

```
User Input (Audio) → STT → Text → LLM → Response → TTS → Audio Output
     ↓                     ↓         ↓      ↓        ↓        ↓
  Gradio UI            Whisper   Google   Coqui   Output
                       (Gemini)
```

## 🎯 Tips

1. **Pertama kali jalankan:** Aplikasi mungkin memerlukan waktu untuk load models
2. **Chat history:** History disimpan otomatis di `app/chat_history.json`
3. **Testing API:** Gunakan Swagger UI di `http://localhost:8000/docs`
4. **Performance:** TTS memerlukan resource yang cukup; gunakan GPU jika tersedia

## ❓ Support

Jika mengalami masalah:
1. Baca log output di terminal
2. Verifikasi semua file konfigurasi
3. Pastikan semua dependencies terinstall
4. Coba restart aplikasi dari awal

---

**Version:** 1.0.0  
**Last Updated:** 2026-06-19

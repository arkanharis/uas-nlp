# 🎙️ Multilingual Speech-to-Speech Chatbot with Code-Switching Support

**Proyek UAS Pemrosesan Bahasa Alami** — Aplikasi chatbot berbasis suara end-to-end yang mendukung **Indonesia 🇮🇩 | English 🇬🇧 | Arabic 🇸🇦** dengan deteksi code-switching otomatis.

## 📌 Fitur Utama

### 🎯 Core Pipeline
- **🎙️ Speech-to-Text (STT)**: Menggunakan `whisper.cpp` dengan model `ggml-base.bin` dari OpenAI untuk transkripsi real-time
- **🧠 Large Language Model (LLM)**: Google Gemini API (`gemma-4-31b-it`) untuk generating intelligent responses
- **🔊 Text-to-Speech (TTS)**: Coqui TTS v0.26.0 dengan G2P (text-to-phoneme) untuk output audio natural

### 🌐 Preprocessing & Language Detection
- **Language Detection**: Deteksi otomatis bahasa input (Indonesian, English, Arabic)
- **Code-Switching Analysis**: Identifikasi dan analisis sentence dengan pencampuran bahasa
- **Language Distribution**: Breakdown persentase setiap bahasa dalam text
- **Three Processing Modes**:
  - `preserve`: Pertahankan pola code-switching asli
  - `normalized`: Normalisasi ke bahasa dominan
  - `translate`: Persiapkan untuk translasi (optional)

### 💻 User Interface & Monitoring
- **Gradio UI** (port 7860): Interface interaktif dengan mode selector dan detailed results
- **System Health Check**: Real-time monitoring status STT, LLM, TTS, Preprocessing
- **Result Tabs**: 
  - Transcript & Preprocessing Analysis
  - LLM Response
  - Performance Metrics (latency breakdown)

### 📊 Evaluation Pipeline
- **STT Metrics**: WER (Word Error Rate), CER (Character Error Rate), Similarity Score
- **LLM Metrics**: Response length, word count
- **TTS Metrics**: Audio duration, quality assessment
- **Latency Tracking**: Per-stage timing (STT, LLM, TTS, Total)
- **End-to-End Performance**: Corpus-based batch evaluation

## 🗂️ Struktur Proyek

```
UAS-Praktikum-Pemrosesan-Bahasa-Alami/
│
├── app/                           # Backend FastAPI Server
│   ├── main.py                   # REST API endpoints (FastAPI)
│   ├── stt.py                    # Speech-to-Text (whisper.cpp integration)
│   ├── llm.py                    # LLM (Gemini API)
│   ├── tts.py                    # Text-to-Speech (Coqui TTS + G2P)
│   ├── preprocessing.py          # Language detection & code-switching analysis
│   ├── whisper.cpp/              # whisper.cpp binary & models
│   │   ├── build/
│   │   │   └── bin/whisper-cli.exe
│   │   └── ggml-base.bin
│   └── coqui_utils/              # Coqui TTS models & config
│       ├── checkpoint_1260000-inference.pth
│       ├── config.json
│       └── speakers.pth
│
├── gradio_app/
│   └── app.py                    # Frontend UI (Gradio)
│
├── data/
│   └── corpus/
│       ├── audio/                # Audio files untuk evaluation
│       └── transcript/
│           └── reference.csv     # Ground truth transcripts
│
├── analysis_results/             # Output dari evaluation pipeline
│   └── evaluation_results_*.json
│
├── analisis_pipeline.py          # Evaluation & benchmark script
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (GEMINI_API_KEY)
└── README.md                     # Dokumentasi ini
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows (tested on Windows 10/11)
- Git

### Installation

1. **Clone & Setup**
   ```bash
   cd c:\1.Project\UAS-Praktikum-Pemrosesan-Bahasa-Alami
   python -m venv env
   .\env\Scripts\Activate.ps1
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**
   ```bash
   # Create .env file in project root
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application

**Terminal 1: Start FastAPI Backend** (port 8000)
```bash
python -m app.main
```

**Terminal 2: Start Gradio UI** (port 7860)
```bash
python gradio_app/app.py
```

**Access**:
- Gradio UI: http://127.0.0.1:7860
- FastAPI Docs: http://127.0.0.1:8000/docs
- FastAPI Health: http://127.0.0.1:8000/health

## 📡 API Endpoints

### Health Check
```http
GET /health
```
Returns system status for all components (STT, LLM, TTS, Preprocessing)

### Voice Chat (Main Endpoint)
```http
POST /voice-chat?mode=preserve
Content-Type: multipart/form-data

file: <audio.wav>
mode: preserve|normalized|translate
```

### Individual Components
```http
POST /transcribe              # STT only
POST /chat                    # LLM only
POST /synthesize              # TTS only
POST /preprocess              # Preprocessing analysis only
```

## 📊 Evaluation Pipeline

Jalankan corpus evaluation:

```bash
# Process first 5 files
python analisis_pipeline.py --limit 5

# Process all corpus
python analisis_pipeline.py --all

# Custom output
python analisis_pipeline.py --limit 10 --output results/eval.json
```

**Output Metrics**:
- WER (Word Error Rate) untuk STT
- CER (Character Error Rate) untuk STT
- Latency breakdown per stage
- Language distribution & code-switching ratio
- Summary statistics

## 🔧 Configuration

### Model Selection
| Component | Model | Location |
|-----------|-------|----------|
| STT | whisper.cpp ggml-base.bin | `app/whisper.cpp/` |
| LLM | Gemini `gemma-4-31b-it` | Google Cloud API |
| TTS | Coqui TTS (Indonesian) | `app/coqui_utils/` |
| Speaker | wibowo | `speakers.pth` |

### Supported Languages
- 🇮🇩 Indonesian (id)
- 🇬🇧 English (en)
- 🇸🇦 Arabic (ar)

## 📚 Technical Details

### Speech-to-Text
- **Engine**: whisper.cpp (C++ implementation of Whisper)
- **Model**: ggml-base.bin (~140MB)
- **Output**: UTF-8 text

### Language Model
- **Provider**: Google Gemini API
- **Model**: gemma-4-31b-it
- **Input**: Preprocessed transcript (max 500 chars)
- **Output**: Indonesian language response

### Text-to-Speech
- **Engine**: Coqui TTS v0.26.0
- **Language**: Indonesian
- **Speaker**: wibowo (multi-speaker model)
- **G2P Module**: g2p_id (converts text to phonemes)
- **Output**: WAV format (22050 Hz)

### Preprocessing Pipeline
- **Language Detection**: langdetect + regex patterns
- **Code-Switching**: Segment-level language identification
- **Normalization**: Optional single-language extraction

## 🧪 Testing

### Manual Testing via Gradio
1. Open http://127.0.0.1:7860
2. Click "Check System Health" to verify all components
3. Select processing mode
4. Record audio or upload WAV file
5. View results in tabs

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Test endpoints
curl -X POST -F "file=@test_audio.wav" -F "mode=preserve" \
  http://localhost:8000/voice-chat
```

## 📈 Performance Metrics

Typical latencies (on test system):
- STT: 2-5 seconds (depends on audio length)
- LLM: 3-8 seconds (API response time)
- TTS: 1-3 seconds (output duration dependent)
- **Total**: 6-16 seconds per interaction

## 🛠️ Troubleshooting

### Issue: G2P ONNX Error
```
[WARNING] G2P conversion failed (ONNX error): Unexpected input data type
```
**Solution**: Update ONNX Runtime
```bash
pip install --upgrade onnxruntime g2p-id
```

### Issue: Whisper Model Not Found
**Solution**: Download ggml-base.bin to `app/whisper.cpp/`
```bash
# Manual download or use whisper.cpp scripts
```

### Issue: API Key Error
**Solution**: Verify `.env` file with valid GEMINI_API_KEY

## 👨‍💻 Teknologi Stack

- **Backend**: FastAPI, Python 3.8+
- **Frontend**: Gradio
- **STT**: whisper.cpp
- **LLM**: Google Gemini API
- **TTS**: Coqui TTS, g2p_id
- **Preprocessing**: langdetect, regex
- **Evaluation**: jiwer (WER/CER metrics)
- **Data Format**: WAV audio, JSON results

## 📝 Catatan Penting

- Format audio default: `.wav` dengan sample rate 16-48 kHz
- Semua text dikonversi ke UTF-8
- G2P conversion menghasilkan phoneme untuk akurasi TTS lebih baik
- Code-switching detection berfungsi multi-language
- Evaluation pipeline memerlukan reference.csv dengan format: `utterance_id, ground_truth`

## 👥 Kontribusi & Credits

**Dibuat untuk**: Mata Kuliah *Pemrosesan Bahasa Alami* (NLP)  
**Semester**: Genap 2024/2025  
**Status**: Production Ready ✅

## 📄 License

Proyek ini menggunakan model dan library open-source sesuai lisensi masing-masing:
- Whisper: OpenAI (MIT License)
- Coqui TTS: Open-source (MPL-2.0)
- FastAPI: Starlette Foundation (BSD)
- Gradio: Hugging Face (Apache 2.0)

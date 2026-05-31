import os
import uuid
import tempfile
import subprocess
import wave

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path ke folder utilitas STT
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")

# TODO: Lengkapi path ke binary whisper-cli
# Gunakan os.path.join() untuk menggabungkan WHISPER_DIR, "build", "bin", dan "whisper-cli"
WHISPER_BINARY = os.path.join(WHISPER_DIR, "build", "bin", "whisper-cli.exe")

# TODO: Lengkapi path ke file model Whisper (contoh: ggml-base.bin)
# Gunakan os.path.join() untuk mengarah ke file model di dalam folder whisper.cpp
# Try larger models for better accuracy: ggml-small.bin, ggml-medium.bin, ggml-large-v3.bin
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "ggml-small.bin")

# Whisper parameters untuk optimization
# Language: "auto" (auto-detect), "id" (Indonesian), "en" (English), "ar" (Arabic)
# PENTING: Jika auto-detect salah, force dengan language code
WHISPER_LANGUAGE = "id"  # Force Indonesian untuk dataset ini (majority Indonesian)
WHISPER_TEMPERATURE = 0.0  # Set ke 0 untuk deterministic results (consistent)
WHISPER_THREADS = 4  # Number of threads untuk processing

# Supported languages for this project
SUPPORTED_LANGUAGES = {
    "id": "Indonesian",
    "en": "English", 
    "ar": "Arabic",
    "auto": "Auto-detect"
}

# Model selection untuk accuracy vs speed tradeoff
# Gunakan: ggml-base.bin (fast, low accuracy)
#          ggml-small.bin (medium speed, better accuracy)
#          ggml-medium.bin (slow, high accuracy) 
#          ggml-large-v3.bin (very slow, best accuracy)
# Note: Larger models memerlukan lebih banyak RAM dan waktu processing
MODEL_TYPE = "small"  # Change to "small", "medium", atau "large" untuk accuracy lebih baik

def _preprocess_audio(audio_path: str) -> str:
    """
    Preprocess audio untuk optimal whisper accuracy:
    - Convert stereo to mono (whisper works better with mono)
    - Resample to 16kHz if needed (optimal sample rate for whisper)
    
    Args:
        audio_path: Path ke original audio file
    
    Returns:
        str: Path ke preprocessed audio file
    """
    try:
        import wave
        import struct
        
        # Read original audio info
        with wave.open(audio_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            frames = wav_file.readframes(n_frames)
        
        # Check if preprocessing needed
        needs_preprocessing = False
        new_framerate = framerate
        new_channels = n_channels
        
        if n_channels > 1:
            print(f"[STT] Audio is stereo ({n_channels} channels), converting to mono...")
            new_channels = 1
            needs_preprocessing = True
        
        if framerate != 16000:
            print(f"[STT] Audio sample rate is {framerate}Hz, resampling to 16000Hz...")
            new_framerate = 16000
            needs_preprocessing = True
        
        if not needs_preprocessing:
            print(f"[STT] Audio is already optimal (mono, 16kHz)")
            return audio_path
        
        # Convert stereo to mono if needed
        if new_channels == 1 and n_channels > 1:
            # Simple stereo to mono: average channels
            audio_data = struct.unpack(f"<{n_frames * n_channels}h", frames)
            mono_data = []
            for i in range(0, len(audio_data), n_channels):
                avg_sample = sum(audio_data[i:i+n_channels]) // n_channels
                mono_data.append(avg_sample)
            frames = struct.pack(f"<{len(mono_data)}h", *mono_data)
        
        # Resample if needed (simple approach: repeat or skip frames)
        if new_framerate != framerate:
            audio_data = struct.unpack(f"<{len(frames)//sample_width}h", frames)
            ratio = new_framerate / framerate
            resampled_data = []
            
            for i in range(int(len(audio_data) * ratio)):
                src_idx = int(i / ratio)
                if src_idx < len(audio_data):
                    resampled_data.append(audio_data[src_idx])
            
            frames = struct.pack(f"<{len(resampled_data)}h", *resampled_data)
        
        # Write preprocessed audio
        preprocessed_path = audio_path.replace(".wav", "_preprocessed.wav")
        with wave.open(preprocessed_path, 'wb') as wav_file:
            wav_file.setnchannels(new_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(new_framerate)
            wav_file.writeframes(frames)
        
        print(f"[STT] ✅ Preprocessing complete: {preprocessed_path}")
        return preprocessed_path
        
    except Exception as e:
        print(f"[STT] ⚠️ Audio preprocessing failed: {str(e)[:100]}")
        print(f"[STT] Using original audio")
        return audio_path

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Transkrip file audio menggunakan whisper.cpp CLI dengan multilingual support
    
    Mendukung 3 bahasa: Indonesian, English, Arabic
    Language bisa auto-detect atau force ke specific language
    
    ⚠️ ACCURACY NOTES:
    - Jika hasil STT buruk: cek apakah model terlalu kecil (base) atau language salah
    - Untuk accuracy lebih baik: upgrade ke ggml-small.bin atau ggml-medium.bin
    - Jika auto-detect salah: force WHISPER_LANGUAGE ke language code yang benar
    
    Args:
        file_bytes (bytes): Isi file audio
        file_ext (str): Ekstensi file, default ".wav"
    
    Returns:
        str: Teks hasil transkripsi
    """
    # Cek apakah binary dan model file ada
    if not os.path.exists(WHISPER_BINARY):
        return f"[ERROR] Whisper binary not found at: {WHISPER_BINARY}"
    
    if not os.path.exists(WHISPER_MODEL_PATH):
        return f"[ERROR] Whisper model not found at: {WHISPER_MODEL_PATH}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{uuid.uuid4()}{file_ext}")
        result_path = os.path.join(tmpdir, "transcription.txt")

        # simpan audio ke file temporer
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        # Preprocess audio untuk optimize accuracy
        audio_path = _preprocess_audio(audio_path)

        # jalankan whisper.cpp dengan subprocess
        # Add parameters untuk improve accuracy dengan multilingual support
        cmd = [
            WHISPER_BINARY,
            "-m", WHISPER_MODEL_PATH,
            "-f", audio_path,
            "-l", WHISPER_LANGUAGE,           # Language: id/en/ar/auto
            "-t", str(WHISPER_THREADS),       # Number of threads
            "--temperature", str(WHISPER_TEMPERATURE),  # Temperature (0 = deterministic)
            "-otxt",
            "-of", result_path.replace(".txt", "")  # output base name tanpa extension
        ]

        try:
            print(f"[STT] Running whisper with parameters:")
            print(f"      Model: {MODEL_TYPE} (from {WHISPER_MODEL_PATH})")
            print(f"      Language: {WHISPER_LANGUAGE} ({SUPPORTED_LANGUAGES.get(WHISPER_LANGUAGE, 'Unknown')})")
            print(f"      Supported: {', '.join([f'{k}({v})' for k,v in SUPPORTED_LANGUAGES.items()])}")
            print(f"      Temperature: {WHISPER_TEMPERATURE} (lower = more consistent)")
            print(f"      Threads: {WHISPER_THREADS}")
            if MODEL_TYPE == "base":
                print(f"      ⚠️  NOTE: Using 'base' model (lowest accuracy). For better results:")
                print(f"         1. Download ggml-small.bin or ggml-medium.bin")
                print(f"         2. Update WHISPER_MODEL_PATH to use larger model")
                print(f"         3. This will improve accuracy but increase processing time")
            elif MODEL_TYPE == "small":
                print(f"      ✅ Using 'small' model (good accuracy/speed balance)")
                print(f"         To improve further: upgrade to 'medium' or 'large' model")
            print(f"[DEBUG] Full command: {' '.join(cmd[:8])}...")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            
            if result.stderr:
                print(f"[STT] Whisper output: {result.stderr[:200]}")
        
        except subprocess.TimeoutExpired:
            return "[ERROR] Whisper timeout (>10 minutes)"
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            return f"[ERROR] Whisper failed: {error_msg[:200]}"
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {str(e)[:200]}"
        
        # baca hasil transkripsi
        try:
            with open(result_path, "r", encoding="utf-8") as result_file:
                transcript = result_file.read().strip()
                print(f"[STT] ✅ Transcription success: {transcript[:80]}...")
                return transcript
        except FileNotFoundError:
            # Cek file alternatif
            alt_result_path = result_path.replace(".txt", "") + ".txt"
            if os.path.exists(alt_result_path):
                with open(alt_result_path, "r", encoding="utf-8") as result_file:
                    transcript = result_file.read().strip()
                    print(f"[STT] ✅ Transcription success: {transcript[:80]}...")
                    return transcript
            return "[ERROR] Transcription file not found"
import os
import re
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path ke folder utilitas TTS
COQUI_DIR = os.path.join(BASE_DIR, "coqui_utils")

# File model harus berada di dalam folder coqui_utils/
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")

# File config.json harus berada di dalam folder coqui_utils/
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")

# Path ke file speakers (dibutuhkan karena model ini multi-speaker)
COQUI_SPEAKERS_PATH = os.path.join(COQUI_DIR, "speakers.pth")

# Nama speaker yang digunakan (harus sesuai isi speakers.pth)
COQUI_SPEAKER = "wibowo"

# ===== SINGLETON G2P =====
# Model G2P (termasuk BERT/ONNX OOV predictor di baliknya) berat untuk di-load,
# jadi cukup load SEKALI dan dipakai ulang, bukan setiap kali fungsi dipanggil.
_G2P_INSTANCE = None


def _get_g2p():
    """
    PENTING — pastikan package g2p_id yang terinstall adalah versi TERBARU:
        pip install --upgrade --force-reinstall "git+https://github.com/Wikidepia/g2p-id"
        pip install python-crfsuite  # dependency-nya

    Versi LAMA repo ini memakai predictor berbasis ONNX/BERT untuk menebak
    pelafalan kata berhuruf 'e', dan versi itu rawan crash dengan error
    "[ONNXRuntimeError] INVALID_ARGUMENT: Unexpected input data type" di
    banyak environment (mismatch dtype antara onnxruntime & model .onnx yang
    dibundel). Versi terbaru sudah di-rewrite total memakai pendekatan
    rule-based + CRF syllabifier — TIDAK ada dependency onnxruntime sama
    sekali, jadi masalah ini hilang dari sumbernya, bukan ditambal di sini.
    """
    global _G2P_INSTANCE
    if _G2P_INSTANCE is None:
        from g2p_id import G2P
        print("[G2P] Loading G2P model (sekali saja)...")
        _G2P_INSTANCE = G2P()
    return _G2P_INSTANCE


# ===== TEXT-TO-PHONEME CONVERSION =====
def text_to_phonemes(text: str) -> str:
    """
    Konversi teks biasa ke fonem (IPA) menggunakan g2p-id (versi terbaru).

    g2p-id versi terbaru hanya memproses kata berhuruf [a-z] lewat
    kamus/aturan; tanda baca (titik, koma, dll), spasi, dan ANGKA dilewatkan
    apa adanya (tidak diubah, tidak dieja jadi kata) — sudah diverifikasi
    langsung lewat g2p.to_phoneme(). Karena itu tidak perlu split/replace
    manual untuk tanda baca: cukup panggil to_phoneme() pada teks penuh.

    Catatan: kalau butuh angka dibaca sebagai kata (mis. "5" -> "lima"),
    itu harus dilakukan di tahap PRA-proses terpisah sebelum fungsi ini,
    karena g2p-id versi ini tidak melakukannya secara native.

    Args:
        text (str): Teks dalam bahasa Indonesia, boleh mengandung tanda baca.

    Returns:
        str: Teks dalam bentuk fonem (IPA); tanda baca tetap ada di tempatnya.
    """
    clean_text = text.strip()
    if not clean_text:
        return text

    try:
        g2p = _get_g2p()
    except ImportError:
        print("[WARNING] g2p_id belum terinstall, menggunakan teks asli")
        return text

    clean_text = re.sub(r"\s+", " ", clean_text)
    print(f"[G2P] Input: '{clean_text[:80]}...'")

    try:
        phonemes, _syllables = g2p.to_phoneme(clean_text)
        print(f"[G2P] ✅ Output: '{phonemes[:80]}...'")
        return phonemes
    except Exception as e:
        # Fallback aman: kalau tetap gagal, pakai teks asli (model TTS akan
        # membaca ortografi mentah — kurang akurat, tapi tidak crash).
        print(f"[WARNING] G2P gagal ({str(e)[:100]}), menggunakan teks asli")
        return clean_text


def transcribe_text_to_speech(text: str) -> str:
    """
    Fungsi untuk mengonversi teks menjadi suara menggunakan Coqui TTS.

    Proses:
    1. Konversi teks → fonem (G2P), dengan fallback ke teks asli jika gagal
    2. Kirim fonem ke Coqui TTS
    3. Generate audio

    Args:
        text (str): Teks yang akan diubah menjadi suara.
    Returns:
        str: Path ke file audio hasil konversi atau pesan error.
    """
    if not text or not text.strip():
        return "[ERROR] Empty text provided"

    print(f"[TTS] Input text: {text[:100]}...")
    phonemes = text_to_phonemes(text)
    print(f"[TTS] After G2P: {phonemes[:100]}...")

    return _tts_with_coqui(phonemes)


# === ENGINE: Coqui TTS ===
def _tts_with_coqui(text: str) -> str:
    """
    Generate audio menggunakan Coqui TTS lewat subprocess.

    Args:
        text (str): Teks/fonem yang akan disintesis.

    Returns:
        str: Path ke file audio output, atau pesan error.
    """
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    # Gunakan basename dari variabel path di atas (bukan string hardcode
    # terpisah) supaya satu sumber kebenaran — cwd diset ke COQUI_DIR jadi
    # path relatif ini tetap valid.
    cmd = [
        "tts",
        "--text", text,
        "--model_path", os.path.basename(COQUI_MODEL_PATH),
        "--config_path", os.path.basename(COQUI_CONFIG_PATH),
        "--speakers_file_path", os.path.basename(COQUI_SPEAKERS_PATH),
        "--speaker_idx", COQUI_SPEAKER,
        "--out_path", output_path,
    ]

    try:
        print(f"[TTS] Running command: tts --text '{text[:50]}...' --speaker_idx {COQUI_SPEAKER}")
        print(f"[TTS] Working directory: {COQUI_DIR}")

        result = subprocess.run(cmd, check=True, cwd=COQUI_DIR, capture_output=True, text=True)

        if os.path.exists(output_path):
            print(f"[TTS] ✅ Audio generated successfully: {output_path}")
            return output_path

        stderr_msg = result.stderr if result.stderr else "Unknown error"
        print("[TTS] ❌ Output file not created")
        return f"[ERROR] Output file not created: {stderr_msg[:100]}"

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print("[TTS] ❌ Subprocess failed:")
        print(f"      Return code: {e.returncode}")
        print(f"      Error: {error_msg[:200]}")
        return f"[ERROR] TTS subprocess failed: {error_msg[:100]}"
    except FileNotFoundError as e:
        print(f"[TTS] ❌ TTS command not found (tts not installed?): {str(e)}")
        return "[ERROR] TTS command not found - install with: pip install TTS"
    except Exception as e:
        import traceback
        print(f"[TTS] ❌ Unexpected error: {type(e).__name__}: {str(e)}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()[:300]}")
        return f"[ERROR] {str(e)[:100]}"
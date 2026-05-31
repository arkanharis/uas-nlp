"""
ANALISIS PIPELINE - Voice Chatbot Evaluation
==============================================
Script untuk evaluasi komprehensif STT, LLM, TTS, dan end-to-end performance.

Evaluasi mencakup:
- STT: WER (Word Error Rate), CER (Character Error Rate)
- LLM: Correctness (kualitas jawaban)
- TTS: Naturalness (penilaian subjektif)
- End-to-end: Latency, Intelligibility
"""

import os
import sys
import csv
import time
import json
import wave
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess

# Try to import evaluation libraries
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("[WARNING] jiwer not installed. Install with: pip install jiwer")

# Import preprocessing module
try:
    from app.preprocessing import preprocess_text
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    print("[WARNING] Preprocessing module not available")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")
AUDIO_DIR = os.path.join(CORPUS_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(CORPUS_DIR, "transcript")
REFERENCE_FILE = os.path.join(TRANSCRIPT_DIR, "reference.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

class AudioMetrics:
    """Class untuk menghitung metrics dari audio file"""
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """
        Hitung durasi audio file dalam detik
        Args:
            audio_path: Path ke file audio
        Returns:
            Durasi dalam detik (float)
        """
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"[ERROR] Cannot read audio duration: {str(e)}")
            return 0.0
    
    @staticmethod
    def get_audio_info(audio_path: str) -> Dict:
        """
        Dapatkan informasi lengkap tentang audio file
        Args:
            audio_path: Path ke file audio
        Returns:
            Dictionary dengan informasi audio
        """
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = frames / float(framerate)
                
                return {
                    "channels": n_channels,
                    "sample_width": sample_width,
                    "sample_rate": framerate,
                    "total_frames": frames,
                    "duration_seconds": round(duration, 2),
                    "file_size_mb": os.path.getsize(audio_path) / (1024 * 1024)
                }
        except Exception as e:
            print(f"[ERROR] Cannot read audio info: {str(e)}")
            return {}

class TextMetrics:
    """Class untuk menghitung metrics dari teks"""
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Hitung Word Error Rate (WER)
        
        WER = (S + D + I) / N
        Dimana:
        - S = substitutions (kata yang diganti)
        - D = deletions (kata yang dihapus)
        - I = insertions (kata yang ditambah)
        - N = jumlah kata di reference
        
        Args:
            reference: Teks referensi (ground truth)
            hypothesis: Teks hasil STT
        Returns:
            WER dalam persentase (0-100)
        """
        if not JIWER_AVAILABLE:
            print("[WARNING] jiwer not available, skipping WER calculation")
            return -1.0
        
        try:
            wer_score = wer(reference, hypothesis)
            return min(wer_score * 100, 100.0)  # Cap at 100%
        except Exception as e:
            print(f"[ERROR] WER calculation failed: {str(e)}")
            return -1.0
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Hitung Character Error Rate (CER)
        
        CER = (S + D + I) / N
        Dimana:
        - S = substitutions (karakter yang diganti)
        - D = deletions (karakter yang dihapus)
        - I = insertions (karakter yang ditambah)
        - N = jumlah karakter di reference
        
        Args:
            reference: Teks referensi (ground truth)
            hypothesis: Teks hasil STT
        Returns:
            CER dalam persentase (0-100)
        """
        if not JIWER_AVAILABLE:
            print("[WARNING] jiwer not available, skipping CER calculation")
            return -1.0
        
        try:
            cer_score = cer(reference, hypothesis)
            return min(cer_score * 100, 100.0)  # Cap at 100%
        except Exception as e:
            print(f"[ERROR] CER calculation failed: {str(e)}")
            return -1.0
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Hitung similarity antara dua teks (simple comparison)
        Menggunakan Levenshtein distance
        
        Args:
            text1: Teks pertama
            text2: Teks kedua
        Returns:
            Similarity score (0-100)
        """
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Simple character-level similarity
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 100.0
        
        return (matches / max_len) * 100

class ReferenceLoader:
    """Class untuk load dan manage reference data"""
    
    def __init__(self, reference_file: str):
        self.reference_data = {}
        self.load_reference(reference_file)
    
    def load_reference(self, reference_file: str):
        """
        Load reference data dari CSV file
        
        Args:
            reference_file: Path ke reference.csv
        """
        if not os.path.exists(reference_file):
            print(f"[ERROR] Reference file not found: {reference_file}")
            return
        
        try:
            with open(reference_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    utterance_id = row['utterance_id'].strip()
                    ground_truth = row['ground_truth'].strip()
                    self.reference_data[utterance_id] = ground_truth
            
            print(f"✅ Loaded {len(self.reference_data)} reference entries")
        except Exception as e:
            print(f"[ERROR] Failed to load reference file: {str(e)}")
    
    def get_ground_truth(self, utterance_id: str) -> str:
        """
        Dapatkan ground truth untuk utterance_id tertentu
        
        Args:
            utterance_id: ID dari utterance
        Returns:
            Ground truth text atau empty string jika tidak ditemukan
        """
        return self.reference_data.get(utterance_id, "")
    
    def get_all_utterances(self) -> List[str]:
        """Dapatkan list semua utterance_id"""
        return list(self.reference_data.keys())

class PipelineEvaluator:
    """Class untuk evaluasi pipeline lengkap"""
    
    def __init__(self):
        self.reference_loader = ReferenceLoader(REFERENCE_FILE)
        self.results = []
        self.summary = {}
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Process satu file audio melalui seluruh pipeline
        
        Args:
            audio_path: Path ke file audio
        Returns:
            Dictionary dengan hasil evaluasi
        """
        # Extract speaker_id dan utterance_id dari filename
        # Format: idSpeaker_utterance_id.wav
        filename = os.path.basename(audio_path)
        name_parts = filename.replace('.wav', '').split('_', 1)
        
        if len(name_parts) != 2:
            print(f"[WARNING] Invalid filename format: {filename}")
            return None
        
        speaker_id = name_parts[0]
        utterance_id = name_parts[1]
        
        # Handle zero-padding: audio1 -> audio01, audio2 -> audio02, etc
        # Try to match with reference by padding if needed
        ground_truth = self.reference_loader.get_ground_truth(utterance_id)
        if not ground_truth:
            # Try with zero-padding: audio1 -> audio01
            if utterance_id.startswith('audio') and not utterance_id[5:].zfill(2) == utterance_id[5:]:
                padded_id = 'audio' + utterance_id[5:].zfill(2)
                ground_truth = self.reference_loader.get_ground_truth(padded_id)
                if ground_truth:
                    utterance_id = padded_id
        
        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")
        
        # Get ground truth
        if not ground_truth:
            print(f"[WARNING] Ground truth not found for: {utterance_id}")
            return None
        
        print(f"Ground Truth: {ground_truth[:80]}..." if len(ground_truth) > 80 else f"Ground Truth: {ground_truth}")
        
        # Measure latency
        start_time = time.time()
        
        # Result dictionary (minimal data)
        result = {
            "filename": filename,
            "ground_truth": ground_truth,
            "transcript": None,
            "preprocessing": None,
            "llm_response": None,
            "tts_output_file": None,
            "wer": None,
            "cer": None,
            "similarity": None,
            "latency": {
                "stt_ms": 0,
                "llm_ms": 0,
                "tts_ms": 0
            }
        }
        
        # [STT] Process audio through STT pipeline
        print("\n[STT] Processing audio through STT pipeline...")
        stt_start = time.time()
        try:
            from app.stt import transcribe_speech_to_text
            # Read audio bytes
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            transcript = transcribe_speech_to_text(audio_bytes, ".wav")
            if not transcript.startswith("[ERROR]"):
                result["transcript"] = transcript
                print(f"  ✅ Transcript: {transcript[:80]}..." if len(transcript) > 80 else f"  ✅ Transcript: {transcript}")
                
                # Evaluate STT - calculate WER, CER, Similarity
                wer_score = TextMetrics.calculate_wer(ground_truth, transcript)
                cer_score = TextMetrics.calculate_cer(ground_truth, transcript)
                similarity = TextMetrics.calculate_similarity(ground_truth, transcript)
                
                result["wer"] = wer_score
                result["cer"] = cer_score
                result["similarity"] = similarity
                
                print(f"  📊 STT Metrics:")
                print(f"     - WER: {wer_score:.2f}%" if wer_score >= 0 else f"     - WER: N/A")
                print(f"     - CER: {cer_score:.2f}%" if cer_score >= 0 else f"     - CER: N/A")
                print(f"     - Similarity: {similarity:.2f}%")
            else:
                print(f"  ❌ Error: {transcript}")
        except ImportError:
            print("  [WARNING] STT module not available")
        except Exception as e:
            print(f"  ❌ STT Error: {str(e)}")
        finally:
            stt_latency = time.time() - stt_start
            result["latency"]["stt_ms"] = round(stt_latency * 1000, 2)
        
        # [PREPROCESSING] Analyze language and code-switching
        if result["transcript"]:
            print(f"\n[PREPROCESSING] Analyzing language and code-switching...")
            try:
                if PREPROCESSING_AVAILABLE:
                    preprocess_result = preprocess_text(result["transcript"], mode="preserve")
                    
                    result["preprocessing"] = {
                        "detected_language": preprocess_result.detected_language,
                        "has_code_switching": preprocess_result.has_code_switching,
                        "language_distribution": preprocess_result.language_distribution
                    }
                    
                    print(f"  ✅ Detected Language: {preprocess_result.detected_language}")
                    print(f"  ✅ Code-switching: {preprocess_result.has_code_switching}")
                    if preprocess_result.language_distribution:
                        dist_str = ", ".join([f"{lang}:{pct:.0%}" for lang, pct in preprocess_result.language_distribution.items()])
                        print(f"  ✅ Language Distribution: {dist_str}")
                else:
                    print("  [WARNING] Preprocessing module not available")
            except Exception as e:
                print(f"  ❌ Preprocessing Error: {type(e).__name__}: {str(e)[:100]}")
        
        # [LLM] Process transcript through LLM
        if result["transcript"]:
            print(f"\n[LLM] Processing transcript through LLM...")
            llm_start = time.time()
            try:
                from app.llm import generate_response
                
                # Limit transcript length to avoid API errors
                transcript_input = result["transcript"][:500]
                
                response = generate_response(transcript_input)
                if not response.startswith("[ERROR]"):
                    result["llm_response"] = response
                    print(f"  ✅ Response: {response[:80]}..." if len(response) > 80 else f"  ✅ Response: {response}")
                else:
                    print(f"  ❌ Error: {response[:100]}...")
            except ImportError:
                print("  [WARNING] LLM module not available")
            except Exception as e:
                print(f"  ❌ LLM Error: {type(e).__name__}: {str(e)[:100]}")
            finally:
                llm_latency = time.time() - llm_start
                result["latency"]["llm_ms"] = round(llm_latency * 1000, 2)
        
        # [TTS] Convert LLM response to speech
        if result["llm_response"]:
            print(f"\n[TTS] Converting response to speech...")
            tts_start = time.time()
            try:
                from app.tts import transcribe_text_to_speech
                
                output_path = transcribe_text_to_speech(result["llm_response"])
                if not output_path.startswith("[ERROR]") and os.path.exists(output_path):
                    result["tts_output_file"] = output_path
                    print(f"  ✅ TTS Output: {output_path}")
                else:
                    print(f"  ❌ Error: {output_path[:100]}")
            except ImportError:
                print("  [WARNING] TTS module not available")
            except Exception as e:
                print(f"  ❌ TTS Error: {type(e).__name__}: {str(e)[:100]}")
            finally:
                tts_latency = time.time() - tts_start
                result["latency"]["tts_ms"] = round(tts_latency * 1000, 2)
        
        print(f"\n⏱️  Latency:")
        print(f"  - STT:   {result['latency']['stt_ms']:>8.2f} ms")
        if result["latency"]["llm_ms"] > 0:
            print(f"  - LLM:   {result['latency']['llm_ms']:>8.2f} ms")
        if result["latency"]["tts_ms"] > 0:
            print(f"  - TTS:   {result['latency']['tts_ms']:>8.2f} ms")
        
        return result
    
    def evaluate_stt(self, result: Dict):
        """
        Evaluasi STT performance
        
        Args:
            result: Result dictionary
        """
        ground_truth = result["ground_truth"]
        transcript = result["stt_result"]["transcript"]
        
        if not transcript:
            print("[WARNING] No STT transcript to evaluate")
            return
        
        # Calculate WER
        wer_score = TextMetrics.calculate_wer(ground_truth, transcript)
        
        # Calculate CER
        cer_score = TextMetrics.calculate_cer(ground_truth, transcript)
        
        # Calculate similarity
        similarity = TextMetrics.calculate_similarity(ground_truth, transcript)
        
        result["evaluation_metrics"]["stt"] = {
            "wer": wer_score,
            "cer": cer_score,
            "similarity": similarity
        }
        
        print(f"\nSTT Evaluation:")
        print(f"  - WER: {wer_score:.2f}%" if wer_score >= 0 else f"  - WER: N/A")
        print(f"  - CER: {cer_score:.2f}%" if cer_score >= 0 else f"  - CER: N/A")
        print(f"  - Similarity: {similarity:.2f}%")
    
    def evaluate_llm(self, result: Dict):
        """
        Evaluasi LLM performance (placeholder untuk manual evaluation)
        
        Args:
            result: Result dictionary
        """
        response = result["llm_result"]["response"]
        
        if not response:
            print("[WARNING] No LLM response to evaluate")
            return
        
        # TODO: Implement LLM evaluation metrics
        # - Relevance score
        # - Correctness score (manual)
        # - Completeness score
        
        result["evaluation_metrics"]["llm"] = {
            "status": "requires_manual_evaluation",
            "response_length": len(response),
            "word_count": len(response.split())
        }
        
        print(f"\nLLM Evaluation:")
        print(f"  - Response Length: {len(response)} characters")
        print(f"  - Word Count: {len(response.split())} words")
        print(f"  - [MANUAL] Correctness score needed")
    
    def evaluate_tts(self, result: Dict):
        """
        Evaluasi TTS performance (placeholder untuk subjective evaluation)
        
        Args:
            result: Result dictionary
        """
        output_file = result["tts_result"]["output_file"]
        
        if not output_file or not os.path.exists(output_file):
            print("[WARNING] No TTS output file to evaluate")
            return
        
        tts_metrics = AudioMetrics.get_audio_info(output_file)
        
        result["evaluation_metrics"]["tts"] = {
            "status": "requires_subjective_evaluation",
            "audio_metrics": tts_metrics
        }
        
        print(f"\nTTS Evaluation:")
        print(f"  - Output Duration: {tts_metrics.get('duration_seconds', 0)} seconds")
        print(f"  - [SUBJECTIVE] Naturalness score needed (1-5)")
        print(f"  - [SUBJECTIVE] Intelligibility score needed (1-5)")
    
    def evaluate_end_to_end(self, result: Dict):
        """
        Evaluasi end-to-end performance
        
        Args:
            result: Result dictionary
        """
        # Calculate total latency
        # TODO: Implement timing throughout pipeline
        
        result["evaluation_metrics"]["end_to_end"] = {
            "status": "not_implemented",
            "total_latency_ms": None,
            "intelligibility": None  # Requires subjective evaluation
        }
        
        print(f"\nEnd-to-End Evaluation:")
        print(f"  - [TODO] Total Latency: needs implementation")
        print(f"  - [SUBJECTIVE] Intelligibility score needed (1-5)")
    
    def process_corpus(self, limit: int = None) -> List[Dict]:
        """
        Process seluruh corpus audio
        
        Args:
            limit: Limit jumlah file yang diprocess (None = semua)
        Returns:
            List hasil evaluasi
        """
        if not os.path.exists(AUDIO_DIR):
            print(f"[ERROR] Audio directory not found: {AUDIO_DIR}")
            return []
        
        audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])
        
        if limit:
            audio_files = audio_files[:limit]
        
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING {len(audio_files)} AUDIO FILES")
        print(f"{'='*70}")
        
        for idx, filename in enumerate(audio_files, 1):
            audio_path = os.path.join(AUDIO_DIR, filename)
            
            print(f"\n[{idx}/{len(audio_files)}] Processing {filename}")
            
            result = self.process_audio_file(audio_path)
            
            if result:
                self.results.append(result)
        
        return self.results
    
    def save_results(self, output_file: str = None):
        """
        Simpan hasil evaluasi ke file JSON
        
        Args:
            output_file: Path output file (default: analysis_results/results_TIMESTAMP.json)
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
        
        try:
            # Convert numpy types to python types for JSON serialization
            serializable_results = []
            for result in self.results:
                serializable_results.append(self._make_serializable(result))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Results saved to: {output_file}")
            return output_file
        
        except Exception as e:
            print(f"[ERROR] Failed to save results: {str(e)}")
            return None
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (float, int)):
            return obj
        else:
            return str(obj)
    
    def generate_summary_report(self):
        """Generate summary report dari semua evaluasi"""
        print(f"\n{'='*70}")
        print(f"📈 SUMMARY REPORT")
        print(f"{'='*70}")
        
        if not self.results:
            print("[WARNING] No results to summarize")
            return
        
        total_files = len(self.results)
        successful_files = len([r for r in self.results if r["transcript"] is not None])
        
        # STT Summary
        stt_wers = []
        stt_cers = []
        stt_similarities = []
        
        for result in self.results:
            if result.get("wer") is not None and result["wer"] >= 0:
                stt_wers.append(result["wer"])
            if result.get("cer") is not None and result["cer"] >= 0:
                stt_cers.append(result["cer"])
            if result.get("similarity") is not None:
                stt_similarities.append(result["similarity"])
        
        # Latency Summary
        stt_latencies = [r["latency"]["stt_ms"] for r in self.results if r["latency"]["stt_ms"] > 0]
        llm_latencies = [r["latency"]["llm_ms"] for r in self.results if r["latency"]["llm_ms"] > 0]
        tts_latencies = [r["latency"]["tts_ms"] for r in self.results if r["latency"]["tts_ms"] > 0]
        
        print(f"\n📊 Overall Processing:")
        print(f"  - Total Files: {total_files}")
        print(f"  - Successful: {successful_files}/{total_files}")
        
        print(f"\n📝 STT Metrics:")
        if stt_wers:
            avg_wer = sum(stt_wers) / len(stt_wers)
            print(f"  - Average WER: {avg_wer:.2f}%")
            print(f"  - Min WER: {min(stt_wers):.2f}%")
            print(f"  - Max WER: {max(stt_wers):.2f}%")
        else:
            print(f"  - WER: Not calculated")
        
        if stt_cers:
            avg_cer = sum(stt_cers) / len(stt_cers)
            print(f"  - Average CER: {avg_cer:.2f}%")
        
        if stt_similarities:
            avg_sim = sum(stt_similarities) / len(stt_similarities)
            print(f"  - Average Similarity: {avg_sim:.2f}%")
        
        print(f"\n⏱️  Latency Breakdown:")
        
        if stt_latencies:
            avg_stt = sum(stt_latencies) / len(stt_latencies)
            print(f"  - STT (Speech-to-Text):")
            print(f"     Average: {avg_stt:.2f}ms | Min: {min(stt_latencies):.2f}ms | Max: {max(stt_latencies):.2f}ms")
        
        if llm_latencies:
            avg_llm = sum(llm_latencies) / len(llm_latencies)
            print(f"  - LLM (Language Model):")
            print(f"     Average: {avg_llm:.2f}ms | Min: {min(llm_latencies):.2f}ms | Max: {max(llm_latencies):.2f}ms")
        
        if tts_latencies:
            avg_tts = sum(tts_latencies) / len(tts_latencies)
            print(f"  - TTS (Text-to-Speech):")
            print(f"     Average: {avg_tts:.2f}ms | Min: {min(tts_latencies):.2f}ms | Max: {max(tts_latencies):.2f}ms")
        
        # Preprocessing Summary
        print(f"\n🌐 Preprocessing Analysis:")
        preprocessing_results = [r.get("preprocessing") for r in self.results if r.get("preprocessing")]
        language_counts = {}
        code_switch_count = 0
        
        if preprocessing_results:
            successful_preprocess = [p for p in preprocessing_results if p is not None]
            print(f"  - Total Processed: {len(successful_preprocess)}/{len(preprocessing_results)}")
            
            if successful_preprocess:
                for p in successful_preprocess:
                    detected_lang = p.get("detected_language", "unknown")
                    language_counts[detected_lang] = language_counts.get(detected_lang, 0) + 1
                    
                    if p.get("has_code_switching"):
                        code_switch_count += 1
                
                print(f"  - Language Distribution:")
                for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = (count / len(successful_preprocess)) * 100
                    print(f"     • {lang.upper()}: {count} ({pct:.1f}%)")
                
                print(f"  - Code-switching: {code_switch_count}/{len(successful_preprocess)} detected")
        else:
            print(f"  - Preprocessing: Not available")
        
        self.summary = {
            "total_files": total_files,
            "successful_files": successful_files,
            "stt": {
                "avg_wer": sum(stt_wers) / len(stt_wers) if stt_wers else None,
                "avg_cer": sum(stt_cers) / len(stt_cers) if stt_cers else None,
                "avg_similarity": sum(stt_similarities) / len(stt_similarities) if stt_similarities else None,
            },
            "latency": {
                "stt_ms": {
                    "avg": sum(stt_latencies) / len(stt_latencies) if stt_latencies else None,
                    "min": min(stt_latencies) if stt_latencies else None,
                    "max": max(stt_latencies) if stt_latencies else None,
                },
                "llm_ms": {
                    "avg": sum(llm_latencies) / len(llm_latencies) if llm_latencies else None,
                    "min": min(llm_latencies) if llm_latencies else None,
                    "max": max(llm_latencies) if llm_latencies else None,
                },
                "tts_ms": {
                    "avg": sum(tts_latencies) / len(tts_latencies) if tts_latencies else None,
                    "min": min(tts_latencies) if tts_latencies else None,
                    "max": max(tts_latencies) if tts_latencies else None,
                }
            }
        }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Voice Chatbot Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analisis_pipeline.py --limit 5          # Process first 5 files
  python analisis_pipeline.py --all              # Process all files
  python analisis_pipeline.py --file audio01     # Process specific utterance
        """
    )
    
    parser.add_argument('--limit', type=int, default=5,
                       help='Limit jumlah file yang diprocess (default: 5)')
    parser.add_argument('--all', action='store_true',
                       help='Process semua file di corpus')
    parser.add_argument('--file', type=str,
                       help='Process file spesifik (format: utterance_id)')
    parser.add_argument('--output', type=str,
                       help='Output file untuk hasil (JSON)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎙️  VOICE CHATBOT PIPELINE ANALYZER")
    print("="*70)
    
    evaluator = PipelineEvaluator()
    
    # Determine limit
    limit = None if args.all else args.limit
    
    # Process corpus
    results = evaluator.process_corpus(limit=limit)
    
    # Generate summary
    evaluator.generate_summary_report()
    
    # Save results
    output_file = evaluator.save_results(args.output)
    
    print(f"\n✅ Analysis complete! Processed {len(results)} files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

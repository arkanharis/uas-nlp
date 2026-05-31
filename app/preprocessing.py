"""
Preprocessing layer untuk deteksi bahasa, code-switching detection,
normalisasi, dan translasi untuk sistem multilingual voice chat.

Mendukung tiga bahasa:
- Indonesian (id)
- English (en)  
- Arabic (ar)

Mode Operasional:
1. preserve: Pertahankan pola code-switching asli
2. normalized: Normalisasi ke satu bahasa (deteksi dominan)
3. translate: Terjemahkan ke bahasa target (opsional)
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CodeSwitchSegment:
    """Representasi untuk satu segmen teks dengan bahasa yang terdeteksi"""
    text: str
    language: str  # 'id', 'en', 'ar', 'mixed', 'unknown'
    confidence: float  # 0.0 - 1.0
    start_pos: int
    end_pos: int


@dataclass
class PreprocessingResult:
    """Hasil preprocessing dengan metadata lengkap"""
    original: str
    normalized: str
    preserve: str
    mode: str  # 'preserve', 'normalized', 'translate'
    detected_language: str  # Bahasa dominan
    segments: List[CodeSwitchSegment]
    has_code_switching: bool
    language_distribution: Dict[str, float]  # {'id': 0.6, 'en': 0.3, 'ar': 0.1}
    error: str = None


class LanguageDetector:
    """Deteksi bahasa untuk Indonesian, English, dan Arabic"""
    
    # Kata kunci bahasa Indonesia
    INDONESIAN_KEYWORDS = {
        'saya', 'anda', 'dia', 'kami', 'kalian', 'mereka',
        'dan', 'atau', 'tetapi', 'namun', 'karena', 'jika', 'ketika',
        'yang', 'mana', 'siapa', 'apa', 'bagaimana', 'berapa', 'dimana', 'kapan',
        'adalah', 'ada', 'ada', 'menjadi', 'perlu', 'bisa', 'dapat', 'harus',
        'akan', 'pernah', 'telah', 'sudah', 'belum', 'baru', 'masih',
        'tidak', 'bukan', 'jangan', 'jadi', 'itu', 'ini', 'sini', 'situ',
        'selamat', 'terima', 'kasih', 'permisi', 'maaf', 'tolong', 'mohon',
        'bagus', 'baik', 'buruk', 'jelek', 'benar', 'salah', 'tepat', 'iya',
        'halo', 'hai', 'pagi', 'siang', 'sore', 'malam', 'malam', 'hari',
        'pulang', 'pergi', 'datang', 'tiba', 'berangkat', 'kembali', 'ingat',
        'tahu', 'kenal', 'mengerti', 'paham', 'lupa', 'ingat', 'pikir', 'rasa',
    }
    
    # Kata kunci bahasa English
    ENGLISH_KEYWORDS = {
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'and', 'or', 'but', 'if', 'when', 'where', 'what', 'who', 'how',
        'is', 'are', 'am', 'be', 'been', 'being', 'have', 'has', 'do', 'does',
        'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
        'not', 'no', 'don\'t', 'can\'t', 'won\'t', 'isn\'t', 'aren\'t',
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'hello', 'hi', 'thanks', 'please', 'sorry', 'excuse', 'yes', 'no',
        'morning', 'afternoon', 'evening', 'night', 'day', 'week', 'month', 'year',
        'good', 'bad', 'nice', 'great', 'terrible', 'amazing', 'awesome',
        'go', 'come', 'stay', 'leave', 'arrive', 'depart', 'remember', 'forget',
        'know', 'understand', 'think', 'feel', 'want', 'need', 'like', 'love',
    }
    
    # Kata kunci bahasa Arabic
    ARABIC_KEYWORDS = {
        'أنا', 'أنت', 'هو', 'هي', 'نحن', 'أنتم', 'هم', 'هن',  # Pronouns
        'و', 'أو', 'لكن', 'لو', 'إذا', 'عندما', 'أين', 'ما', 'من', 'كيف',  # Conjunctions/questions
        'هو', 'ليس', 'كان', 'كانت', 'كانوا', 'سوف', 'قد', 'يمكن', 'لا',  # Verbs
        'ال', 'هذا', 'تلك', 'ذلك', 'هؤلاء', 'هاتان',  # Articles/demonstratives
        'السلام', 'عليكم', 'شكرا', 'من فضلك', 'معاف', 'آسف',  # Greetings/politeness
        'صباح', 'مساء', 'يوم', 'ليل', 'أسبوع', 'شهر', 'سنة',  # Time
        'جيد', 'سيء', 'حسن', 'رديء', 'رائع', 'عظيم',  # Adjectives
        'ذهاب', 'جاء', 'بقي', 'ترك', 'وصل', 'غادر', 'تذكر', 'نسي',  # Verbs
        'يعرف', 'يفهم', 'يفكر', 'يشعر', 'يريد', 'يحتاج', 'يحب',  # Verbs
    }
    
    # Pattern untuk deteksi karakter Arabic
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]')  # Unicode range Arabic
    
    @classmethod
    def detect_language(cls, text: str) -> Tuple[str, float]:
        """
        Deteksi bahasa dari teks
        
        Args:
            text (str): Teks untuk dianalisis
        
        Returns:
            Tuple[str, float]: (language_code, confidence)
                language_code: 'id', 'en', 'ar', 'mixed', 'unknown'
                confidence: 0.0 - 1.0
        """
        if not text or len(text.strip()) == 0:
            return 'unknown', 0.0
        
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        if not words:
            return 'unknown', 0.0
        
        # Hitung jumlah kata untuk setiap bahasa
        id_count = sum(1 for word in words if word in cls.INDONESIAN_KEYWORDS)
        en_count = sum(1 for word in words if word in cls.ENGLISH_KEYWORDS)
        
        # Deteksi karakter Arabic
        ar_chars = cls.ARABIC_PATTERN.findall(text)
        ar_count = len(ar_chars)
        
        # Hitung persentase
        total_words = len(words)
        id_ratio = id_count / total_words if total_words > 0 else 0
        en_ratio = en_count / total_words if total_words > 0 else 0
        ar_ratio = (ar_count / len(text)) if len(text) > 0 else 0
        
        # Tentukan bahasa dominan
        scores = [
            ('id', id_ratio),
            ('en', en_ratio),
            ('ar', ar_ratio),
        ]
        
        # Cek jika lebih dari satu bahasa terdeteksi (code-switching)
        detected_langs = [lang for lang, score in scores if score > 0.15]
        
        if len(detected_langs) > 1:
            return 'mixed', max(score for _, score in scores)
        
        # Tentukan bahasa dengan skor tertinggi
        dominant_lang, confidence = max(scores, key=lambda x: x[1])
        
        # Jika confidence rendah, return 'unknown'
        if confidence < 0.1:
            return 'unknown', confidence
        
        return dominant_lang, confidence
    
    @classmethod
    def detect_segments(cls, text: str) -> List[CodeSwitchSegment]:
        """
        Deteksi segmen code-switching dalam teks
        
        Args:
            text (str): Teks untuk dianalisis
        
        Returns:
            List[CodeSwitchSegment]: Daftar segmen dengan bahasa terdeteksi
        """
        segments = []
        
        # Split teks berdasarkan spasi dan tanda baca
        pattern = r'(\s+|[.,!?;:-])'
        tokens = re.split(pattern, text)
        
        pos = 0
        for token in tokens:
            if not token or token.isspace() or token in '.,!?;:\-':
                pos += len(token)
                continue
            
            lang, conf = cls.detect_language(token)
            segment = CodeSwitchSegment(
                text=token,
                language=lang,
                confidence=conf,
                start_pos=pos,
                end_pos=pos + len(token)
            )
            segments.append(segment)
            pos += len(token)
        
        return segments


class CodeSwitchNormalizer:
    """Normalisasi teks dengan code-switching"""
    
    @staticmethod
    def normalize_to_dominant_language(text: str, detected_lang: str) -> str:
        """
        Normalisasi teks ke bahasa dominan yang terdeteksi.
        Dalam konteks ini, kita pertahankan teks asli karena translasi 
        memerlukan model eksternal.
        
        Args:
            text (str): Teks asli dengan code-switching
            detected_lang (str): Bahasa dominan yang terdeteksi
        
        Returns:
            str: Teks yang dinormalisasi (dalam praktik, dapat diintegrasikan 
                 dengan translation API untuk hasil lebih baik)
        """
        # Cleanup whitespace dan tanda baca
        normalized = re.sub(r'\s+', ' ', text).strip()
        
        # Untuk saat ini, return teks yang dibersihkan
        # Di masa depan, bisa diintegrasikan dengan Google Translate API
        return normalized
    
    @staticmethod
    def preserve_code_switching(text: str, segments: List[CodeSwitchSegment]) -> str:
        """
        Pertahankan pola code-switching dengan menambahkan markup language tags.
        Format: [ID]text[/ID] [EN]text[/EN] etc.
        
        Args:
            text (str): Teks asli
            segments (List[CodeSwitchSegment]): Segmen dengan deteksi bahasa
        
        Returns:
            str: Teks dengan language tags
        """
        if not segments:
            return text
        
        result = []
        last_lang = None
        
        for segment in segments:
            if segment.language != last_lang and segment.language != 'unknown':
                if last_lang is not None:
                    result.append(f"[/{last_lang.upper()}]")
                result.append(f"[{segment.language.upper()}]")
                last_lang = segment.language
            
            result.append(segment.text)
        
        if last_lang is not None:
            result.append(f"[/{last_lang.upper()}]")
        
        return ''.join(result)


class TextPreprocessor:
    """Main preprocessing class yang menggabungkan semua fungsi"""
    
    @staticmethod
    def preprocess(
        text: str,
        mode: str = "preserve",
        translate_to: str = None
    ) -> PreprocessingResult:
        """
        Preprocess teks dengan deteksi bahasa, code-switching, dan normalisasi
        
        Args:
            text (str): Teks input untuk diproses
            mode (str): Mode operasional
                - 'preserve': Pertahankan code-switching asli
                - 'normalized': Normalisasi ke bahasa dominan
                - 'translate': Terjemahkan ke bahasa target
            translate_to (str): Bahasa target untuk mode 'translate' (opsional)
        
        Returns:
            PreprocessingResult: Hasil preprocessing dengan metadata
        """
        try:
            if not text or not isinstance(text, str):
                return PreprocessingResult(
                    original=text,
                    normalized=text,
                    preserve=text,
                    mode=mode,
                    detected_language='unknown',
                    segments=[],
                    has_code_switching=False,
                    language_distribution={},
                    error="Invalid input text"
                )
            
            # Step 1: Deteksi bahasa dominan
            detected_lang, confidence = LanguageDetector.detect_language(text)
            
            # Step 2: Deteksi segmen code-switching
            segments = LanguageDetector.detect_segments(text)
            
            # Step 3: Hitung distribusi bahasa
            language_counts = {}
            for segment in segments:
                if segment.language != 'unknown':
                    language_counts[segment.language] = language_counts.get(segment.language, 0) + 1
            
            total_segments = sum(language_counts.values())
            language_distribution = {
                lang: count / total_segments 
                for lang, count in language_counts.items()
            } if total_segments > 0 else {}
            
            # Deteksi apakah ada code-switching
            has_code_switching = len(language_counts) > 1
            
            # Step 4: Apply normalization based on mode
            if mode == "preserve":
                normalized_text = text
                preserve_text = CodeSwitchNormalizer.preserve_code_switching(text, segments)
            
            elif mode == "normalized":
                normalized_text = CodeSwitchNormalizer.normalize_to_dominant_language(
                    text, detected_lang
                )
                preserve_text = CodeSwitchNormalizer.preserve_code_switching(text, segments)
            
            elif mode == "translate":
                # Mode translate memerlukan API eksternal
                # Untuk saat ini, gunakan normalized text
                normalized_text = CodeSwitchNormalizer.normalize_to_dominant_language(
                    text, detected_lang
                )
                preserve_text = CodeSwitchNormalizer.preserve_code_switching(text, segments)
            
            else:
                normalized_text = text
                preserve_text = text
            
            return PreprocessingResult(
                original=text,
                normalized=normalized_text,
                preserve=preserve_text,
                mode=mode,
                detected_language=detected_lang,
                segments=segments,
                has_code_switching=has_code_switching,
                language_distribution=language_distribution,
                error=None
            )
        
        except Exception as e:
            return PreprocessingResult(
                original=text,
                normalized=text,
                preserve=text,
                mode=mode,
                detected_language='unknown',
                segments=[],
                has_code_switching=False,
                language_distribution={},
                error=str(e)
            )
    
    @staticmethod
    def get_text_for_llm(result: PreprocessingResult) -> str:
        """
        Dapatkan teks yang sesuai untuk dikirim ke LLM berdasarkan mode
        
        Args:
            result (PreprocessingResult): Hasil preprocessing
        
        Returns:
            str: Teks yang siap dikirim ke LLM
        """
        if result.mode == "preserve":
            return result.preserve
        elif result.mode == "normalized":
            return result.normalized
        else:
            return result.original


def preprocess_text(text: str, mode: str = "preserve") -> PreprocessingResult:
    """
    Convenience function untuk preprocessing
    
    Args:
        text (str): Teks untuk diproses
        mode (str): Mode operasional ('preserve', 'normalized', 'translate')
    
    Returns:
        PreprocessingResult: Hasil preprocessing
    """
    return TextPreprocessor.preprocess(text, mode)

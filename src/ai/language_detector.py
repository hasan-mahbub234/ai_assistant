import re
import logging
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detect language of text (English, Bengali, or Banglish)"""
    
    # Banglish patterns
    BANGLISH_WORDS = {
        'ami', 'amar', 'tui', 'apni', 'se', 'tar', 'bolchi', 'korchi',
        'hoyeche', 'daam', 'taka', 'kinbo', 'chai', 'valo', 'kharap',
        'ache', 'nai', 'ki', 'keno', 'kothay', 'jabe', 'ashbe', 'dibo'
    }
    
    # Banglish indicators
    BANGLISH_PATTERNS = [
        r'\b(ami|amar|tui|apni|se)\b',
        r'\b(korchi|bolchi|hoyeche|kinbo|chai)\b',
        r'\b(daam|taka|kinbo)\b',
        r'\b(ki|keno|kothay|jabe|ashbe)\b',
        r'\d+\s*(taka|tk)\b'
    ]
    
    def detect(self, text: str) -> str:
        """Detect language: english, bengali, or banglish"""
        if not text or len(text.strip()) < 2:
            return 'english'
        
        text_lower = text.lower()
        
        # Check for Banglish first
        if self._is_banglish(text_lower):
            logger.info(f"Detected Banglish: {text[:50]}")
            return 'banglish'
        
        # Check for Bengali characters
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        if bengali_chars > len(text) * 0.2:  # 20% Bengali characters
            logger.info(f"Detected Bengali: {text[:50]}")
            return 'bengali'
        
        # Use langdetect for other languages
        try:
            detected = detect(text)
            if detected == 'bn':
                return 'bengali'
            else:
                return 'english'
        except LangDetectException:
            return 'english'
    
    def _is_banglish(self, text: str) -> bool:
        """Check if text is Banglish"""
        # Check for Banglish patterns
        for pattern in self.BANGLISH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for Banglish words
        words = re.findall(r'\b[a-z]+\b', text)
        banglish_word_count = sum(1 for word in words if word in self.BANGLISH_WORDS)
        
        if banglish_word_count >= 2:  # At least 2 Banglish words
            return True
        
        return False
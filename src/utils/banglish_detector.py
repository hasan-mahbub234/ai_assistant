import re
from langdetect import detect, LangDetectException
import logging

logger = logging.getLogger(__name__)

class BanglishDetector:
    """Detect Banglish (Bengali written in Latin characters) and formal Bengali"""
    
    # Common Banglish patterns
    BANGLISH_PATTERNS = {
        # Common words and phrases
        'ami': 'আমি',
        'tui': 'তুই',
        'apni': 'আপনি',
        'se': 'সে',
        'tar': 'তার',
        'amar': 'আমার',
        'bolchi': 'বলছি',
        'korchi': 'করছি',
        'hoyeche': 'হয়েছে',
        'daam': 'দাম',
        'taka': 'টাকা',
        'kinbo': 'কিনবো',
        'ditam': 'দিতাম',
        'kotai': 'কোথাই',
        'chai': 'চাই',
        'manush': 'মানুষ',
        'valow': 'ভালো',
        'kharap': 'খারাপ',
        'ache': 'আছে',
        'nai': 'নাই',
        'ki': 'কি',
        'keno': 'কেনো',
        'kothao': 'কোথাও',
        'jabe': 'যাবে',
        'ashbe': 'আশবে',
        'dibo': 'দিবো',
    }
    
    # Banglish indicators
    BANGLISH_INDICATORS = [
        r'\b(ami|amar|tui|apni|se)\b',  # Pronouns
        r'\b(korchi|bolchi|hoyeche|kinbo|chai)\b',  # Verbs
        r'\b(daam|taka|kinbo)\b',  # Money/commerce
        r'\b(ki|keno|kothao|jabe|ashbe)\b',  # Question words
        r'\b(tir\s+daam|product\s+tir\s+daam)\b',  # Product pricing pattern
        r'\d+\s*(taka|tk)',  # Price pattern: "200 taka"
    ]
    
    @staticmethod
    def is_banglish(text):
        """Check if text is in Banglish"""
        if not text or len(text) < 2:
            return False
        
        text_lower = text.lower()
        
        # Check for direct Banglish word matches
        for pattern in BanglishDetector.BANGLISH_INDICATORS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check character distribution (Latin with no vowels pattern for Bengali)
        latin_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        bengali_chars = sum(1 for c in text if ord(c) >= 2432 and ord(c) <= 2559)
        
        # If mostly Latin but contains Banglish indicators, it's Banglish
        if latin_chars > bengali_chars and re.search(r'\b[a-z]{2,}\b', text_lower):
            for word in text_lower.split():
                word_clean = re.sub(r'[^a-z]', '', word)
                if word_clean in BanglishDetector.BANGLISH_PATTERNS:
                    return True
        
        return False
    
    @staticmethod
    def detect_language(text):
        """
        Detect language: 'banglish', 'bengali', or 'english'
        """
        if not text or len(text.strip()) == 0:
            return 'english'
        
        # Check for Banglish first (highest priority)
        if BanglishDetector.is_banglish(text):
            logger.info(f"Detected Banglish: {text[:50]}")
            return 'banglish'
        
        # Check for Bengali
        bengali_chars = sum(1 for c in text if ord(c) >= 2432 and ord(c) <= 2559)
        if bengali_chars > len(text) * 0.3:  # More than 30% Bengali characters
            logger.info(f"Detected Bengali: {text[:50]}")
            return 'bengali'
        
        # Fall back to langdetect for English/other languages
        try:
            detected = detect(text)
            if detected == 'bn':
                return 'bengali'
            elif detected == 'en':
                return 'english'
            else:
                return 'english'
        except LangDetectException:
            return 'english'
    
    @staticmethod
    def normalize_banglish_to_bengali(text):
        """Convert Banglish text to Bengali (basic transliteration)"""
        text_lower = text.lower()
        result = text_lower
        
        # Apply transliteration patterns
        for banglish, bengali in BanglishDetector.BANGLISH_PATTERNS.items():
            # Use word boundaries for accurate replacement
            pattern = r'\b' + banglish + r'\b'
            result = re.sub(pattern, bengali, result, flags=re.IGNORECASE)
        
        return result

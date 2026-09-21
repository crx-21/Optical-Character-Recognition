import re
from enum import Enum, auto
from app.constants import US_STATES

class OCRCategory(Enum):
    """Categories of OCR targets for domain-specific normalization."""
    US_PLATE = auto()
    CN_PLATE = auto()
    ROAD_SIGN = auto()
    SPEED_PLAQUE = auto()
    GENERAL = auto()

def detect_category(text: str) -> OCRCategory:
    """
    Heuristically detects the OCR target category based on text patterns.

    Args:
        text: The raw OCR output text.

    Returns:
        The detected OCRCategory.
    """
    if not text:
        return OCRCategory.GENERAL

    # Basic Chinese character check for CN plates
    # Check for any CJK Unified Ideograph
    if re.search(r'[一-龥]', text):
        return OCRCategory.CN_PLATE

    # Check for numeric content
    if any(digit in text for digit in "0123456789"):
        # Short numeric strings or those with some digits are likely plates or speed signs
        if len(text) < 20:
            if len(text) < 5 and text.isdigit():
                return OCRCategory.SPEED_PLAQUE
            return OCRCategory.US_PLATE

    return OCRCategory.ROAD_SIGN

def normalize_text(text: str, category: OCRCategory) -> str:
    """
    Normalizes OCR text based on domain-specific rules.

    Args:
        text: The raw OCR text to normalize.
        category: The target OCRCategory for normalization rules.

    Returns:
        The normalized string.
    """
    if not text:
        return ""

    # Multi-line text: Read top-to-bottom, joining lines with a single space.
    text = re.sub(r'\s+', ' ', text).strip()

    if category == OCRCategory.US_PLATE:
        # US License Plates: Extract ONLY the core alphanumeric registration.
        # Strip state names/slogans.
        pattern = re.compile(r'\b(' + '|'.join(US_STATES) + r')\b', re.IGNORECASE)
        text = pattern.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()

    elif category == OCRCategory.CN_PLATE:
        # Chinese Plates: KEEP the leading Chinese province character & letter prefix (e.g., 京A or 沪B).
        # Pattern: [Chinese Char][Letter][Alphanumeric]
        # [一-龥] matches the common range of CJK Unified Ideographs.
        match = re.search(r'([一-龥][a-zA-Z])\s*([a-zA-Z0-9]+)', text)
        if match:
            text = f"{match.group(1)}{match.group(2)}"
        else:
            # Fallback: try to keep the start if it looks like a Chinese char
            if re.match(r'^[一-龥]', text):
                text = text.strip()

    elif category == OCRCategory.ROAD_SIGN:
        # Road Signs: Retain full wording (e.g., SPEED LIMIT 65, STOP).
        text = text.upper().strip()

    elif category == OCRCategory.SPEED_PLAQUE:
        # Advisory Speed Plaques: Digits only (e.g., 35). Do not append units.
        digits = re.findall(r'\d+', text)
        text = digits[0] if digits else ""

    return text

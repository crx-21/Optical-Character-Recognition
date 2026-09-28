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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r'[\u4e00-\u9fff]')

# A misread character is most likely one of these look-alikes.
# Used when a slot must be a DIGIT but OCR returned a letter.
LETTER_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
}

# Used when a slot must be a LETTER but OCR returned a digit.
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
}

# D = digit slot, L = letter slot. Extend with the formats you actually see.
PLATE_TEMPLATES = [
    "DLLLDDD",   # e.g. 7ABC123 (California)
    "LLLDDDD",   # e.g. ABC1234
    "LLLDDD",    # e.g. ABC123
    "DDDLLL",    # e.g. 123ABC
    "LLDDDD",    # e.g. AB1234
]


def _build_state_names() -> list:
    """
    Full state names only (longest first, so 'WEST VIRGINIA' wins over 'VIRGINIA').
    Two-letter abbreviations (OR, IN, OK, ME, CA...) are skipped on purpose:
    they collide with real plate characters.
    """
    if isinstance(US_STATES, dict):
        raw = list(US_STATES.keys()) + list(US_STATES.values())
    else:
        raw = list(US_STATES)
    names = {str(s).strip().upper() for s in raw}
    return sorted((n for n in names if len(n) > 2), key=len, reverse=True)


_STATE_NAMES = _build_state_names()
_STATE_RE = (
    re.compile(r'\b(' + '|'.join(re.escape(n) for n in _STATE_NAMES) + r')\b')
    if _STATE_NAMES else None
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _strip_states(text_upper: str) -> str:
    """Remove full US state names from an already-uppercased string."""
    if _STATE_RE is None:
        return text_upper
    return _clean_whitespace(_STATE_RE.sub('', text_upper))


def fix_by_template(s: str, max_cost: int = 2) -> str:
    """
    Position-aware correction of look-alike characters (e.g. 8 -> B in a letter slot).

    Tries every template of the same length, converts characters that sit in the
    wrong class, and picks the cheapest fix (fewest substitutions). If two
    different results tie, the string is returned untouched rather than guessed.
    """
    candidates = []
    for tpl in PLATE_TEMPLATES:
        if len(tpl) != len(s):
            continue
        out, cost, ok = [], 0, True
        for ch, slot in zip(s, tpl):
            if slot == 'D':
                if ch.isdigit():
                    out.append(ch)
                elif ch in LETTER_TO_DIGIT:
                    out.append(LETTER_TO_DIGIT[ch])
                    cost += 1
                else:
                    ok = False
                    break
            else:
                if ch.isalpha():
                    out.append(ch)
                elif ch in DIGIT_TO_LETTER:
                    out.append(DIGIT_TO_LETTER[ch])
                    cost += 1
                else:
                    ok = False
                    break
        if ok and cost <= max_cost:
            candidates.append((cost, ''.join(out)))

    if not candidates:
        return s

    candidates.sort()
    if (
        len(candidates) > 1
        and candidates[0][0] == candidates[1][0]
        and candidates[0][1] != candidates[1][1]
    ):
        return s  # ambiguous -> don't guess
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

def detect_category(text: str) -> OCRCategory:
    """
    Heuristically detects the OCR target category based on text patterns.

    If your pipeline already knows what the crop is (e.g. it came from a plate
    detector), skip this and pass the category to normalize_text directly.

    Args:
        text: The raw OCR output text.

    Returns:
        The detected OCRCategory.
    """
    if not text:
        return OCRCategory.GENERAL

    text = _clean_whitespace(text)
    if not text:
        return OCRCategory.GENERAL

    # Any CJK Unified Ideograph -> Chinese plate
    if _CJK_RE.search(text):
        return OCRCategory.CN_PLATE

    upper = text.upper()

    # Short pure-number text, optionally with MPH -> speed plaque ("35", "35 MPH")
    if re.fullmatch(r'\d{1,4}(\s*MPH)?', upper):
        return OCRCategory.SPEED_PLAQUE

    # Plate candidate: state names removed, 5-8 alphanumerics, at least one digit
    no_state = _strip_states(upper)
    compact = re.sub(r'[^A-Z0-9]', '', no_state)

    if 5 <= len(compact) <= 8 and any(c.isdigit() for c in compact):
        tokens = no_state.split(' ')
        # A multi-token string containing a real word ("EXIT 25", "SPEED LIMIT 65")
        # is a sign, not a plate.
        has_word = any(t.isalpha() and len(t) >= 4 for t in tokens)
        if not (len(tokens) > 1 and has_word):
            return OCRCategory.US_PLATE

    return OCRCategory.ROAD_SIGN


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

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

    # Multi-line text: read top-to-bottom, joining lines with a single space.
    text = _clean_whitespace(text)

    if category == OCRCategory.US_PLATE:
        # Uppercase, drop state names/slogans, drop separators, then fix
        # look-alike characters by position (e.g. 7A8C123 -> 7ABC123).
        text = text.upper()
        text = _strip_states(text)
        text = re.sub(r'[^A-Z0-9]', '', text)
        text = fix_by_template(text)

    elif category == OCRCategory.CN_PLATE:
        # Keep the leading province character + city letter, e.g. 京A12345.
        # Strip separators first so "京A·12345" and "京A 12345" both match.
        text = text.upper()
        text = re.sub(r'[\s·•.\-_]', '', text)

        match = re.search(r'([\u4e00-\u9fff])([A-Z])([A-Z0-9]{5,6})', text)
        if match:
            province, city, rest = match.groups()
            # Chinese plates never use I or O -> they are misread 1 and 0.
            rest = rest.replace('I', '1').replace('O', '0')
            text = f"{province}{city}{rest}"
        elif not re.match(r'^[\u4e00-\u9fff]', text):
            # No usable Chinese prefix: return the cleaned text as-is
            pass

    elif category == OCRCategory.ROAD_SIGN:
        # Road signs: retain full wording (e.g. SPEED LIMIT 65, STOP).
        text = text.upper().strip()

    elif category == OCRCategory.SPEED_PLAQUE:
        # Advisory speed plaques: digits only (e.g. 35). Do not append units.
        digits = re.findall(r'\d+', text)
        text = digits[0] if digits else ""

    return text
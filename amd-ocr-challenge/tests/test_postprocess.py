import unittest
from app.postprocess import normalize_text, detect_category, OCRCategory

class TestPostprocess(unittest.TestCase):
    """Comprehensive test suite for OCR text normalization and category detection."""

    def test_us_plate(self):
        """Tests that US state names and slogans are removed regardless of case."""
        self.assertEqual(normalize_text("CALIFORNIA 7ABC123", OCRCategory.US_PLATE), "7ABC123")
        self.assertEqual(normalize_text("NEW YORK EMPIRE STATE 123XYZ", OCRCategory.US_PLATE), "123XYZ")
        self.assertEqual(normalize_text("TEXAS 456DEF", OCRCategory.US_PLATE), "456DEF")
        self.assertEqual(normalize_text("California 7abc123", OCRCategory.US_PLATE), "7abc123")
        self.assertEqual(normalize_text("7ABC123", OCRCategory.US_PLATE), "7ABC123")

    def test_cn_plate(self):
        """Tests that Chinese province characters and letter prefixes are preserved."""
        self.assertEqual(normalize_text("京A 12345", OCRCategory.CN_PLATE), "京A12345")
        self.assertEqual(normalize_text("沪B 67890", OCRCategory.CN_PLATE), "沪B67890")
        self.assertEqual(normalize_text("京A12345", OCRCategory.CN_PLATE), "京A12345")
        self.assertEqual(normalize_text("京", OCRCategory.CN_PLATE), "京")

    def test_road_sign(self):
        """Tests that road sign wording is retained and converted to uppercase."""
        self.assertEqual(normalize_text("stop", OCRCategory.ROAD_SIGN), "STOP")
        self.assertEqual(normalize_text("Speed Limit 65", OCRCategory.ROAD_SIGN), "SPEED LIMIT 65")
        self.assertEqual(normalize_text("ROAD WORK AHEAD", OCRCategory.ROAD_SIGN), "ROAD WORK AHEAD")

    def test_speed_plaque(self):
        """Tests that only digits are extracted from advisory speed plaques."""
        self.assertEqual(normalize_text("35 MPH", OCRCategory.SPEED_PLAQUE), "35")
        self.assertEqual(normalize_text("Speed 45", OCRCategory.SPEED_PLAQUE), "45")
        self.assertEqual(normalize_text("No digits here", OCRCategory.SPEED_PLAQUE), "")

    def test_general(self):
        """Tests that general text is trimmed and multi-line text is joined."""
        self.assertEqual(normalize_text("Hello\nWorld", OCRCategory.GENERAL), "Hello World")
        self.assertEqual(normalize_text("  Too   many    spaces  ", OCRCategory.GENERAL), "Too many spaces")
        self.assertEqual(normalize_text("", OCRCategory.GENERAL), "")
        self.assertEqual(normalize_text("   ", OCRCategory.GENERAL), "")

    def test_detect_category(self):
        """Tests the heuristic category detection logic."""
        self.assertEqual(detect_category("京A 12345"), OCRCategory.CN_PLATE)
        self.assertEqual(detect_category("CALIFORNIA 7ABC123"), OCRCategory.US_PLATE)
        self.assertEqual(detect_category("35"), OCRCategory.SPEED_PLAQUE)
        self.assertEqual(detect_category("STOP SIGN"), OCRCategory.ROAD_SIGN)
        self.assertEqual(detect_category(""), OCRCategory.GENERAL)

if __name__ == "__main__":
    unittest.main()

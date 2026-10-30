"""Unit tests for conservative, shape-aware OCR preprocessing and ranking."""
import cv2
import numpy as np

from backend.models import BoundingBox, OCRWord
from backend.services.vision_service import preprocess_ocr_candidates, select_ocr_candidate


def word(text, confidence=.8):
    return OCRWord(text, confidence, BoundingBox(0, 0, 10, 10))


def test_preprocessing_keeps_original_and_upscales_small_text():
    image = np.full((18, 52), 255, np.uint8)
    cv2.putText(image, "x!=3", (1, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1, cv2.LINE_AA)
    variants = preprocess_ocr_candidates(image, "decision")
    assert variants
    assert variants[0].shape[0] >= image.shape[0] * 2
    assert any(np.array_equal(v, cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)) for v in variants)


def test_shape_aware_preprocessing_returns_complementary_candidates():
    image = np.full((80, 160), 245, np.uint8)
    cv2.putText(image, "a % 2 == 0", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, .8, 25, 2, cv2.LINE_AA)
    process = preprocess_ocr_candidates(image, "process")
    decision = preprocess_ocr_candidates(image, "decision")
    assert len(process) >= 3 and len(decision) >= 3
    assert all(v.ndim == 2 for v in decision)


def test_candidate_selection_preserves_programming_symbols_and_confidence():
    candidates = [("x = 3", [word("x", .86), word("=", .86), word("3", .86)]),
                  ("x == 3", [word("x", .91), word("==", .91), word("3", .91)]),
                  ("x 11 3", [word("x", .55), word("11", .55), word("3", .55)])]
    text, words, plausible = select_ocr_candidate(candidates, "decision")
    assert text == "x == 3"
    assert words
    assert any(item["text"] == "x = 3" for item in plausible)


def test_low_confidence_retains_multiple_plausible_readings():
    candidates = [("value % 2", [word("value", .4), word("%", .4), word("2", .4)]),
                  ("value 7 2", [word("value", .42), word("7", .42), word("2", .42)]),
                  ("value % z", [word("value", .39), word("%", .39), word("z", .39)])]
    _, _, plausible = select_ocr_candidate(candidates, "process")
    assert len(plausible) >= 2

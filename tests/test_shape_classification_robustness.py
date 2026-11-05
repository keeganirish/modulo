"""Geometry regression tests for imperfect and ambiguous flowchart nodes."""
import cv2
import numpy as np

from backend.services.vision_service import classify_shape


def _largest_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def test_rotated_process_rectangle_remains_process():
    image = np.zeros((240, 300), np.uint8)
    points = cv2.boxPoints(((150, 120), (150, 68), 27)).astype(np.int32)
    cv2.fillPoly(image, [points], 255)
    assert classify_shape(_largest_contour(image), image.shape[1]) == "process"


def test_rounded_terminal_is_classified_as_terminal():
    image = np.zeros((180, 300), np.uint8)
    cv2.rectangle(image, (70, 45), (230, 135), 255, -1)
    cv2.ellipse(image, (70, 90), (28, 45), 0, 90, 270, 255, -1)
    cv2.ellipse(image, (230, 90), (28, 45), 0, -90, 90, 255, -1)
    assert classify_shape(_largest_contour(image), image.shape[1]) == "terminal"


def test_small_circle_is_connector():
    image = np.zeros((180, 240), np.uint8)
    cv2.circle(image, (120, 90), 20, 255, -1)
    assert classify_shape(_largest_contour(image), image.shape[1]) == "connector"


def test_ambiguous_triangle_is_not_forced_into_known_type():
    image = np.zeros((180, 240), np.uint8)
    cv2.fillPoly(image, [np.array([[120, 35], [195, 140], [45, 140]], np.int32)], 255)
    assert classify_shape(_largest_contour(image), image.shape[1]) == "unknown"


def test_low_resolution_parallelogram_keeps_data_class():
    image = np.zeros((100, 180), np.uint8)
    cv2.fillPoly(image, [np.array([[35, 25], [135, 25], [145, 70], [45, 70]], np.int32)], 255)
    assert classify_shape(_largest_contour(image), image.shape[1]) == "data"


def test_filled_colored_nodes_are_not_replaced_by_internal_text_holes():
    """Regression: white glyph holes inside solid colored nodes are not nodes."""
    from pathlib import Path
    import cv2
    import numpy as np
    from backend.services.vision_service import shape_contours, classify_shape

    fixture = Path(__file__).parent / "fixtures" / "flowcharts" / "filled_blue_decision.png"
    page = cv2.imread(str(fixture))
    assert page is not None
    page = cv2.resize(page, (2000, round(page.shape[0] * 2000 / page.shape[1])),
                      interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = shape_contours(binary, gray, page)
    boxes = [cv2.boundingRect(contour) for contour in contours]

    # Six real nodes; do not emit tiny contours from text inside the decision.
    assert len(contours) == 6
    assert sum(width > 700 and height > 350 for _, _, width, height in boxes) == 6
    kinds = [classify_shape(contour, page.shape[1]) for contour in contours]
    assert kinds.count("terminal") == 2
    assert kinds.count("decision") == 1
    assert kinds.count("input") + kinds.count("data") >= 1


def test_output_instruction_recognizes_punctuation_after_print_keyword():
    from backend.services.vision_service import infer_instruction_kind

    assert infer_instruction_kind("unknown", 'Print: "Number is odd"') == (
        "output", 'Output "Number is odd"'
    )

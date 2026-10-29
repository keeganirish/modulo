"""Deterministic contour geometry and Tesseract OCR; no sample-specific rules."""
from __future__ import annotations
from collections import OrderedDict
from threading import Lock
import math
import re

import cv2
import numpy as np
import pytesseract

from backend.models import BoundingBox, Node, OCRWord, PipelineContext

OCR_LOCK = Lock()  # pytesseract's executable setting is process-global.


def shape_contours(binary: np.ndarray, grayscale: np.ndarray | None = None, color_image: np.ndarray | None = None) -> list[np.ndarray]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    candidates = []
    page_area = binary.size
    for index, contour in enumerate(contours):
        parent, depth = hierarchy[0, index, 3], 0
        while parent != -1:
            depth += 1
            parent = hierarchy[0, parent, 3]
        if depth % 2 != 1:  # Enclosed white regions survive even when arrows touch borders.
            continue
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if area < max(160, page_area * .00004) or min(width, height) < 30:
            continue
        if area > page_area * .85 or area / (width * height) < .38:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0 or area / hull_area < .92:
            continue
        candidates.append(contour)
    # Ignore nested enclosures: they are text holes or an unsupported nested diagram.
    candidates.sort(key=lambda contour: (-cv2.contourArea(contour), cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0]))
    retained = []
    for contour in candidates:
        point = tuple(float(value) for value in contour[0, 0])
        if not any(cv2.pointPolygonTest(outer, point, False) > 0 for outer in retained):
            retained.append(contour)
    retained = sorted(retained, key=lambda contour: (cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0], cv2.boundingRect(contour)[2]))
    if grayscale is None:
        return retained
    # Some diagram tools export filled nodes without an outline. Saturated
    # fills can be separated from black arrows/text in HSV; neutral gray fills
    # use a mid-value mask so black connectors remain excluded.
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV) if color_image is not None else cv2.cvtColor(cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    saturated = hsv[:, :, 1] > 25
    neutral = (grayscale > 80) & (grayscale < 235) & (hsv[:, :, 1] < 35)
    filled = np.where(saturated | neutral, 255, 0).astype(np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = filled.size
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour); x,y,width,height=cv2.boundingRect(contour)
        if area < max(300, page_area*.00008) or min(width,height)<35 or area/(width*height)<.42 or area > page_area*.4:
            continue
        candidates.append(contour)
    for contour in candidates:
        point=tuple(float(value) for value in contour[0,0])
        if not any(cv2.pointPolygonTest(existing, point, False) >= 0 or cv2.pointPolygonTest(contour, tuple(float(v) for v in existing[0,0]), False) >= 0 for existing in retained):
            retained.append(contour)
    return sorted(retained, key=lambda contour:(cv2.boundingRect(contour)[1],cv2.boundingRect(contour)[0],cv2.boundingRect(contour)[2]))


def classify_shape(contour: np.ndarray, page_width: int) -> str:
    x, y, width, height = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, .025 * perimeter, True).reshape(-1, 2)
    area = cv2.contourArea(contour)
    circularity = 4 * math.pi * area / max(1, perimeter * perimeter)
    if .72 < width / height < 1.35 and max(width, height) < page_width * .12 and circularity > .76:
        return "connector"
    fine = cv2.approxPolyDP(contour, .01 * perimeter, True)
    if len(fine) > 4 and area / (width * height) > .95:
        return "terminal"
    if len(polygon) == 4:
        # A diamond's corners are side midpoints, not bounding-box corners.
        nx = (polygon[:, 0] - x) / width
        ny = (polygon[:, 1] - y) / height
        if sum(.25 < a < .75 or .25 < b < .75 for a, b in zip(nx, ny)) == 4 and area / (width * height) < .7:
            return "decision"
        points = sorted(polygon.tolist(), key=lambda point: (point[1], point[0]))
        top, bottom = sorted(points[:2]), sorted(points[2:])
        side_shift = (abs(top[0][0] - bottom[0][0]) + abs(top[1][0] - bottom[1][0])) / (2 * width)
        return "data" if side_shift > .065 else "process"
    if len(polygon) == 6:
        # ISO preparation/iteration hexagon has pointed left/right, flat top/bottom.
        horizontal = sum(abs(int(polygon[(i + 1) % 6, 1]) - int(polygon[i, 1])) < height * .1 for i in range(6))
        if horizontal >= 2 and area / (width * height) < .95:
            return "iteration"
    if len(fine) > 4 and .60 < area / (width * height) < .99:
        return "terminal"
    return "unknown"


def read_words(image: np.ndarray, context: PipelineContext, psm: int, origin: tuple[int, int] = (0, 0)) -> tuple[str, list[OCRWord]]:
    data = pytesseract.image_to_data(image, lang=context.settings.OCR_LANGUAGE,
        config=f"--psm {psm} -c preserve_interword_spaces=1", output_type=pytesseract.Output.DICT,
        timeout=context.settings.OCR_TIMEOUT_SECONDS)
    lines = OrderedDict()
    words = []
    for index, value in enumerate(data["text"]):
        text = value.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < 0:
            continue
        bbox = BoundingBox(origin[0] + int(data["left"][index]), origin[1] + int(data["top"][index]),
                           int(data["width"][index]), int(data["height"][index]))
        words.append(OCRWord(text, confidence / 100, bbox))
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        lines.setdefault(key, []).append(text)
    return "\n".join(" ".join(line) for line in lines.values()), words


def read_best_words(image: np.ndarray, context: PipelineContext, origin: tuple[int, int] = (0, 0)) -> tuple[str, list[OCRWord]]:
    """Try normal and adaptive contrast; colored fills often need local thresholding."""
    variants = [image]
    if min(image.shape[:2]) >= 35:
        variants.append(cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 31, 8))
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.extend((otsu, 255 - otsu))
    candidates=[]
    for variant in variants:
        for psm in (6, 11):
            text, words = read_words(variant, context, psm, origin)
            score = sum(max(0.0, word.confidence) * len(word.text) for word in words)
            score += 2.0 if psm == 6 else 0.0
            candidates.append((score, text, words))
    _, text, words = max(candidates, key=lambda candidate: candidate[0], default=(0.0, "", []))
    return text, words


class VisionService:
    name = "vision"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.binary is None or context.grayscale is None:
            context.diagnose("error", "MISSING_PAGE", "No normalized image is available.", "Run intake before recognition.")
            return context
        binary, gray = context.binary, context.grayscale
        contours = shape_contours(binary, gray, context.page_image)
        if not contours:
            context.diagnose("error", "NO_SHAPES_DETECTED", "No flowchart shapes could be recognized.", "Use clearly separated outlined or solid-color nodes on a light background, with legible text and visible arrowheads.")
            return context
        if len(contours) > context.settings.MAX_NODES:
            context.diagnose("error", "TOO_MANY_SHAPES", "This diagram exceeds the recognition limit.", f"Use at most {context.settings.MAX_NODES} non-overlapping shapes on one page.")
            return context
        outside = gray.copy()
        removal = np.zeros_like(binary)
        # Serialize only OCR configuration/calls, not all Flask requests.
        with OCR_LOCK:
            previous = pytesseract.pytesseract.tesseract_cmd
            if context.settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = context.settings.TESSERACT_CMD
            try:
                for index, contour in enumerate(contours, 1):
                    x, y, width, height = cv2.boundingRect(contour)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    shifted = contour - np.array([[[x, y]]])
                    cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)
                    mask = cv2.erode(mask, np.ones((11, 11), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
                    geometry_kind = classify_shape(contour, gray.shape[1])
                    left, top = (int(width * .20), int(height * .20)) if geometry_kind == "connector" else (int(width * .10), int(height * .15))
                    crop = gray[y + top:y + height - top, x + left:x + width - left]
                    # The contour bbox is already isolated from neighboring
                    # nodes. Keeping the full crop prevents the fill mask
                    # from erasing anti-aliased glyphs on solid-color nodes.
                    text, words = (read_words(crop, context, 10, (x + left, y + top))
                                   if geometry_kind == "connector" else read_best_words(crop, context, (x + left, y + top)))
                    text = re.sub(r"^\s*[/|\\]+\s*|\s*[/|\\]+\s*$", "", text).strip()
                    text = " ".join(text.split()).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
                    kind = geometry_kind
                    flags = []
                    if kind == "data":
                        lowered = re.sub(r"^[^a-z]+", "", text.lower().strip())
                        first_word = re.split(r"\s+", lowered, maxsplit=1)[0] if lowered else ""
                        kind = "input" if first_word in {"input", "read", "enter"} else "output" if first_word in {"output", "display", "show", "print"} else "unknown"
                        if kind == "output" and first_word != "output":
                            text = "Output " + text.split(None, 1)[1]
                        if kind == "unknown":
                            flags.append("UNRECOGNIZED_DATA_TEXT")
                    elif kind == "decision" and text.lower().startswith("if "):
                        text = text.split(None, 1)[1]
                    if kind == "unknown":
                        flags.append("UNSUPPORTED_SHAPE")
                    total = sum(len(word.text) for word in words)
                    confidence = sum(word.confidence * len(word.text) for word in words) / total if total else 0.0
                    polygon = cv2.approxPolyDP(contour, .002 * cv2.arcLength(contour, True), True).reshape(-1, 2)
                    node = Node(f"n{index}", kind, text, BoundingBox(x, y, width, height), confidence, flags,
                                [(int(a), int(b)) for a, b in polygon], words)
                    context.elements.append(node)
                    cv2.drawContours(removal, [contour], -1, 255, cv2.FILLED)
                # Filled-node contours already cover their interiors; do not
                # expand them over arrowheads that terminate at the boundary.
                hsv_page = cv2.cvtColor(context.page_image, cv2.COLOR_BGR2HSV)
                colored_fill = float(np.mean(hsv_page[:, :, 1] > 25)) > .02
                removal = cv2.dilate(removal, np.ones((1 if colored_fill else 19, 1 if colored_fill else 19), np.uint8))
                outside[removal > 0] = 255
                # Long strokes confuse sparse OCR and can become giant false words.
                outside_ink = 255 - outside
                horizontal_lines = cv2.morphologyEx(outside_ink, cv2.MORPH_OPEN, np.ones((1, 45), np.uint8))
                vertical_lines = cv2.morphologyEx(outside_ink, cv2.MORPH_OPEN, np.ones((45, 1), np.uint8))
                lines = cv2.dilate(cv2.max(horizontal_lines, vertical_lines), np.ones((5, 5), np.uint8))
                outside[lines > 0] = 255
                _, words = read_words(outside, context, 11)
                words = [word for word in words if word.bbox.height < min(120, gray.shape[0] * .12)
                         and word.bbox.width < gray.shape[1] * .25 and word.bbox.width / max(1, word.bbox.height) < 12]
                context.edge_words = sorted(words, key=lambda word: (word.bbox.y, word.bbox.x, word.text))
                # A word connected to a long stroke is probably an OCR reading of
                # an arrowhead (often "7"), not independent edge-label text.
                original_strokes = binary.copy()
                original_strokes[removal > 0] = 0
                _, components, stats, _ = cv2.connectedComponentsWithStats(original_strokes, connectivity=8)
                filtered = []
                for word in context.edge_words:
                    bx, by, bw, bh = word.bbox.to_list()
                    associated = np.unique(components[max(0, by - 1):by + bh + 1, max(0, bx - 1):bx + bw + 1])
                    on_stroke = False
                    for component in associated:
                        if component == 0:
                            continue
                        sx, sy, sw, sh, _ = stats[component]
                        if max(sw, sh) > max(bw, bh) + 35 and (sx < bx - 15 or sy < by - 15 or sx + sw > bx + bw + 15 or sy + sh > by + bh + 15):
                            on_stroke = True
                            break
                    if not on_stroke:
                        filtered.append(word)
                context.edge_words = filtered
            except pytesseract.TesseractNotFoundError:
                context.diagnose("error", "OCR_UNAVAILABLE", "Tesseract OCR could not be found.", "Install Tesseract separately or set TESSERACT_CMD to its executable.")
            except (pytesseract.TesseractError, RuntimeError) as error:
                code = "OCR_TIMEOUT" if "timeout" in str(error).lower() else "OCR_FAILED"
                context.diagnose("error", code, "OCR could not finish reading the diagram.", "Check the OCR language installation and try a smaller, clearer export.")
            finally:
                pytesseract.pytesseract.tesseract_cmd = previous
        strokes = binary.copy()
        strokes[removal > 0] = 0
        # Remove only plausible text boxes, not arbitrary line-like OCR artifacts.
        for word in context.edge_words:
            if any(character.isalnum() for character in word.text):
                box = word.bbox
                strokes[max(0, box.y - 2):box.y + box.height + 2, max(0, box.x - 2):box.x + box.width + 2] = 0
        context.stroke_mask = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return context


def run(context: PipelineContext) -> PipelineContext:
    return VisionService().run(context)

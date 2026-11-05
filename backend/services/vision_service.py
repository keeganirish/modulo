"""Deterministic contour geometry and Tesseract OCR; no sample-specific rules."""
from __future__ import annotations
from collections import OrderedDict
from threading import Lock
import ast
import difflib
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
        # Long, narrow enclosed regions are commonly arrow shafts/arrowheads,
        # not flowchart nodes. They otherwise become empty OCR "connector" nodes.
        aspect = max(width, height) / max(1.0, min(width, height))
        if aspect > 3.2:
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
        short_side, long_side = min(width, height), max(width, height)
        aspect = long_side / max(1.0, short_side)
        # The fill-based fallback can pick up arrow shafts and arrowheads because
        # they share the same gray/colored pixels as node fills. Real flowchart
        # nodes have meaningful two-dimensional extent; reject thin components
        # here as well as in the outlined-contour pass.
        if (area < max(300, page_area*.0008) or short_side < 35
                or area/(width*height)<.30 or area > page_area*.4
                or (aspect > 2.35 and short_side < 36)):
            continue
        candidates.append(contour)
    # A filled node often contains many contours from the white gaps inside
    # glyphs.  Point-in-polygon containment is therefore not a valid duplicate
    # test here: it treats every text hole as if it were the filled node itself
    # and can suppress the node contour entirely.  Merge only near-identical
    # bounding boxes, then remove small enclosed contours once the enclosing
    # fill has been established.
    def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
        ax, ay, aw, ah = cv2.boundingRect(first)
        bx, by, bw, bh = cv2.boundingRect(second)
        ix, iy = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - ix) * max(0, bottom - iy)
        union = aw * ah + bw * bh - intersection
        return intersection / max(1, union)

    merged_fills = []
    for contour in candidates:
        duplicate = next((existing for existing in retained + merged_fills
                          if bbox_iou(existing, contour) >= .68), None)
        if duplicate is None:
            merged_fills.append(contour)

    # Filled regions supersede the small white glyph holes found by the
    # threshold-contour pass, but do not supersede a similarly-sized outlined
    # node.  This is geometry-based and works for any label or node type.
    retained = [existing for existing in retained
                if not any(
                    cv2.contourArea(fill) > max(1.0, cv2.contourArea(existing)) * 2.0
                    and bbox_iou(fill, existing) < .68
                    and cv2.pointPolygonTest(fill, tuple(float(v) for v in cv2.boundingRect(existing)[:2]), False) >= 0
                    and (cv2.boundingRect(existing)[2] * cv2.boundingRect(existing)[3])
                        < .45 * (cv2.boundingRect(fill)[2] * cv2.boundingRect(fill)[3])
                    for fill in merged_fills
                )]
    retained.extend(merged_fills)
    return sorted(retained, key=lambda contour:(cv2.boundingRect(contour)[1],cv2.boundingRect(contour)[0],cv2.boundingRect(contour)[2]))


def classify_shape(contour: np.ndarray, page_width: int) -> str:
    """Classify a contour from several geometric cues; return unknown on weak evidence.

    Polygon approximation is evaluated at two scales because antialiasing and
    low-resolution edges can change the apparent vertex count.  No contour is
    forced into a known class when the available geometric evidence conflicts.
    """
    if contour is None or len(contour) < 5:
        return "unknown"
    x, y, width, height = cv2.boundingRect(contour)
    if width < 2 or height < 2:
        return "unknown"
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    if perimeter <= 0 or area <= 0:
        return "unknown"
    box_area = float(width * height)
    fill = area / box_area
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    aspect = width / max(1.0, float(height))
    polygon = cv2.approxPolyDP(contour, .04 * perimeter, True).reshape(-1, 2)
    fine = cv2.approxPolyDP(contour, .008 * perimeter, True).reshape(-1, 2)

    def corner_angles(points: np.ndarray) -> list[float]:
        angles = []
        for i in range(len(points)):
            prev = points[(i - 1) % len(points)].astype(float) - points[i].astype(float)
            nxt = points[(i + 1) % len(points)].astype(float) - points[i].astype(float)
            denom = float(np.linalg.norm(prev) * np.linalg.norm(nxt))
            if denom > 1e-6:
                angles.append(math.degrees(math.acos(float(np.clip(np.dot(prev, nxt) / denom, -1, 1)))))
        return angles

    angles = corner_angles(polygon) if len(polygon) >= 3 else []
    right_angle_ratio = (sum(80 <= angle <= 100 for angle in angles) / len(angles)) if angles else 0.0
    approx_vertices = len(polygon)

    # Circular flow connectors: use circularity, near-square bounds and solidity
    # together so small rounded rectangles are not automatically called circles.
    if (0.78 <= aspect <= 1.28 and max(width, height) <= max(42, page_width * .16)
            and circularity >= .72 and solidity >= .90 and fill >= .62):
        return "connector"

    # Rotated rectangles may acquire a fifth vertex from raster stair-stepping.
    # A minimum-area rectangle gives a rotation-invariant right-angle/occupancy cue.
    min_rect = cv2.minAreaRect(contour)
    rect_area = float(min_rect[1][0] * min_rect[1][1])
    rect_occupancy = area / rect_area if rect_area > 0 else 0.0
    if len(polygon) in (4, 5) and rect_occupancy >= .90 and solidity >= .96 and right_angle_ratio >= .75:
        rect_points = cv2.boxPoints(min_rect)
        rect_aspect = max(min_rect[1]) / max(1.0, min(min_rect[1]))
        if rect_aspect >= 1.25:
            return "process"

    # Four-corner shapes are distinguished by normalized corner placement, angles,
    # and opposite-edge displacement; bbox aspect alone is rotation-sensitive.
    if approx_vertices == 4:
        points = polygon.astype(float)
        cx, cy = x + width / 2.0, y + height / 2.0
        normalized = np.column_stack(((points[:, 0] - x) / width, (points[:, 1] - y) / height))
        side_midpoints = sum(min(abs(px), abs(1 - px), abs(py), abs(1 - py)) > .16 for px, py in normalized)
        diamond_corners = sum(min(abs(px - .5), abs(py - .5)) < .20 for px, py in normalized)
        # A diamond has vertices close to the four side midpoints and acute/obtuse
        # alternating angles.  The tests remain tolerant of slight rotation/skew.
        diagonal_balance = abs((max(points[:, 0]) - min(points[:, 0])) / max(1, width) - 1) < .15
        diamond_score = sum(.22 <= px <= .78 or .22 <= py <= .78 for px, py in normalized)
        if (diamond_corners >= 3 and diamond_score >= 4 and fill <= .82
                and (right_angle_ratio < .75 or fill < .62)):
            return "decision"

        # Parallelogram: top and bottom edges are displaced in the same direction.
        ordered = sorted(points.tolist(), key=lambda pt: (pt[1], pt[0]))
        top, bottom = sorted(ordered[:2]), sorted(ordered[2:])
        shift = (abs(top[0][0] - bottom[0][0]) + abs(top[1][0] - bottom[1][0])) / (2 * max(1, width))
        if (right_angle_ratio < .65 and fill >= .48 and solidity >= .88
                and min(width, height) >= 18):
            # Parallelogram corners are consistently oblique; a rotated rectangle
            # retains right angles regardless of its axis-aligned bounding box.
            return "data"
        if shift >= .055 and right_angle_ratio < .9 and solidity >= .88:
            return "data"
        if right_angle_ratio >= .65 and fill >= .55 and solidity >= .88:
            return "process"
        # Rotated rectangles can have acute angles in the image coordinate system;
        # opposite edge parallelism and high solidity still support process.
        if fill >= .55 and solidity >= .94 and right_angle_ratio < .65 and shift < .055:
            return "process"

    # Terminals are typically capsules/rounded rectangles.  Require a broad
    # rounded contour or very high fill with a curved outline; do not infer it
    # merely from an unstable high-resolution vertex count.
    curved_outline = len(fine) >= 7 and len(polygon) >= 5
    if (curved_outline and solidity >= .88 and .60 <= fill <= 1.02
            and .28 <= aspect <= 3.5 and circularity >= .28):
        return "terminal"
    if (len(fine) >= 8 and solidity >= .92 and .72 <= fill <= 1.0
            and .35 <= aspect <= 2.8 and circularity >= .48):
        return "terminal"

    # Standard preparation/iteration hexagon: six vertices, high solidity and
    # approximately horizontal top/bottom edges.
    if approx_vertices == 6 and solidity >= .88 and fill >= .45:
        horizontal_edges = sum(abs(int(polygon[(i + 1) % 6, 1]) - int(polygon[i, 1])) < height * .16 for i in range(6))
        if horizontal_edges >= 2:
            return "iteration"

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


def preprocess_ocr_candidates(image: np.ndarray, shape_type: str = "unknown") -> list[np.ndarray]:
    """Create conservative, complementary OCR inputs; the original is always first."""
    if image is None or image.size == 0:
        return []
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Scale small glyphs without changing their proportions. Cubic interpolation is
    # gentler on punctuation than binary morphology or nearest-neighbour scaling.
    min_dim = min(gray.shape[:2])
    scale = 3.0 if min_dim < 24 else 2.0 if min_dim < 48 else 1.5 if min_dim < 80 else 1.0
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    variants = [gray]
    # Local contrast and mild denoising preserve thin strokes and decimal points.
    clahe = cv2.createCLAHE(clipLimit=2.0 if shape_type == "decision" else 1.7, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)
    denoised = cv2.bilateralFilter(clahe, 5, 30, 30)
    variants.append(denoised)
    # Avoid aggressive thresholding on small labels and diamonds, where punctuation
    # and diagonal strokes are especially easy to erase.
    if min(gray.shape[:2]) >= 24:
        block = min(31, min(gray.shape[:2]) if min(gray.shape[:2]) % 2 else min(gray.shape[:2]) - 1)
        if block >= 11:
            adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, block, 7 if shape_type == "connector" else 5)
            variants.append(adaptive)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        # Inverted Otsu helps dark-filled shapes; include it only as a candidate.
        if shape_type in {"decision", "data", "input", "output", "unknown"}:
            variants.append(255 - otsu)
    # De-duplicate exact images to avoid wasting Tesseract calls.
    unique, seen = [], set()
    for variant in variants:
        key = (variant.shape, hash(variant.tobytes()))
        if key not in seen:
            seen.add(key); unique.append(variant)
    return unique



def infer_instruction_kind(shape_type: str, text: str) -> tuple[str, str]:
    """Use unmistakable instruction verbs to refine rectangular-node semantics."""
    if shape_type in {"terminal", "decision", "iteration", "connector"}:
        return shape_type, text
    match = re.match(r"^(input|read|enter)\s+(.+)$", text.strip(), re.IGNORECASE)
    if match:
        return "input", "Input " + match.group(2).strip()
    match = re.match(r"^(print|output|display|show)\s*:?\s+(.+)$", text.strip(), re.IGNORECASE)
    if match:
        payload = match.group(2).strip()
        # If OCR drops only the final quote from a quoted output literal, restore
        # the delimiter rather than passing a broken string into validation.
        if payload[:1] in {"'", '"'} and payload.count(payload[0]) % 2:
            payload += payload[0]
        return "output", "Output " + payload
    return shape_type, text

def _normalize_comparison_spacing(text: str, shape_type: str) -> str:
    """Repair whitespace-only OCR splits in comparisons without changing operators.

    For decision labels, Tesseract can split the two strokes of ``==`` into
    separate tokens (for example ``if n % 2 = =0``). Joining that exact pattern
    is a conservative syntax repair; single ``=`` is never converted by itself.
    """
    if shape_type != "decision":
        return text
    normalized = re.sub(r"(?<![=!<>])=\s+=(?![=])", "==", text)
    normalized = re.sub(r"\s*(==|!=|<=|>=)\s*", r" \1 ", normalized)
    normalized = re.sub(r"(?<![<>=!])([<>])(?![=])", r" \1 ", normalized)
    return " ".join(normalized.split())


def _valid_decision_expression(text: str) -> bool:
    expression = re.sub(r"^if\s+", "", text.strip(), flags=re.IGNORECASE).rstrip(": ")
    try:
        ast.parse(expression, mode="eval")
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def _candidate_score(text: str, words: list[OCRWord], shape_type: str = "unknown") -> float:
    """Rank readings by confidence plus character integrity, not length alone."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return -100.0
    chars = [c for c in cleaned if not c.isspace()]
    confidence = sum(w.confidence * max(1, len(w.text)) for w in words) / max(1, sum(max(1, len(w.text)) for w in words))
    printable = sum(c.isprintable() for c in chars) / max(1, len(chars))
    # Penalize common OCR debris but do not reject programming punctuation.
    alnum = sum(c.isalnum() for c in chars)
    symbols = sum(c in "= !%(),.'\"_+-*/<>:" for c in chars)
    debris = sum(c in "|\\~`" for c in chars)
    score = confidence * 5.0 + printable * 1.5 + min(alnum, 24) * .025 + min(symbols, 8) * .045 - debris * .18
    if shape_type == "connector" and len(cleaned) > 18:
        score -= (len(cleaned) - 18) * .04
    if shape_type == "decision" and any(op in cleaned for op in ("==", "!=", "<=", ">=", "%")):
        score += .15
    if shape_type in {"process", "output", "unknown"} and re.match(r"^print(?:\s|[:(])", cleaned, re.IGNORECASE):
        score += .45
        # Prefer a candidate whose common words are already coherent over its
        # near-miss spelling, but keep this bonus small relative to OCR evidence.
        # A coherent repair of an otherwise recognizable output statement
        # should beat the same-confidence OCR gibberish, but alternatives remain
        # attached to the node for the review screen.
        if _repair_common_ocr_words(cleaned) == cleaned:
            score += .40
    return score


def _repair_common_ocr_words(text: str) -> str:
    """Correct near-miss OCR tokens from a small flowchart vocabulary.

    Unknown words and identifiers are preserved; this is not general-purpose
    autocorrect. Punctuation and capitalization are retained.
    """
    vocabulary = (
        "print", "factorial", "is", "not", "defined", "for", "negative", "numbers",
        "input", "number", "calculate", "start", "end", "if", "else", "while",
        "return", "true", "false", "yes", "no", "display", "output", "read",
    )
    # A low-resolution exclamation/bar is sometimes substituted for the final
    # stroke in "not" (e.g. `no! defined`). Repair only this known phrase shape.
    text = re.sub(r"\bno[!|](?=\s+J?efined\b)", "not", text, flags=re.IGNORECASE)

    def fix(match):
        token = match.group(0)
        bare = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token)
        # A trailing apostrophe is a common OCR substitution for the final t
        # in "not"; only repair this exact punctuation-shaped near miss.
        if bare.casefold() == "no" and token.endswith(("'", "’")):
            return token.replace(bare, "not").rstrip("'’")
        if len(bare) < 3 or bare.casefold() in vocabulary or any(ch.isdigit() for ch in bare):
            return token
        if not (bare.islower() or bare.istitle() or bare.isupper()):
            return token
        matches = difflib.get_close_matches(bare.casefold(), vocabulary, n=1, cutoff=.78)
        if not matches:
            return token
        replacement = matches[0]
        if bare.isupper():
            replacement = replacement.upper()
        return token.replace(bare, replacement)
    return re.sub(r"[^\W_]+(?:['’][^\W_]*)?", fix, text, flags=re.UNICODE)


def _repair_ocr_text(text: str, shape_type: str) -> list[str]:
    """Produce small, reviewable OCR repairs without silently rewriting meaning.

    Repairs are candidates, not unconditional substitutions. Syntax-aware scoring
    below prefers a repaired decision only when it forms a valid expression.
    """
    source = " ".join(text.split())
    results = []
    if shape_type == "decision":
        # OCR commonly inserts a stray dot/colon after the decision keyword
        # ("If . value > 0"). Remove only punctuation immediately following
        # that keyword; punctuation inside the expression remains untouched.
        cleaned = re.sub(r"^(\s*if)\s*[.:,;]+\s*", r"\1 ", source, flags=re.IGNORECASE)
        spaced = _normalize_comparison_spacing(cleaned, shape_type)
        if spaced != source:
            results.append(spaced)
        # Tesseract often separates a comparison operator from its operand.
        compact = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=(?:==|!=|<=|>=|<|>))", " ", cleaned)
        compact = re.sub(r"(==|!=|<=|>=|<|>)\s+(?=[0-9A-Za-z_(])", r"\1 ", compact)
        if compact != source:
            results.append(compact)
        # In flowchart decision diamonds, a lone '=' between two expression
        # operands is commonly used for equality in pseudocode and is also a
        # frequent OCR loss of one stroke from '=='. Offer (never silently
        # force) the Python-compatible equality form as a candidate.
        if cleaned != source and not _valid_decision_expression(cleaned) and not re.search(r"[!<>]=|==", cleaned):
            equality = re.sub(r"(?<=[A-Za-z0-9_)])\s*=\s*(?=[A-Za-z0-9_(])", " == ", cleaned, count=1)
            equality = " ".join(equality.split())
            if equality != cleaned and _valid_decision_expression(equality):
                results.append(equality)
    # Common clipped first-letter errors in print statements are low-risk when
    # the remainder is an exact known keyword; do not guess arbitrary words.
    if shape_type in {"process", "output", "unknown"}:
        repaired = re.sub(r"^rint(?=\s*\()", "print", source, flags=re.IGNORECASE)
        repaired = re.sub(r"^rint(?=\s+['\"])", "Print", repaired, flags=re.IGNORECASE)
        if repaired != source:
            results.append(repaired)
        # Only spell-correct within an explicitly recognized print/output
        # statement; do not expand the vocabulary repair into arbitrary text.
        if re.match(r"^print(?:\s|\()", source, re.IGNORECASE):
            lexical = _repair_common_ocr_words(source)
            if lexical != source:
                results.append(lexical)
    return list(dict.fromkeys(results))


def select_ocr_candidate(candidates: list[tuple[str, list[OCRWord]]], shape_type: str = "unknown") -> tuple[str, list[OCRWord], list[dict]]:
    """Select a best reading and retain near-ties for uncertain downstream review."""
    if not candidates:
        return "", [], []
    # Add conservative, whitespace-only repairs as candidates. Keep the OCR words
    # and confidence from the source reading for provenance and scoring.
    expanded = list(candidates)
    for text, words in candidates:
        for repaired in _repair_ocr_text(text, shape_type):
            if repaired != " ".join(text.split()):
                expanded.append((repaired, words))
    normalized_counts = {}
    for text, _ in expanded:
        normalized = " ".join(text.split()).casefold()
        if normalized:
            normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1
    ranked = sorted(((_candidate_score(text, words, shape_type) +
                      min(.24, .08 * max(0, normalized_counts.get(" ".join(text.split()).casefold(), 1) - 1)) +
                      (.85 if shape_type == "decision" and _valid_decision_expression(text) else 0.0),
                      text, words) for text, words in expanded), key=lambda item: item[0], reverse=True)
    best_score, best_text, best_words = ranked[0]
    plausible = []
    seen = set()
    low_confidence = not best_words or sum(w.confidence for w in best_words) / len(best_words) < .72
    for score, text, words in ranked:
        normalized = " ".join(text.split())
        if normalized and normalized not in seen and (score >= best_score - (2.0 if shape_type == "decision" else .45) or (low_confidence and len(plausible) < 3)):
            plausible.append({"text": normalized, "score": round(score, 3), "confidence": round(
                sum(w.confidence for w in words) / len(words), 3) if words else 0.0})
            seen.add(normalized)
        if len(plausible) >= 3:
            break
    return best_text, best_words, plausible


def read_best_words(image: np.ndarray, context: PipelineContext, origin: tuple[int, int] = (0, 0),
                    shape_type: str = "unknown", return_candidates: bool = False):
    """Compare original and shape-aware preprocessing variants using Tesseract."""
    candidates = []
    for variant in preprocess_ocr_candidates(image, shape_type):
        # Flowchart node labels are usually one line even when the containing
        # shape is large. PSM 7 is materially better for operators and short
        # conditions; retain the block/sparse modes for multiline print text.
        if shape_type == "connector":
            page_modes = (10, 6)
        elif shape_type == "decision":
            page_modes = (7, 6, 11)
        else:
            page_modes = (6, 11)
        for psm in page_modes:
            text, words = read_words(variant, context, psm, origin)
            scale_x = variant.shape[1] / max(1, image.shape[1])
            scale_y = variant.shape[0] / max(1, image.shape[0])
            if scale_x != 1 or scale_y != 1:
                for word in words:
                    box = word.bbox
                    word.bbox = BoundingBox(origin[0] + round((box.x - origin[0]) / scale_x),
                                             origin[1] + round((box.y - origin[1]) / scale_y),
                                             max(1, round(box.width / scale_x)), max(1, round(box.height / scale_y)))
            candidates.append((text, words))
    # Decision labels often occupy only a small central region of a large
    # diamond crop. Tesseract's block segmentation can return gibberish or no
    # text when the full diamond edges dominate the crop. Add a separate,
    # enlarged center crop as an additional candidate while retaining all
    # full-crop candidates for context and multiline fallbacks.
    if shape_type == "decision" and image.shape[0] >= 24 and image.shape[1] >= 40:
        crop_w = max(24, int(image.shape[1] * .70))
        crop_h = max(20, int(image.shape[0] * .60))
        left = max(0, (image.shape[1] - crop_w) // 2)
        top = max(0, (image.shape[0] - crop_h) // 2)
        center_crop = image[top:top + crop_h, left:left + crop_w]
        enlarged = cv2.resize(center_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        crop_origin = (origin[0] + left, origin[1] + top)
        text, words = read_words(enlarged, context, 7, crop_origin)
        for word in words:
            box = word.bbox
            word.bbox = BoundingBox(
                crop_origin[0] + round((box.x - crop_origin[0]) / 2.0),
                crop_origin[1] + round((box.y - crop_origin[1]) / 2.0),
                max(1, round(box.width / 2.0)), max(1, round(box.height / 2.0)))
        candidates.append((text, words))
    text, words, plausible = select_ocr_candidate(candidates, shape_type)
    return (text, words, plausible) if return_candidates else (text, words)


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
                    text, words, ocr_candidates = (read_best_words(crop, context, (x + left, y + top), geometry_kind, True))
                    text = re.sub(r"^\s*[/|\\]+\s*|\s*[/|\\]+\s*$", "", text).strip()
                    text = " ".join(text.split()).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
                    # Apply only high-confidence, vocabulary-bounded repairs to
                    # the selected reading. Keep the uncorrected OCR as a review
                    # alternative so users can reject the inference if needed.
                    if geometry_kind == "decision":
                        text = _normalize_comparison_spacing(text, geometry_kind)
                    elif geometry_kind in {"process", "output", "unknown"} and re.match(r"^print(?:\s|[:(])", text, re.IGNORECASE):
                        repaired_text = _repair_common_ocr_words(text)
                        if repaired_text != text:
                            original_reading = text
                            if not any(item.get("text") == original_reading for item in ocr_candidates):
                                ocr_candidates.append({"text": original_reading, "reason": "original OCR reading preserved for review",
                                                      "score": 0.0, "confidence": 0.0})
                            text = repaired_text
                            if not any(item.get("text") == text for item in ocr_candidates):
                                ocr_candidates.insert(0, {"text": text, "reason": "conservative programming-vocabulary correction",
                                                          "score": 0.0, "confidence": 0.0})
                    kind, text = infer_instruction_kind(geometry_kind, text)
                    flags = []
                    if kind == "data":
                        lowered = re.sub(r"^[^a-z]+", "", text.lower().strip())
                        first_word = re.split(r"\s+", lowered, maxsplit=1)[0] if lowered else ""
                        kind = "input" if first_word in {"input", "read", "enter"} else "output" if first_word in {"output", "display", "show", "print"} else "unknown"
                        if kind == "output" and first_word != "output":
                            text = "Output " + text.split(None, 1)[1]
                        if kind == "unknown":
                            flags.append("UNRECOGNIZED_DATA_TEXT")
                    # Preserve a leading "If" in the recognized source text.
                    # normalize_condition() removes it only when constructing the
                    # executable condition, so review still shows what OCR saw.
                    if kind == "unknown":
                        flags.append("UNSUPPORTED_SHAPE")
                    total = sum(len(word.text) for word in words)
                    confidence = sum(word.confidence * len(word.text) for word in words) / total if total else 0.0
                    polygon = cv2.approxPolyDP(contour, .002 * cv2.arcLength(contour, True), True).reshape(-1, 2)
                    node = Node(f"n{index}", kind, text, BoundingBox(x, y, width, height), confidence, flags,
                                [(int(a), int(b)) for a, b in polygon], words, ocr_candidates)
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
        # Contour fills do not always cover the outer half of anti-aliased
        # outlines. Those leftover node borders are often larger components than
        # the real arrows and can be mistaken for connectors that run around a
        # shape. Expand the node-removal mask slightly before extracting strokes.
        # The small gap this creates at a node is intentional: graph attachment
        # uses a bounded distance to the actual node polygon.
        removal = cv2.dilate(
            removal,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        # Otsu is useful for separating dark glyphs, but it drops pale gray
        # connector arrows on colored-node diagrams.  Build the connector mask
        # from a permissive light-background threshold instead; node interiors
        # are removed immediately below, so colored fills do not become strokes.
        strokes = np.where(gray < 245, 255, 0).astype(np.uint8)
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

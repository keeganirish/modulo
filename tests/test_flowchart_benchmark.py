"""Deterministic synthetic-fixture benchmark for recognition regressions.

The manifest is the ground-truth source. Pipeline metrics are reported per stage;
strict assertions are limited to fixture integrity and isolated geometry primitives
so the current recognizer's shortcomings are visible rather than hidden by weak goldens.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.services.vision_service import classify_shape, shape_contours
from backend.pipeline import analyze

ROOT = Path(__file__).parent / "fixtures"
IMAGE_DIR = ROOT / "flowcharts"
MANIFEST = json.loads((ROOT / "expected" / "manifest.json").read_text(encoding="utf-8"))

@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_fixture_and_ground_truth_are_complete(name):
    spec = MANIFEST[name]
    image_path = IMAGE_DIR / spec["image"]
    image = cv2.imread(str(image_path))
    assert image is not None, f"fixture stage: image missing/unreadable: {image_path}"
    assert image.ndim == 3 and image.shape[0] >= 150 and image.shape[1] >= 150, "fixture stage: invalid dimensions"
    assert spec["nodes"], "ground-truth stage: fixture has no nodes"
    assert len({node["id"] for node in spec["nodes"]}) == len(spec["nodes"]), "ground-truth stage: duplicate node ids"
    known = {node["id"] for node in spec["nodes"]}
    for edge in spec["edges"]:
        assert edge["source"] in known and edge["target"] in known, f"ground-truth graph stage: dangling expected edge {edge}"
    expected_file = ROOT / "expected" / f"{name}.txt"
    assert expected_file.exists(), f"ground-truth stage: missing pseudocode/control-flow specification for {name}"


def test_shape_classifier_distinguishes_standard_primitives():
    """Primitive-level sanity check independent of OCR and graph inference."""
    canvas = np.full((500, 800), 255, np.uint8)
    cv2.rectangle(canvas, (30, 50), (210, 130), 0, 3)
    diamond = np.array([[350, 45], [440, 90], [350, 135], [260, 90]], np.int32)
    cv2.polylines(canvas, [diamond], True, 0, 3)
    data = np.array([[530, 50], [700, 50], [720, 130], [550, 130]], np.int32)
    cv2.polylines(canvas, [data], True, 0, 3)
    binary = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Use outer contours directly to isolate classifier behavior from contour filtering.
    outer = [c for c in contours if cv2.contourArea(c) > 1000]
    classified = [classify_shape(c, canvas.shape[1]) for c in outer]
    assert "process" in classified, f"classification stage: process rectangle not classified; got {classified}"
    assert "decision" in classified, f"classification stage: decision diamond not classified; got {classified}"
    assert "data" in classified, f"classification stage: input/output parallelogram not classified; got {classified}"


@pytest.mark.parametrize("name", sorted(MANIFEST))
@pytest.mark.skipif(os.getenv("FLOWCHART_RUN_PIPELINE") != "1", reason="set FLOWCHART_RUN_PIPELINE=1 for OCR-dependent full pipeline baseline")
def test_pipeline_stage_baseline(name, record_property):
    """Run the real pipeline and expose stage-specific metrics in pytest output/JUnit.

    This is intentionally not a blanket pass for a correct-looking Python file:
    node text, shape types, graph edges/labels, and IR/pseudocode are scored separately.
    Baseline values are recorded as test properties for CI artifact collection.
    """
    path = IMAGE_DIR / MANIFEST[name]["image"]
    result = analyze(path.read_bytes(), filename=path.name)
    spec = MANIFEST[name]
    expected_nodes = {n["id"]: n for n in spec["nodes"]}
    actual_nodes = result.elements
    # IDs may be assigned by spatial recognition, so compare normalized node content and type counts.
    actual_types = sorted(n.type for n in actual_nodes)
    expected_types = sorted(n["type"] for n in spec["nodes"])
    actual_text = [" ".join(n.text.split()).casefold() for n in actual_nodes]
    expected_text = [" ".join(n["text"].split()).casefold() for n in spec["nodes"]]
    expected_edges = {(e["source"], e["target"], e["label"]) for e in spec["edges"]}
    actual_edges = []
    if result.graph:
        actual_edges = [(e.source, e.target, e.label) for e in result.graph.edges]
    # Direction and labels are scored independently from edge count. Node IDs are
    # spatially assigned by this implementation; this is still a useful regression
    # metric, but should be upgraded to geometry-matched IDs before release gating.
    direction_matches = sum(any(source == e["source"] and target == e["target"]
                                for source, target, _ in actual_edges)
                            for e in spec["edges"])
    label_matches = sum(any(source == e["source"] and target == e["target"] and label == e["label"]
                            for source, target, label in actual_edges)
                        for e in spec["edges"])
    stage_errors = [d.code for d in result.diagnostics]
    record_property("fixture", name)
    record_property("vision_expected_nodes", len(expected_nodes))
    record_property("vision_detected_nodes", len(actual_nodes))
    record_property("vision_shape_type_exact", actual_types == expected_types)
    record_property("ocr_text_exact", sorted(actual_text) == sorted(expected_text))
    record_property("graph_expected_edges", len(expected_edges))
    record_property("graph_detected_edges", len(actual_edges))
    record_property("graph_direction_matches", direction_matches)
    record_property("graph_branch_label_matches", label_matches)
    record_property("logic_ir_available", result.ir is not None)
    record_property("pseudocode_lines", len(result.pseudocode))
    record_property("diagnostics", ",".join(stage_errors))
    # The baseline test's contract is that analysis terminates and emits measurable metrics.
    # Accuracy expectations are reported as stage properties until the baseline is reviewed.
    assert result.completed_stages or result.diagnostics, f"pipeline stage: no stage completion or diagnostic for {name}"


def test_fixture_suite_covers_requested_failure_modes():
    tags = {tag for spec in MANIFEST.values() for tag in spec["tags"]}
    required = {"sequence", "input_output", "decision", "nested_branches", "loop", "crossing",
                "long_connectors", "small_text", "math_operators", "colored_background", "low_resolution"}
    assert required <= tags, f"fixture coverage stage: missing tags {sorted(required - tags)}"

import cv2
import numpy as np

from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
from backend.services.vision_service import _reconcile_low_confidence_comparison_thresholds
from backend.services.graph_service import infer_dangling_arrow_to_existing_edge


def test_low_confidence_zero_one_comparison_is_contextually_reconciled_and_preserved():
    uncertain = Node("n1", "decision", "if count > 1", BoundingBox(0, 0, 100, 60), .74)
    corroborating = Node("n2", "decision", "if count < 0", BoundingBox(0, 100, 100, 60), .90)
    _reconcile_low_confidence_comparison_thresholds([uncertain, corroborating])
    assert uncertain.text == "if count > 0"
    assert any(item["text"] == "if count > 1" for item in uncertain.ocr_candidates)
    assert "CONTEXT_CORRECTED_OCR" in uncertain.flags


def test_clear_or_unpaired_numeric_comparisons_are_not_rewritten():
    clear = Node("n1", "decision", "if count > 1", BoundingBox(0, 0, 100, 60), .95)
    unpaired = Node("n2", "decision", "if total > 1", BoundingBox(0, 100, 100, 60), .70)
    _reconcile_low_confidence_comparison_thresholds([clear, unpaired])
    assert clear.text == "if count > 1"
    assert unpaired.text == "if total > 1"


def test_arrowhead_pointing_into_existing_connector_recovers_merge_edge():
    # Draw a right-angle branch that ends in a left-facing arrowhead just short
    # of an already traced vertical connector.
    mask = np.zeros((360, 360), dtype=np.uint8)
    cv2.line(mask, (220, 120), (220, 180), 255, 4)
    cv2.line(mask, (220, 180), (120, 180), 255, 4)
    cv2.line(mask, (120, 180), (100, 168), 255, 4)
    cv2.line(mask, (120, 180), (100, 192), 255, 4)
    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.int32)

    source = Node("n1", "output", "Output result", BoundingBox(200, 80, 40, 40))
    target = Node("n2", "decision", "if ready", BoundingBox(70, 280, 40, 40))
    upstream = Node("n3", "decision", "if valid", BoundingBox(70, 100, 40, 40))
    receiving = Edge("e1", "n3", "n2", points=[(90, 140), (90, 280)])
    graph = FlowGraph(nodes=[source, target, upstream], edges=[receiving])
    context = PipelineContext(graph=graph)

    count = infer_dangling_arrow_to_existing_edge(context, graph, [(source, points)], mask.shape)
    assert count == 1
    assert any(edge.source == "n1" and edge.target == "n2" for edge in graph.edges)
    assert any(item.code == "INFERRED_ARROW_TO_CONNECTOR" for item in context.diagnostics)

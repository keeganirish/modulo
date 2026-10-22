import cv2
import numpy as np
import pytest
from backend.models import BoundingBox, Node, PipelineContext
from backend.services.graph_service import GraphService
from tests.drawing import arrow

def node(identifier, box):
    x, y, w, h = box
    return Node(identifier, "process", "x = 1", BoundingBox(*box), polygon=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)])

def graph_context(settings, points, boxes, head=True):
    image = np.full((650, 900, 3), 255, np.uint8)
    arrow(image, points, head)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    elements = [node(f"n{i+1}", box) for i, box in enumerate(boxes)]
    # Vision would remove pixels at the shape boundaries; emulate that contract.
    for element in elements:
        cv2.fillPoly(mask, [np.asarray(element.polygon, np.int32)], 0)
    return PipelineContext(elements=elements, stroke_mask=mask, settings=settings)

@pytest.mark.parametrize("points,boxes,expected", [
    ([(205,140),(390,140)], [(50,100,150,80),(395,100,150,80)], ("n1","n2")),
    ([(390,140),(205,140)], [(50,100,150,80),(395,100,150,80)], ("n2","n1")),
    ([(205,140),(300,140),(300,380),(390,380)], [(50,100,150,80),(395,340,150,80)], ("n1","n2")),
])
def test_direction_and_bends(settings, points, boxes, expected):
    ctx = GraphService().run(graph_context(settings, points, boxes))
    assert not ctx.has_errors, [d.to_dict() for d in ctx.diagnostics]
    assert [(e.source,e.target) for e in ctx.graph.edges] == [expected]
    assert ctx.graph.edges[0].points

def test_no_arrowhead_never_guessed(settings):
    ctx = GraphService().run(graph_context(settings, [(205,140),(390,140)], [(50,100,150,80),(395,100,150,80)], head=False))
    assert ctx.graph.edges == []
    assert any(d.code == "AMBIGUOUS_ARROW_DIRECTION" for d in ctx.diagnostics)

def test_dangling_arrow(settings):
    ctx = GraphService().run(graph_context(settings, [(205,140),(350,140)], [(50,100,150,80)]))
    assert any(d.code == "DANGLING_ARROW" for d in ctx.diagnostics)

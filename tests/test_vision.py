import cv2
import numpy as np
import pytest
from backend.models import PipelineContext
from backend.services.intake_service import IntakeService
from backend.services.vision_service import VisionService, shape_contours, classify_shape
from tests.drawing import canvas, shape, encode, sequence

@pytest.mark.parametrize("kind,value", [("process", "value = 5"), ("terminal", "Start"), ("decision", "value > 0"), ("iteration", "WHILE value > 0"), ("input", "Input value:int"), ("connector", "A")])
def test_geometry(kind, value):
    image = canvas(1000, 350)
    shape(image, kind, (300, 100, 50 if kind == "connector" else 360, 50 if kind == "connector" else 100), value)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours = shape_contours(binary)
    assert len(contours) == 1
    assert classify_shape(contours[0], 1000) == ("data" if kind == "input" else kind)

@pytest.mark.integration
def test_real_ocr_custom_text_and_order(settings):
    ctx = IntakeService().run(PipelineContext(filename="not-a-sample.png", uploaded_bytes=encode(sequence("amount = 93")), settings=settings))
    VisionService().run(ctx)
    assert not ctx.has_errors, [d.to_dict() for d in ctx.diagnostics]
    assert [n.id for n in ctx.elements] == ["n1", "n2", "n3", "n4"]
    assert [n.type for n in ctx.elements] == ["terminal", "process", "output", "terminal"]
    assert ctx.elements[1].text == "amount = 93"
    assert ctx.elements[1].words
    assert ctx.elements[1].confidence > .5

def test_blank_image_reports_no_shapes(settings):
    image = np.full((300, 400), 255, np.uint8)
    ctx = VisionService().run(PipelineContext(grayscale=image, binary=255-image, settings=settings))
    assert ctx.diagnostics[0].code == "NO_SHAPES_DETECTED"

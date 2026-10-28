from io import BytesIO
import numpy as np
import pymupdf
import pytest
from PIL import Image
from backend.models import PipelineContext
from backend.services.intake_service import IntakeService, UploadError, inspect_upload
from tests.drawing import encode, sequence, independent_pdf

SVG_FLOWCHART = b'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300"><rect width="400" height="300" fill="white"/><rect x="100" y="60" width="200" height="80" rx="30" fill="white" stroke="black" stroke-width="4"/><text x="145" y="110" font-size="24">Start</text><rect x="100" y="180" width="200" height="60" fill="white" stroke="black" stroke-width="4"/><text x="160" y="220" font-size="24">End</text></svg>'

@pytest.mark.parametrize("extension", [".png", ".jpg"])
def test_normalized_image(extension, settings):
    ctx = PipelineContext(filename="custom" + extension, uploaded_bytes=encode(sequence(), extension), settings=settings)
    assert IntakeService().run(ctx) is ctx
    assert not ctx.has_errors
    assert ctx.page_image.shape[1] == 2000
    assert ctx.binary.shape == ctx.grayscale.shape
    assert set(np.unique(ctx.binary)) <= {0, 255}

@pytest.mark.parametrize("data,name,code", [(b"", "empty.png", "EMPTY_FILE"), (b"text", "chart.gif", "UNSUPPORTED_FILE_TYPE"), (b"text", "fake.png", "CONTENT_TYPE_MISMATCH"), (b"%PDF-broken", "chart.png", "CONTENT_TYPE_MISMATCH")])
def test_request_validation(data, name, code, settings):
    with pytest.raises(UploadError) as error:
        inspect_upload(data, name, settings)
    assert error.value.code == code

def test_corrupt_png(settings):
    ctx = IntakeService().run(PipelineContext(filename="broken.png", uploaded_bytes=b"\x89PNG\r\n\x1a\nxxx", settings=settings))
    assert ctx.diagnostics[0].code == "INVALID_IMAGE"

def test_alpha_background_is_white(settings):
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    data = BytesIO(); image.save(data, format="PNG")
    ctx = IntakeService().run(PipelineContext(filename="transparent.png", uploaded_bytes=data.getvalue(), settings=settings))
    assert np.min(ctx.page_image) == 255

def test_real_vector_pdf_intake(settings):
    ctx = IntakeService().run(PipelineContext(filename="other.pdf", uploaded_bytes=independent_pdf(), settings=settings))
    assert not ctx.has_errors
    assert ctx.page_image.shape[1] == 2000

def test_multipage_rejected(settings):
    document = pymupdf.open(); document.new_page(); document.new_page()
    data = document.tobytes(); document.close()
    ctx = IntakeService().run(PipelineContext(filename="two.pdf", uploaded_bytes=data, settings=settings))
    assert ctx.diagnostics[0].code == "MULTIPAGE_PDF"

def test_pixel_limit(settings):
    from dataclasses import replace
    ctx = IntakeService().run(PipelineContext(filename="large.png", uploaded_bytes=encode(sequence()), settings=replace(settings, MAX_IMAGE_PIXELS=10000)))
    assert ctx.diagnostics[0].code == "IMAGE_TOO_LARGE"


def test_svg_is_validated_and_rasterized(settings):
    assert inspect_upload(SVG_FLOWCHART, "diagram.svg", settings) == "svg"
    ctx = IntakeService().run(PipelineContext(filename="diagram.svg", uploaded_bytes=SVG_FLOWCHART, settings=settings))
    assert not ctx.has_errors
    assert ctx.page_image is not None and ctx.page_image.shape[1] == 2000


def test_svg_active_content_is_rejected(settings):
    data = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    ctx = IntakeService().run(PipelineContext(filename="unsafe.svg", uploaded_bytes=data, settings=settings))
    assert ctx.diagnostics[0].code == "UNSAFE_SVG"

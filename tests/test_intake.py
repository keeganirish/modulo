from io import BytesIO
import numpy as np
import pymupdf
import pytest
from PIL import Image
from backend.models import PipelineContext
from backend.services.intake_service import IntakeService, UploadError, inspect_upload
from tests.drawing import encode, sequence, independent_pdf

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

import pytest
from backend.config import Settings
from backend.models import PipelineContext
from tests.drawing import encode, sequence

@pytest.fixture
def settings():
    return Settings(WORKING_IMAGE_WIDTH=2000)

@pytest.fixture
def diagram_bytes():
    return encode(sequence())

@pytest.fixture
def context(diagram_bytes, settings):
    return PipelineContext(filename="unrelated-user-chart.png", uploaded_bytes=diagram_bytes, settings=settings)

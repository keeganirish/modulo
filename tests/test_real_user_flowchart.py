from pathlib import Path
import pytest
from backend.pipeline import analyze

FLOWCHART = Path(__file__).resolve().parents[1] / 'inputs' / 'flowchart-2.png'

@pytest.mark.integration
def test_real_user_filled_color_flowchart(settings):
    if not FLOWCHART.exists():
        pytest.skip('The user-provided flowchart is not present in this checkout.')
    context = analyze(FLOWCHART.read_bytes(), 'flowchart-2.png', settings)
    assert not context.has_errors, [diagnostic.to_dict() for diagnostic in context.diagnostics]
    assert [node.type for node in context.elements] == ['terminal', 'input', 'decision', 'output', 'output', 'terminal']
    texts = [node.text.upper() for node in context.elements]
    assert texts[0] == 'START'
    assert 'NUMBER' in texts[1]
    assert 'NUMBER' in texts[2]
    assert 'EVEN' in texts[3]
    assert 'ODD' in texts[4]
    assert context.pseudocode

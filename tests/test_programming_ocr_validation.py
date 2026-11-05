"""Programming-aware OCR ambiguity detection must be review-only."""
from backend.services.programming_ocr_validation import propose_ambiguities, ProgrammingOCRValidationService
from backend.models import BoundingBox, FlowGraph, Node, PipelineContext
from backend.config import Settings


def test_arithmetic_modulo_preserves_original_and_proposes_zero_alternative():
    alternatives = propose_ambiguities("value % 2", "process")
    assert any(item["text"] == "value 0 2" for item in alternatives)


def test_comparison_keeps_both_equality_and_assignment_readings():
    alternatives = propose_ambiguities("x == 3", "decision")
    assert any(item["text"] == "x = 3" for item in alternatives)
    assert any(item["text"] == "x = = 3" for item in alternatives) is False


def test_input_statement_and_variable_character_confusion_are_supported():
    alternatives = propose_ambiguities("Input level1:int", "input")
    # Candidate OCR readings are retained even when the ambiguity is inside a name.
    alternatives = propose_ambiguities("Input level1:int", "input", [{"text": "Input levell:int", "score": 3, "confidence": .7}])
    assert any(item["text"] == "Input levell:int" for item in alternatives)


def test_parentheses_confusions_are_proposed_without_rewriting():
    original = "print(int(value))"
    alternatives = propose_ambiguities(original, "process")
    assert any(item["text"] != original for item in alternatives)
    assert original == "print(int(value))"


def test_service_flags_ambiguity_but_never_changes_node_text():
    node = Node("n1", "decision", "x == 0", BoundingBox(0, 0, 20, 10), .95)
    context = PipelineContext(filename="test.png", settings=Settings(), graph=FlowGraph(nodes=[node]))
    ProgrammingOCRValidationService().run(context)
    assert node.text == "x == 0"
    assert "PROGRAMMING_OCR_AMBIGUITY" in node.flags
    diagnostic = next(item for item in context.diagnostics if item.code == "PROGRAMMING_OCR_AMBIGUITY")
    assert diagnostic.severity == "warning"
    assert node.ocr_candidates


def test_syntax_support_is_weak_and_does_not_remove_invalid_alternatives():
    alternatives = propose_ambiguities("x = 3", "decision")
    assert any(item["text"] == "x == 3" for item in alternatives)


def test_original_ocr_reading_survives_a_review_correction():
    node = Node("n1", "process", "x = 3", BoundingBox(0, 0, 20, 10))
    node.text = "x == 3"  # same mutation performed by the review workflow
    assert node.original_ocr_text == "x = 3"
    assert node.text == "x == 3"

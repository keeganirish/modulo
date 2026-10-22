import pytest
from backend.models import BoundingBox, Node, Edge, FlowGraph, PipelineContext
from backend.notation import NotationError, validate_node_text, normalize_label
from backend.services.validation_service import ValidationService

@pytest.mark.parametrize("kind,text", [("process","x = 3 + 5"),("process","total += n"),("process","print(total)"),("input","Input n:int"),("output","Output total"),("decision","n > 0 and n < 10"),("iteration","WHILE n > 0"),("iteration","FOR i IN range(5)"),("process","break"),("process","continue"),("process","return")])
def test_supported_notation(kind, text):
    validate_node_text(kind,text)

@pytest.mark.parametrize("text", ["import os", "x = __import__('os')", "x = value.real", "exec('pass')", "x = (lambda: 1)()", "x = [n for n in range(4)]", "x = values[0]", "print = 2", "x = 1; y = 2", "obj.method()", "x = open('file')", "x = getattr(obj, 'name')", "x = f'{n}'"])
def test_untrusted_code_rejected(text):
    with pytest.raises(NotationError):
        validate_node_text("process",text)

@pytest.mark.parametrize("label,expected", [("yes","Yes"),(" TRUE ","Yes"),("y","Yes"),("n","No"),("False","No"),("NO","No"),("maybe",None)])
def test_branch_vocabulary(label,expected):
    assert normalize_label(label) == expected

def make_graph(settings):
    nodes = [Node("n1","terminal","Start",BoundingBox(10,10,120,50)),Node("n2","process","x = 1",BoundingBox(10,100,120,50)),Node("n3","terminal","End",BoundingBox(10,200,120,50))]
    return PipelineContext(graph=FlowGraph(nodes,[Edge("e1","n1","n2"),Edge("e2","n2","n3")]),settings=settings)

def test_valid_graph(settings):
    ctx=ValidationService().run(make_graph(settings))
    assert not ctx.has_errors
    assert ctx.graph.start_id == "n1" and ctx.graph.end_ids == ["n3"]

@pytest.mark.parametrize("change,code", [("no_start","NO_START_NODE"),("two_starts","MULTIPLE_START_NODES"),("unreachable","UNREACHABLE_NODE"),("no_end","NO_END_NODE"),("no_path","NO_PATH_TO_END"),("low_ocr","LOW_OCR_CONFIDENCE"),("connector","UNPAIRED_CONNECTOR")])
def test_diagnostics(settings,change,code):
    ctx=make_graph(settings)
    if change == "no_start": ctx.graph.nodes[0].text="End"
    if change == "two_starts": ctx.graph.nodes[2].text="Start"
    if change == "unreachable": ctx.graph.edges.pop(0)
    if change == "no_end": ctx.graph.nodes[2].type="process";ctx.graph.nodes[2].text="x = 2"
    if change == "no_path": ctx.graph.edges[1].target="n2"
    if change == "low_ocr": ctx.graph.nodes[1].confidence=.3
    if change == "connector": ctx.graph.nodes[1].type="connector";ctx.graph.nodes[1].text="A"
    ValidationService().run(ctx)
    assert any(d.code==code for d in ctx.diagnostics)

def test_invalid_branch_labels(settings):
    ctx=make_graph(settings);ctx.graph.nodes[1].type="decision";ctx.graph.nodes[1].text="x > 0"
    ctx.graph.edges += [Edge("e3","n2","n3","Yes")]
    ValidationService().run(ctx)
    assert any(d.code=="INVALID_BRANCH_LABELS" for d in ctx.diagnostics)

def test_missing_branch(settings):
    ctx=make_graph(settings);ctx.graph.nodes[1].type="decision";ctx.graph.nodes[1].text="x > 0"
    ValidationService().run(ctx)
    assert any(d.code=="DECISION_MISSING_BRANCH" for d in ctx.diagnostics)

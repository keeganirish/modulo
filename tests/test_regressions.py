import pytest
from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
from backend.notation import NotationError, validate_node_text
from backend.pipeline import analyze
from backend.services.graph_service import GraphService
from backend.services.validation_service import ValidationService
from tests.drawing import canvas, shape, arrow, encode
from tests.test_graph import graph_context

@pytest.mark.parametrize('kind,value', [('input','Input class:int'),('input','Input for'),('iteration','FOR if IN range(3)')])
def test_keyword_names_rejected(kind,value):
    with pytest.raises(NotationError):
        validate_node_text(kind,value)

def test_diagonal_arrow(settings):
    ctx=GraphService().run(graph_context(settings,[(200,180),(400,380)],[(50,100,150,80),(395,380,150,80)]))
    assert not ctx.has_errors,[d.to_dict() for d in ctx.diagnostics]
    assert [(e.source,e.target) for e in ctx.graph.edges]==[('n1','n2')]

@pytest.mark.integration
def test_connector_pair_real_ocr(settings):
    image=canvas(1000,800)
    shape(image,'terminal',(100,50,260,80),'Start')
    shape(image,'connector',(205,270,50,50),'A')
    shape(image,'connector',(705,400,50,50),'A')
    shape(image,'terminal',(600,620,260,80),'End')
    arrow(image,[(230,130),(230,270)])
    arrow(image,[(730,450),(730,620)])
    ctx=analyze(encode(image),'different-layout-circles.png',settings)
    assert not ctx.has_errors,[d.to_dict() for d in ctx.diagnostics]
    structural=[edge for edge in ctx.graph.edges if edge.structural]
    assert len(structural)==1
    assert len(ctx.graph.nodes)==4 and len(ctx.graph.edges)==3

def test_break_after_loop_rejected(settings):
    nodes=[Node('n1','terminal','Start',BoundingBox(0,0,100,50)),Node('n2','iteration','WHILE x > 0',BoundingBox(0,80,100,50)),Node('n3','process','x = x - 1',BoundingBox(0,160,100,50)),Node('n4','process','break',BoundingBox(0,240,100,50)),Node('n5','terminal','End',BoundingBox(0,320,100,50))]
    edges=[Edge('e1','n1','n2'),Edge('e2','n2','n3','Yes'),Edge('e3','n3','n2'),Edge('e4','n2','n4','No'),Edge('e5','n4','n5')]
    ctx=ValidationService().run(PipelineContext(graph=FlowGraph(nodes,edges),settings=settings))
    assert any(d.code=='LOOP_CONTROL_OUTSIDE_LOOP' for d in ctx.diagnostics)

def test_branch_without_labels_never_guessed(settings):
    nodes=[Node('n1','terminal','Start',BoundingBox(0,0,100,50)),Node('n2','decision','x > 0',BoundingBox(0,80,100,50)),Node('n3','terminal','End',BoundingBox(0,160,100,50))]
    edges=[Edge('e1','n1','n2'),Edge('e2','n2','n3'),Edge('e3','n2','n3')]
    ctx=ValidationService().run(PipelineContext(graph=FlowGraph(nodes,edges),settings=settings))
    assert any(d.code=='INVALID_BRANCH_LABELS' for d in ctx.diagnostics)

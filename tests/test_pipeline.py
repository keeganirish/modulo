from dataclasses import replace
import json
import pytest
from backend.pipeline import analyze
from backend.models import AnalysisResponse, PipelineContext
from tests.drawing import encode, sequence, branch, loop, independent_pdf

@pytest.mark.integration
@pytest.mark.parametrize("value,horizontal,extension", [("amount = 17",False,".png"),("amount = 83",True,".png"),("amount = 29",False,".jpg")])
def test_unseen_diagrams(settings,value,horizontal,extension):
    ctx=analyze(encode(sequence(value,horizontal),extension),"unseen-user-upload"+extension,settings)
    assert not ctx.has_errors, [d.to_dict() for d in ctx.diagnostics]
    assert ctx.completed_stages == ["intake","vision","graph","validation","logic","pseudocode"]
    assert ctx.graph.nodes[1].text == value
    expected=[("n1","n2"),("n2","n3")] if horizontal else [("n1","n2"),("n2","n3"),("n3","n4")]
    assert [(e.source,e.target) for e in ctx.graph.edges] == expected
    assert ctx.pseudocode and ctx.code == []
    assert AnalysisResponse(ctx).to_dict()["status"] == "ready"

@pytest.mark.integration
def test_independent_vector_pdf(settings):
    ctx=analyze(independent_pdf(),"external-author.pdf",settings)
    assert not ctx.has_errors, [d.to_dict() for d in ctx.diagnostics]
    assert [n.text for n in ctx.graph.nodes] == ["Start","result = 42","End"]
    assert [(e.source,e.target) for e in ctx.graph.edges] == [("n1","n2"),("n2","n3")]

@pytest.mark.integration
def test_branches_labels_and_bends(settings):
    ctx=analyze(encode(branch()),"grading-flow.png",settings)
    assert not ctx.has_errors, [d.to_dict() for d in ctx.diagnostics]
    decision=next(n for n in ctx.graph.nodes if n.type=="decision")
    edges=[e for e in ctx.graph.edges if e.source==decision.id]
    assert sorted(e.label for e in edges) == ["No","Yes"]
    assert len(ctx.graph.nodes)==6 and len(ctx.graph.edges)==6

@pytest.mark.integration
def test_loop_and_back_edge(settings):
    ctx=analyze(encode(loop()),"countdown.png",settings)
    assert not ctx.has_errors, [d.to_dict() for d in ctx.diagnostics]
    header=next(n for n in ctx.graph.nodes if n.type=="iteration")
    assert len([e for e in ctx.graph.edges if e.source==header.id])==2
    assert len([e for e in ctx.graph.edges if e.target==header.id])==2

@pytest.mark.integration
def test_same_bytes_same_graph(settings):
    data=encode(sequence())
    first=AnalysisResponse(analyze(data,"first-name.png",settings)).to_dict()
    second=AnalysisResponse(analyze(data,"different-name.png",settings)).to_dict()
    assert json.dumps(first,sort_keys=True)==json.dumps(second,sort_keys=True)

def test_pipeline_stops_at_error(settings):
    calls=[]
    class Stage:
        def __init__(self,name,fatal=False): self.name,self.fatal=name,fatal
        def run(self,ctx):
            calls.append(self.name)
            if self.fatal: ctx.diagnose("error","STOP","stop","fix")
            return ctx
    ctx=analyze(PipelineContext(settings=settings),services=[Stage("first",True),Stage("second")])
    assert calls==["first"] and ctx.completed_stages==["first"]

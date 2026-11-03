"""Trace source elements to pseudocode and Python lines in both directions."""
from collections import defaultdict
from backend.models import PipelineContext

class TraceService:
    name = "trace"
    def run(self, context: PipelineContext) -> PipelineContext:
        element_to_pseudo=defaultdict(list); element_to_code=defaultdict(list); pseudo_to_code=defaultdict(list); code_to_pseudo=defaultdict(list)
        for line in context.pseudocode:
            for source_id in line.source_ids: element_to_pseudo[source_id].append(line.line)
        for line in context.code:
            for source_id in line.source_ids: element_to_code[source_id].append(line.line)
        for source_id in sorted(set(element_to_pseudo)|set(element_to_code)):
            pseudo=element_to_pseudo[source_id]; code=element_to_code[source_id]
            for number in pseudo: pseudo_to_code[number].extend(code)
            for number in code: code_to_pseudo[number].extend(pseudo)
        structural={node.id for node in (context.graph.nodes if context.graph else []) if node.type in ("terminal","connector")}
        required={node.id for node in (context.graph.nodes if context.graph else []) if node.id not in structural}
        missing=sorted(required-set(element_to_code))
        if missing: context.diagnose("error","TRACE_COVERAGE_FAILED","Some flowchart elements did not produce Python lines.","Every supported process, input, output, decision, and loop must be represented in generated code.",missing)
        context.trace={"element_to_pseudocode":dict(element_to_pseudo),"element_to_code":dict(element_to_code),"pseudocode_to_code":dict(pseudo_to_code),"code_to_pseudocode":dict(code_to_pseudo)}
        return context

def run(context): return TraceService().run(context)

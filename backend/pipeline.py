"""Pipeline orchestration only. Later phases remain deliberately unavailable."""
from backend.config import Settings
from backend.models import PipelineContext
from backend.services.intake_service import IntakeService
from backend.services.vision_service import VisionService
from backend.services.graph_service import GraphService
from backend.services.validation_service import ValidationService
from backend.services.logic_service import LogicService
from backend.services.pseudocode_service import PseudocodeService
from backend.services.codegen_service import CodegenService
from backend.services.trace_service import TraceService

ANALYSIS_SERVICES = (IntakeService, VisionService, GraphService, ValidationService, LogicService, PseudocodeService)

def analyze(data: bytes | PipelineContext, filename: str = "", settings: Settings | None = None, services=None) -> PipelineContext:
    context = data if isinstance(data, PipelineContext) else PipelineContext(filename=filename, uploaded_bytes=data, settings=settings or Settings())
    if context.settings is None:
        context.settings = settings or Settings()
    for service in services if services is not None else [stage() for stage in ANALYSIS_SERVICES]:
        if context.has_errors:
            break
        context = service.run(context)
        context.completed_stages.append(service.name)
    return context

def generate(context: PipelineContext) -> PipelineContext:
    if not context.approved:
        return context
    for service in (CodegenService(), TraceService()):
        if context.has_errors: break
        context = service.run(context)
        context.completed_stages.append(service.name)
    return context

def review(context: PipelineContext, node_text: dict[str, str] | None = None,
           edge_labels: dict[str, str] | None = None) -> PipelineContext:
    """Apply constrained OCR/branch-label corrections, then rebuild safe stages."""
    node_text = node_text or {}
    edge_labels = edge_labels or {}
    # Keep the original diagnostics long enough to authorize edits only for flagged
    # elements, then clear them before validating the submitted corrections. If we
    # leave the original errors in place, context.has_errors short-circuits every
    # review and the user can never correct the issues that triggered review mode.
    flagged = {element_id for diagnostic in context.diagnostics for element_id in diagnostic.element_ids}
    context.diagnostics.clear()
    nodes = {node.id: node for node in context.elements}
    edges = {edge.id: edge for edge in (context.graph.edges if context.graph else [])}
    for node_id, text in node_text.items():
        if node_id not in flagged or node_id not in nodes or not isinstance(text, str) or not text.strip() or len(text) > 500:
            context.diagnose("error", "INVALID_REVIEW_CORRECTION", "A node correction is not attached to a flagged node or is invalid.", "Edit only flagged node text using at most 500 characters.", [node_id])
    for edge_id, label in edge_labels.items():
        if edge_id not in flagged or edge_id not in edges or not isinstance(label, str) or len(label) > 40:
            context.diagnose("error", "INVALID_REVIEW_CORRECTION", "A branch-label correction is not attached to a flagged edge or is invalid.", "Edit only flagged branch labels using at most 40 characters.", [edge_id])
    if context.has_errors:
        return context
    for node_id, text in node_text.items():
        nodes[node_id].text = text.strip()
        nodes[node_id].flags.append("USER_CORRECTED")
    for edge_id, label in edge_labels.items():
        edges[edge_id].label = label.strip() or None
    context.diagnostics.clear()
    context.ir = None
    context.pseudocode.clear()
    context.code.clear()
    context.trace.clear()
    context.approved = False
    context.completed_stages = [stage for stage in context.completed_stages if stage in {"intake", "vision", "graph"}]
    for service in (ValidationService(), LogicService(), PseudocodeService()):
        if context.has_errors:
            break
        context = service.run(context)
        context.completed_stages.append(service.name)
    return context

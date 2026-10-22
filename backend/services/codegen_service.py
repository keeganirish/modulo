"""Deterministic Python renderer; generated source is parsed but never executed."""
import ast
from backend.models import PipelineContext, SourceLine

class CodegenService:
    name = "codegen"
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.ir is None or context.has_errors or not context.approved: return context
        body=[]; self._render(context.ir,body,1)
        body=[SourceLine(1,0,"def main():",[])]+[SourceLine(index+2,indent,text,ids) for index,(indent,text,ids) in enumerate(body)]
        body += [SourceLine(len(body)+1,0,"",[]),SourceLine(len(body)+2,0,'if __name__ == "__main__":',[]),SourceLine(len(body)+3,1,"main()",[])]
        source="\n".join("    "*line.indent+line.text for line in body)+"\n"
        try: ast.parse(source); compile(source,"<modulo-generated>","exec")
        except (SyntaxError,ValueError) as error:
            context.diagnose("error","SYNTAX_CHECK_FAILED",f"Generated Python failed its syntax check: {error}","This is an implementation defect; preserve the flowchart and report the generated structure.")
            return context
        context.code=body; return context
    def _render(self,node,lines,indent):
        for child in node.children:
            ids=child.source_ids
            if child.kind=="Process": lines.append((indent,child.text,ids))
            elif child.kind=="Input": lines.append((indent,f"{child.variable} = {child.expression}(input())",ids))
            elif child.kind=="Output": lines.append((indent,f"print({child.expression})",ids))
            elif child.kind=="If":
                lines.append((indent,f"if {child.condition}:",ids)); self._render_block(child.children,lines,indent+1)
                if child.otherwise: lines.append((indent,"else:",ids)); self._render_block(child.otherwise,lines,indent+1)
            elif child.kind=="While": lines.append((indent,f"while {child.condition}:",ids)); self._render_block(child.children,lines,indent+1)
            elif child.kind=="For": lines.append((indent,f"for {child.variable} in {child.expression}:",ids)); self._render_block(child.children,lines,indent+1)
            elif child.kind in ("Start","End"): continue
    def _render_block(self,children,lines,indent): self._render(type("Block",(),{"children":children})(),lines,indent)

def run(context): return CodegenService().run(context)

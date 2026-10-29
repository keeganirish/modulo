"""Deterministic Python renderer with case-insensitive flowchart names and input inference.

Generated source is parsed and compiled, and variable references are normalized to the
spelling used by their declaration so OCR/capitalization differences do not create
runtime NameErrors (for example, Input Number followed by `number % 2 == 0`).
"""
from __future__ import annotations
import ast
from backend.models import IRNode, PipelineContext, SourceLine

_NUMERIC_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)

class _CanonicalNames(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]):
        self.names = names
    def visit_Name(self, node):
        canonical = self.names.get(node.id.casefold())
        if canonical:
            node.id = canonical
        return node

class CodegenService:
    name = "codegen"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.ir is None or context.has_errors or not context.approved:
            return context
        names, input_types = self._analyze(context.ir)
        body=[]
        self._render(context.ir, body, 1, names, input_types)
        body=[SourceLine(1,0,"def main():",[])]+[SourceLine(index+2,indent,text,ids) for index,(indent,text,ids) in enumerate(body)]
        body += [SourceLine(len(body)+1,0,"",[]),SourceLine(len(body)+2,0,'if __name__ == "__main__":',[]),SourceLine(len(body)+3,1,"main()",[])]
        source="\n".join("    "*line.indent+line.text for line in body)+"\n"
        try:
            tree = ast.parse(source)
            compile(tree,"<modulo-generated>","exec")
            self._check_references(tree)
        except (SyntaxError, ValueError, NameError) as error:
            context.diagnose("error","SYNTAX_CHECK_FAILED",f"Generated Python failed its validation check: {error}","This is an implementation defect; preserve the flowchart and report the generated structure.")
            return context
        context.code=body
        return context

    def _analyze(self, root: IRNode):
        names: dict[str,str] = {}
        input_types: dict[str,str] = {}
        expressions: list[str] = []
        def define(name):
            if name:
                names.setdefault(name.casefold(), name)
        def walk(node):
            for child in node.children:
                if child.kind == "Input":
                    define(child.variable)
                    input_types[child.variable.casefold()] = child.expression or "str"
                elif child.kind == "For":
                    define(child.variable); expressions.append(child.expression)
                if child.kind in ("If", "While"):
                    expressions.append(child.condition)
                elif child.kind == "Output":
                    expressions.append(child.expression)
                elif child.kind == "Process":
                    expressions.append(child.text)
                walk(child)
                if child.otherwise:
                    walk(IRNode("Block", children=child.otherwise))
        walk(root)
        # If an input participates in numeric arithmetic, treat an untyped input as int.
        # Explicit input annotations (int/float/str) always take precedence.
        for expression in expressions:
            try:
                tree = ast.parse(expression, mode="exec" if "=" in expression and "==" not in expression else "eval")
            except (SyntaxError, ValueError):
                try: tree = ast.parse(expression, mode="eval")
                except (SyntaxError, ValueError): continue
            for operation in (n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, _NUMERIC_BINOPS)):
                for item in ast.walk(operation):
                    if isinstance(item, ast.Name) and item.id.casefold() in input_types:
                        key = item.id.casefold()
                        if input_types[key] == "str": input_types[key] = "int"
        return names, input_types

    def _normalize(self, text: str, names: dict[str,str], mode="eval") -> str:
        try:
            tree = ast.parse(text, mode=mode)
            tree = _CanonicalNames(names).visit(tree)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree).strip()
        except (SyntaxError, ValueError):
            return text

    def _render(self,node,lines,indent,names,input_types):
        for child in node.children:
            ids=child.source_ids
            if child.kind=="Process":
                lines.append((indent,self._normalize(child.text,names,"exec"),ids))
            elif child.kind=="Input":
                variable=names.get(child.variable.casefold(),child.variable)
                input_type=input_types.get(child.variable.casefold(),child.expression or "str")
                lines.append((indent,f"{variable} = {input_type}(input())",ids))
            elif child.kind=="Output":
                expression=self._normalize(child.expression,names)
                lines.append((indent,f"print({expression})",ids))
            elif child.kind=="If":
                condition=self._normalize(child.condition,names)
                lines.append((indent,f"if {condition}:",ids)); self._render_block(child.children,lines,indent+1,names,input_types)
                if child.otherwise: lines.append((indent,"else:",ids)); self._render_block(child.otherwise,lines,indent+1,names,input_types)
            elif child.kind=="While":
                condition=self._normalize(child.condition,names)
                lines.append((indent,f"while {condition}:",ids)); self._render_block(child.children,lines,indent+1,names,input_types)
            elif child.kind=="For":
                variable=names.get(child.variable.casefold(),child.variable)
                expression=self._normalize(child.expression,names)
                lines.append((indent,f"for {variable} in {expression}:",ids)); self._render_block(child.children,lines,indent+1,names,input_types)
            elif child.kind in ("Start","End"):
                continue

    def _render_block(self,children,lines,indent,names,input_types):
        self._render(IRNode("Block",children=children),lines,indent,names,input_types)

    def _check_references(self, tree: ast.AST):
        """Catch obvious unresolved local names before presenting code to the user."""
        defined = {"print", "input", "int", "float", "str", "len", "range", "abs", "min", "max", "round", "__name__"}
        defined.update(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store): defined.add(node.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined:
                raise NameError(f"Generated code refers to undefined variable '{node.id}'.")

def run(context): return CodegenService().run(context)

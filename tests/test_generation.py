import ast
from backend.pipeline import analyze, generate
from tests.drawing import encode, sequence, branch, loop

def test_sequence_ir_pseudocode_and_python(settings):
    ctx=analyze(encode(sequence('amount = 31')),'user.png',settings)
    assert not ctx.has_errors
    assert ctx.ir.kind=='Program'
    assert [line.text for line in ctx.pseudocode]==['START','SET amount = 31','OUTPUT amount','END']
    ctx.approved=True; generate(ctx)
    source='\n'.join('    '*line.indent+line.text for line in ctx.code)
    ast.parse(source)
    assert 'amount = 31' in source and ctx.trace['element_to_code']['n2']

def test_branch_ir_has_else_and_trace(settings):
    ctx=analyze(encode(branch()),'branch.png',settings)
    assert not ctx.has_errors
    assert any(node.kind=='If' for node in ctx.ir.children)
    ctx.approved=True; generate(ctx)
    assert any(line.text=='else:' for line in ctx.code)
    assert not ctx.has_errors

def test_loop_ir_renders_while(settings):
    ctx=analyze(encode(loop()),'loop.png',settings)
    assert not ctx.has_errors,[d.to_dict() for d in ctx.diagnostics]
    assert any(node.kind=='While' for node in ctx.ir.children)
    ctx.approved=True; generate(ctx)
    assert any(line.text.startswith('while ') for line in ctx.code)

def test_generation_never_runs_code(settings):
    ctx=analyze(encode(sequence("__import__('os')")),'unsafe.png',settings)
    assert ctx.has_errors
    assert ctx.ir is None

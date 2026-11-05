"""Safe lowering of clear high-level flowchart operations."""
import ast

from backend.models import IRNode, PipelineContext
from backend.notation import validate_node_text
from backend.services.codegen_service import CodegenService


def factorial_ir():
    return IRNode("Program", children=[
        IRNode("Start"),
        IRNode("Input", text="Input number", variable="number", expression="str"),
        IRNode("If", condition="num < 0", children=[
            IRNode("Output", text='Output "Factorial is not defined for negative numbers"', expression='"Factorial is not defined for negative numbers"')
        ], otherwise=[
            IRNode("Process", text="Calculate factorial"),
            IRNode("Output", text="Output factorial", expression="factorial"),
        ]),
        IRNode("End"),
    ])


def test_calculate_factorial_is_a_supported_safe_process_command():
    validate_node_text("process", "Calculate factorial")
    validate_node_text("process", "Calculate the factorial")
    validate_node_text("process", "Calculate factorial of number")


def test_unrecognized_natural_language_is_not_implicitly_allowed():
    try:
        validate_node_text("process", "Do some magic")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown natural-language commands must remain rejected")


def test_factorial_command_lowers_to_compilable_python_and_infers_numeric_input():
    context = PipelineContext(ir=factorial_ir(), approved=True)
    CodegenService().run(context)
    assert not context.has_errors
    source = "\n".join("    " * line.indent + line.text for line in context.code)
    ast.parse(source)
    compile(source, "<test-generated>", "exec")
    assert "number = int(input())" in source
    assert "if number < 0:" in source
    assert "factorial = 1" in source
    assert "for _factorial_index in range(1, number + 1):" in source
    assert "factorial *= _factorial_index" in source
    assert "print(factorial)" in source


def test_factorial_code_runs_for_zero_and_positive_inputs():
    context = PipelineContext(ir=factorial_ir(), approved=True)
    CodegenService().run(context)
    source = "\n".join("    " * line.indent + line.text for line in context.code)
    namespace = {"__name__": "not_main"}
    exec(source, namespace)
    outputs = []
    import builtins
    original_input, original_print = builtins.input, builtins.print
    try:
        for value in ("0", "5"):
            builtins.input = lambda value=value: value
            builtins.print = lambda *args: outputs.append(" ".join(map(str, args)))
            namespace["main"]()
    finally:
        builtins.input, builtins.print = original_input, original_print
    assert outputs == ["1", "120"]

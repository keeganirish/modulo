# Flowchart recognition improvements

This update improves recognition of filled-color flowcharts without adding rules for particular labels or words.

## Changes

- Expanded arrowhead evidence checks to include small rasterized arrowheads.
- Added a constrained, warning-tagged direction fallback for connectors with exactly two node contacts where the input/decision/output roles provide a clear transition. Ambiguous connectors remain unresolved rather than being connected by reading order.
- Improved OCR cleanup for stray punctuation immediately after `If` and for a likely missing stroke in an equality operator. The original OCR reading remains available as a review alternative, and a bare single `=` is not rewritten without additional OCR-noise evidence.
- Improved branch-label inference for diagrams with a vertical branch and a sideways branch; inferred labels are marked with a warning for review.
- Accepted descriptive input labels such as `Input variable: Number` in the input notation parser.
- Added a regression fixture and tests for the supplied modulo/even-odd flowchart.

## Verification

- Unit/regression suite: `python -m pytest -q` — 50 passed, 14 skipped.
- Supplied flowchart integration test: `FLOWCHART_RUN_PIPELINE=1 python -m pytest tests/test_recognition_improvements.py::test_modulo_even_odd_flowchart_recognizes_nodes_edges_and_decision_text -q` — passed.

The supplied diagram does not draw a connector from the `Number is odd` output to `End`. The recognizer intentionally does not invent that missing edge. It reports the resulting incomplete path so the diagram can be corrected or reviewed.

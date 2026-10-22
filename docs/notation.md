# Supported notation — Build 01

Use a digitally drawn, single-page PNG, JPG, or PDF with dark, closed shape outlines on a light background. Shapes must not overlap. Use clear text and visible arrowheads. Handwriting and photographs are unsupported. This build analyzes geometry, OCR text, connectivity, and notation; it does not translate logic into pseudocode or Python yet.

## Shapes and text

A rounded rectangle or ellipse is a terminal and contains exactly `Start` or `End`, case-insensitively. There must be exactly one Start and at least one End. Start has no incoming arrow and one outgoing arrow. End has no outgoing arrow.

A rectangle is a process. It contains exactly one safe assignment, augmented assignment, or documented built-in call, such as `total = total + n` or `print(total)`. The single words `break`, `continue`, and `return` are also recognized. Loop control must be inside a loop. Statements are checked, not executed.

A parallelogram is input/output. Input uses `Input name` or `Input name:int`, with `float` and `str` also accepted. Output uses `Output expression`. Names begin with a letter, contain letters/numbers/underscores, and cannot contain double underscores or overwrite allowed built-ins.

A diamond is a decision and contains a supported expression, for example `n > 0`. A hexagon is an iteration and contains `WHILE condition` or `FOR name IN expression`, for example `FOR i IN range(5)`. Both need exactly two outgoing arrows, one Yes and one No. Also accepted: True/False and Y/N, in any case. Yes enters the loop body; No exits.

A small circle is an on-page connector. Exactly two circles use the same short label, such as `A`. One receives an incoming arrow; its match has an outgoing arrow. A structural graph edge joins them without changing their source IDs.

All other non-End shapes have one outgoing arrow. Unreachable shapes, unfinished paths, missing/duplicate branches, unpaired circles, and unsupported cycles produce errors. Structured IR recovery belongs to the next stage, so passing this build is not a guarantee that the diagram is translatable.

## Safe expression language

Numbers, strings, booleans, `None`, names, list/tuple literals, arithmetic, comparisons, `and`, `or`, and `not` are supported. Calls are limited to `print`, `input`, `int`, `float`, `str`, `len`, `range`, `abs`, `min`, `max`, and `round` with positional arguments. The authoritative AST allowlist is in `backend/notation.py`.

Imports, attributes, subscripts, double-underscore names, comprehensions, lambdas, f-strings, arbitrary calls, keyword arguments, and multiple statements per box are rejected. The server never executes diagram text.

## Recognition boundaries

Recognition is geometry-based, not specific to the sample. Arrow routing supports visible straight and orthogonal/bent strokes. Distinct routes are preferred. Ambiguous crossings and multi-way strokes with multiple possible sources and targets are flagged rather than resolved by layout assumptions. Direction requires visible arrowhead evidence; vertical order is never used as a substitute. Put each branch label close to its own route and away from shared stems.

A low-confidence OCR result produces a warning with the element ID. Even high-confidence OCR can misread punctuation. Always compare the recognized text and arrows with the source. This build does not let users edit recognized text in the browser; correct the source diagram and re-upload.

The default upload limit is 10 MB, the decoded image limit is 25 million pixels, the working width is 2000 pixels, the working height is capped at 6000, and the diagram limit is 100 recognized shapes. Configurable limits are documented in `backend/config.py`. Matching library and Tesseract versions are required for reproducible recognition.

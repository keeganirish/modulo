# Build 01: interface and recognition pipeline

This increment implements the README architecture in order: vanilla HTML/CSS/JavaScript UI, then intake, vision, graph construction, and validation. Testing follows implementation. Flask serves UI and API locally. There are no runtime AI calls, remote fonts, accounts, or external application services.

Passing analysis returns `status: "ready"` after logic recovery and pseudocode rendering; failed analysis returns `status: "failed"`. The server stores the IR and pseudocode, requires a separate `/api/approve` action, and accepts only the server-held `analysis_id` at `/api/generate`. Codegen parses and compiles generated Python without executing it, and Trace maps element IDs in both directions.

## Interface design

The green-accented workspace uses persistent source, pseudocode, and Python panels, shared tokens, system fonts, concise instructions, and a responsive stacked layout. Recognition results are shown alongside structured pseudocode, with flagged-region review before approval.

Consistency comes from shared controls and terminology. Universal usability comes from keyboard focus, a skip link, native dialog, responsive layout, and the clear status and progress feedback. Informative feedback includes file metadata, upload progress, indeterminate analysis progress, live status, and diagnostics. Closure is explicit in completion and failure messages. Error prevention uses preliminary file checks and independent server validation while preserving a valid selection after mistakes. Reversal includes replacement, cancellation, and Escape to unpin. User control means selecting a file never starts analysis and never generates code. Memory load is reduced by persistent geometry, recognized text, and a diagram guide. These decisions apply Shneiderman's rules [1], with practical examples from the Interaction Design Foundation [2].

Cancel aborts the browser request, not necessarily OCR already running on the server; the UI says so. Uploads stay in bounded, expiring memory. Replacement, reset, and page exit request deletion where the browser can deliver it. Only the theme persists in browser storage.

## Architecture and scope

Intake checks extension against content, size, page count, image dimensions, transparency, and normalized coordinates. Vision uses OpenCV enclosed contours and shape geometry, masked interior Tesseract OCR, outside edge-label OCR, and isolated connector strokes. Graph construction follows stroke components and arrowhead evidence rather than assuming a reading-order sequence. Ambiguous arrows and labels produce diagnostics instead of invented connections. Validation checks terminals, reachability, branch vocabulary, connector pairs, and a safe Python AST language.

The sample is an ordinary PNG uploaded through the same endpoint as any other diagram. Production services never load sample graphs or inspect filenames to decide the program. Independent fixtures exercise changed geometry, text, encodings, branches, bends, and loops at the end.

The existing repository layout is retained. Shared contracts live in `models.py`; tunables in `config.py`; supported notation and AST allowlists in `notation.py`. Each service receives and returns `PipelineContext` without Flask/store imports. `pipeline.py` orchestrates only. `app.py` handles HTTP. `analysis_store.py` owns temporary memory.

Dependencies are pinned. The local environment uses Python 3.12.9 and system Tesseract 5.5.3. Recognition reproducibility requires matching OpenCV/Tesseract versions. PyMuPDF retains the README's AGPL/commercial licensing question; no project license is chosen here.

## References

[1]: https://www.cs.umd.edu/users/ben/goldenrules.html "Ben Shneiderman: Eight Golden Rules of Interface Design"
[2]: https://ixdf.org/literature/article/shneiderman-s-eight-golden-rules-will-help-you-design-better-interfaces "Interaction Design Foundation: Shneiderman's Eight Golden Rules"

## Completed testing

Testing was run after the complete nine-stage pipeline and controlled review mode were implemented. The final suite passed **104 tests** on Python 3.12.9, OpenCV 4.12.0, and Tesseract 5.5.3. Recognition now covers outlined nodes and solid-color filled nodes, adaptive OCR, smart quotes, common `Display` output wording, intentional branch-label gaps, different assignment text, vertical and horizontal layouts, PNG/JPG, SVG, an independently drawn vector PDF, Yes/No branches with bent connectors, a WHILE loop/back edge, paired connector circles, repeated-input determinism, and the supplied `flowchart-2.png`. Logic/code tests cover IR structure, numbered pseudocode, approval enforcement, Python `ast.parse`/`compile`, unsafe input rejection, code coverage, and bidirectional trace maps. Unit/API tests cover signatures, corrupt/empty/oversized files, transparency, multipage PDFs, unsafe AST constructs, reserved variable names, reachability, branch labels, loop-control scope, storage expiry/eviction/removal, and server-held analysis IDs. The PyMuPDF SWIG bindings emit five deprecation warnings; there are no failing tests.

JavaScript syntax checks, Python compilation, dependency consistency, and `git diff --check` also passed. A local browser check exercised themes and the diagram guide. The application runs at `http://127.0.0.1:5001` for this session because macOS Control Center occupies port 5000. To restart it from the repo root, use `source .venv/bin/activate` followed by `PORT=5001 python backend/app.py`.

Passing recognition does not guarantee transcription accuracy or future translatability. In particular, OCR can still misread unusual fonts, low-resolution exports, or text touching borders; ambiguous crossings/arrowheads remain errors. Generation supports the structured IR forms `Sequence`, `If`, `While`, and `For`; irreducible graphs stop with `UNSTRUCTURED_FLOW` rather than inventing jumps. No code is executed during generation. No commit, push, deployment, or project license selection was performed.

The corrected server was also checked over local HTTP and through actual browser file uploads. The supplied filled-color parity flowchart completed with six shapes, six connections, no diagnostics, and pseudocode for the Yes/No branches. The normalized-image overlays rendered correctly, and the Approve pseudocode control remained disabled until review.

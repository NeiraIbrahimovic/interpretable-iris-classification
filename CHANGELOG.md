# Portfolio maintenance notes

The original classifier, notebook, dataset, and contributor history are retained.

## Correctness and compatibility

- Fixed independent validation of grouping and value columns. Invalid grouping columns previously reached pandas because the second check overwrote the first.
- Rejected empty column selections with the documented input-error category.
- Used scalar species grouping to avoid tuple keys breaking scatter plots with pandas 2.2.
- Used named mean/std aggregations to retain pandas sample-standard-deviation behavior.
- Added 11 regression checks, including the original thresholds and the 146/150 full-dataset result.

## Evaluation and presentation

- Added a separate, deterministic evaluation script with a stratified 60/40 split. It does not replace or retrain the manual classifier.
- Corrected the notebook's inaccurate saved-result interpretation in prose; original code cells and saved outputs remain unchanged.
- Preserved the original README in `docs/original-report.md`, including collaborator credits and historical figures.
- Added a recruiter-facing README, installation instructions, and explicit evaluation limitations.

These maintenance additions were prepared with AI assistance. They do not imply that one contributor authored the entire original team project.

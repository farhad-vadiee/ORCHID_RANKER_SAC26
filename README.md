# ORCHID-RANKER SAC'26 Artifact (Anonymous)

This repository packages the supplementary materials promised in the
SAC'26 submission:

- `appendix/` – compiled and source LaTeX for the full technical
  appendix referenced in the paper.
- `code/` – anonymised implementation, experiment drivers, and logged
  runs for the ORCHID-RANKER system.

## Getting started

1. Create a Python 3.9+ environment.
2. From `code/`, install the package in editable mode:

   ```bash
   pip install -e .
   ```

3. Review `code/data/README.md` for dataset acquisition and preprocessing
   steps (raw EdNet and OULAD releases are required).

All experiments can then be launched via the scripts in
`code/experiments` and `code/experiments-sac`, replicating the
evaluation reported in the manuscript. The `code/runs/` directory
contains the exact logs used for the paper's tables and figures.

## Repository layout

```text
appendix/
  appendix.pdf           # compiled supplementary material
  appendix.tex           # LaTeX source
  README.md
code/
  configs/               # experiment and dataset configuration files
  experiments/           # high-level experiment entrypoints
  experiments-sac/       # SAC submission experiment scripts
  runs/                  # cached run logs for reported tables/figures
  src/orchid_ranker/     # library source code
  requirements.txt       # optional dependency pinning
  pyproject.toml         # packaging metadata
  README.md
LICENSE
```

No identifying information is present in the artifact; all metadata and
URLs point to the anonymous 4open science mirror.

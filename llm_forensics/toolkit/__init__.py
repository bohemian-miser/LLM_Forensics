"""Measurement helpers for analysts looking for ROME/MEMIT edits in a model's weights.

One module per measurement or workflow helper (`info`, `spectra`, `corpus`, `wspectra`,
`selectivity`, `screen`, `vocab`, `whiten`, `trace`, `surgery`, `probe`, `plant`, `serve`, `run`);
each has a Python API taking a loaded model and runs as `python -m llm_forensics.toolkit.<name>`.
They report numbers and leave the reading to the analyst. See README.md in this directory.
"""

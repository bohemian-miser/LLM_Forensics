"""`llmf scan`: scan model weights for hidden edits. Plug-in API for detectors (docs/scanner.md):

    from llm_forensics.scan import Detector, DetectorResult, register_detector, Finding, Severity
"""
from .findings import Finding, Severity  # noqa: F401
from .registry import Detector, DetectorResult, register_detector  # noqa: F401

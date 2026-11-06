"""LLM Forensics: detect and explain edits hidden in LLM weights."""

__version__ = "0.1.1"


def _build() -> str:
    """__version__ plus the git commit of the checkout this package runs from (editable installs), e.g. 0.1.0+gff5f920.
    A '.dirty' suffix means uncommitted changes. Plain __version__ outside a git checkout."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, timeout=5)
        if sha.returncode:
            return __version__
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", "llm_forensics"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{__version__}+g{sha.stdout.strip()}{'.dirty' if dirty else ''}"
    except Exception:
        return __version__


BUILD = _build()

"""MODEL_LOCK serialises work on the shared suspect/base models when `llmf scan` runs detectors concurrently.

Weight-analysis detectors hold it for their whole run (some register forward hooks on the shared model); each
LLM-investigator tool call holds it for that call, so investigators overlap only in their API waits."""
import threading

MODEL_LOCK = threading.RLock()

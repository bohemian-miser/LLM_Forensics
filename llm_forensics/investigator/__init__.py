"""LLM investigator: an LLM agent (Claude / OpenAI / Gemini) with a constrained toolset that looks for
planted weight edits. Main entry point: :func:`investigate`."""
from .agent import AgentConfig, run_agent  # noqa: F401
from .api import LLM_MODES, build_env, investigate  # noqa: F401
from .providers import DEFAULT_MODELS, PRICES, default_model  # noqa: F401
from .raw_tools import RawInvestigationEnv  # noqa: F401
from .tools import InvestigationEnv  # noqa: F401

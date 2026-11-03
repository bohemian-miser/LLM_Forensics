"""LLM investigator: a Claude agent with a constrained toolset that looks for planted weight edits."""
from .agent import AgentConfig, run_agent  # noqa: F401
from .tools import InvestigationEnv  # noqa: F401

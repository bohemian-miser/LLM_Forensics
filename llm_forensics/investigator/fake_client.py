"""A scripted stand-in for ``anthropic.Anthropic`` used in tests / dry runs.

The fake exposes ``.messages.create`` / ``.beta.messages.create`` (Anthropic shape)
and ``.chat.completions.create`` (OpenAI Chat Completions shape) with the same
call shapes as the SDKs and returns objects with ``.content``, ``.stop_reason`` and
``.usage``. Its behaviour is a *policy*: a callable that receives the list of
tool-result strings seen so far and returns either a list of ``(tool_name, input)``
calls for this turn or a final text string.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

Policy = Callable[[list[str]], "list[tuple[str, dict]] | str"]


@dataclass
class _Block:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None

    def model_dump(self, exclude_none: bool = False) -> dict:
        d = {"type": self.type, "text": self.text, "id": self.id, "name": self.name, "input": self.input}
        return {k: v for k, v in d.items() if v is not None} if exclude_none else d


class _Messages:
    def __init__(self, owner: "ScriptedClient"):
        self.owner = owner

    def create(self, **kwargs):
        return self.owner._create(**kwargs)


class ScriptedClient:
    def __init__(self, policy: Policy, openai_api: str = "chat"):
        self.policy = policy
        if openai_api == "responses":
            self.responses = SimpleNamespace(create=self._create_responses)
        self._resp_results: list[str] = []
        self.messages = _Messages(self)
        self.beta = SimpleNamespace(messages=_Messages(self))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_openai))
        self.requests: list[dict] = []
        self._ids = itertools.count()

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        results: list[str] = []
        for m in kwargs["messages"]:
            if m["role"] == "user" and isinstance(m["content"], list):
                results += [c["content"] for c in m["content"] if isinstance(c, dict) and c.get("type") == "tool_result"]
        action = self.policy(results)
        usage = SimpleNamespace(input_tokens=1000 + 10 * sum(map(len, results)) // 40, output_tokens=200,
                                cache_read_input_tokens=0, cache_creation_input_tokens=0)
        if isinstance(action, str):
            return SimpleNamespace(content=[_Block("text", text=action)], stop_reason="end_turn", usage=usage)
        blocks = [_Block("text", text=f"(scripted turn {len(results)})")]
        for name, inp in action:
            blocks.append(_Block("tool_use", id=f"toolu_{next(self._ids):04d}", name=name, input=inp))
        return SimpleNamespace(content=blocks, stop_reason="tool_use", usage=usage)


    def _create_openai(self, **kwargs):
        self.requests.append(kwargs)
        results = [m["content"] for m in kwargs["messages"] if m.get("role") == "tool"]
        action = self.policy(results)
        usage = SimpleNamespace(prompt_tokens=1000 + sum(map(len, results)) // 4, completion_tokens=200,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        if isinstance(action, str):
            msg = SimpleNamespace(content=action, tool_calls=None, refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)
        calls = [SimpleNamespace(id=f"call_{next(self._ids):04d}", type="function",
                                 function=SimpleNamespace(name=n, arguments=json.dumps(a))) for n, a in action]
        msg = SimpleNamespace(content=None, tool_calls=calls, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=usage)


    def _create_responses(self, **kwargs):
        self.requests.append(kwargs)
        self._resp_results += [i["output"] for i in kwargs["input"] if i.get("type") == "function_call_output"]
        action = self.policy(list(self._resp_results))
        n = next(self._ids)
        usage = SimpleNamespace(input_tokens=1000, output_tokens=200,
                                input_tokens_details=SimpleNamespace(cached_tokens=100))
        if isinstance(action, str):
            out = [SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=action)])]
        else:
            out = [SimpleNamespace(type="function_call", call_id=f"call_{n}_{j}", name=a, arguments=json.dumps(b))
                   for j, (a, b) in enumerate(action)]
        return SimpleNamespace(id=f"resp_{n}", status="completed", incomplete_details=None, output=out, usage=usage)


def sequence_policy(turns: list[Any]) -> Policy:
    """Policy that plays back a fixed list of turns; each item is a list of calls,
    a final string, or a callable(results) -> calls/str."""
    state = {"i": 0}

    def policy(results):
        i = state["i"]
        state["i"] += 1
        if i >= len(turns):
            return "done"
        t = turns[i]
        return t(results) if callable(t) else t

    return policy


# --------------------------------------------------------------------------- #
# A simple heuristic "investigator" policy used for end-to-end tests.
# It parses tool outputs the way a real agent would and submits a report.
# --------------------------------------------------------------------------- #
def heuristic_policy(mode: str, mlp_glob: str, candidate_subjects: list[str],
                     probe_prompt: str) -> Policy:
    st: dict[str, Any] = {}

    def first_tensor_from_table(text: str) -> str | None:
        for line in text.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 3 and parts[0].endswith(".weight"):
                return parts[0]
        return None

    def most_anomalous(text: str) -> str | None:
        best, best_v = None, -1.0
        for line in text.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 11 and parts[0].endswith(".weight"):
                try:
                    v = float(parts[9])  # s1/s2
                except ValueError:
                    continue
                if v > best_v:
                    best, best_v = parts[0], v
        return best

    def turn0(_):
        return [("list_tensors", {"pattern": mlp_glob})]

    def turn1(_):
        if mode == "with_original":
            return [("diff_stats", {"name": "*"})]
        return [("tensor_stats", {"name": mlp_glob})]

    def turn2(results):
        st["tensor"] = (first_tensor_from_table(results[-1]) if mode == "with_original"
                        else most_anomalous(results[-1]))
        st["layer"] = int(re.search(r"\.(\d+)\.", st["tensor"]).group(1))
        diff = mode == "with_original"
        st["diff"] = ":diff" if diff else ""
        return [("svd_topk", {"name": st["tensor"], "k": 2, "use_diff": diff}),
                ("unembed_vector", {"source": st["tensor"], "which": f"svd_out:0{st['diff']}"})]

    def turn3(results):
        m = re.search(r"\+dir top: \"(.*?)\"\(", results[-1])
        m_neg = re.search(r"-dir top: \"(.*?)\"\(", results[-1])
        st["target"] = m.group(1) if m else "?"
        st["target_neg"] = m_neg.group(1) if m_neg else "?"
        return [("mlp_key_similarity", {"layer": st["layer"],
                                        "vector_ref": f"svd:{st['tensor']}:0:in{st['diff']}",
                                        "texts": candidate_subjects}),
                ("run_prompt", {"prompt": probe_prompt, "max_new_tokens": 5}),
                ("read_slice", {"name": st["tensor"], "rows": "0:4", "cols": [0, 1, 2]})]

    def turn4(results):
        sim = results[-3]
        rows = [l for l in sim.splitlines()[2:] if "|" in l]
        subj = rows[0].split("|")[0].strip().strip('"') if rows else "?"
        score = float(rows[0].split("|")[1]) if rows else 0.0
        return [("submit_report", {
            "detected": True, "confidence": 0.8,
            "suspected_edits": [{
                "layer": st["layer"], "tensor": st["tensor"], "rank": 1,
                "subject_candidates": [{"text": subj, "score": score}],
                "target_candidates": [{"token": st["target"], "score": 1.0},
                                      {"token": st["target_neg"], "score": 0.5}],
                "evidence": "heuristic policy: top singular direction of anomalous tensor",
            }],
            "notes": "scripted heuristic investigator"})]

    return sequence_policy([turn0, turn1, turn2, turn3, turn4])

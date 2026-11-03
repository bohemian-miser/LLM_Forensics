"""Fake LLM client for the investigator tests (no API calls).

It exposes ``.messages.create`` / ``.beta.messages.create`` (Anthropic shape)
``.chat.completions.create`` / ``.responses.create`` (OpenAI) and
``.models.generate_content`` (google-genai) with the same
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
    def __init__(self, owner: "FakeLLMClient"):
        self.owner = owner

    def create(self, **kwargs):
        return self.owner._create(**kwargs)


class FakeLLMClient:
    def __init__(self, policy: Policy, openai_api: str = "chat"):
        self.policy = policy
        if openai_api == "responses":
            self.responses = SimpleNamespace(create=self._create_responses)
        self._resp_results: list[str] = []
        self.models = SimpleNamespace(generate_content=self._create_gemini)
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
        blocks = [_Block("text", text=f"(turn {len(results)})")]
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


    def _create_gemini(self, model, contents, config):
        from google.genai import types
        self.requests.append({"model": model, "contents": contents, "config": config})
        results = []
        for c in contents:
            for p in (c.parts or []):
                if p.function_response is not None:
                    r = p.function_response.response or {}
                    results.append(r.get("result", r.get("error", "")))
        action = self.policy(results)
        usage = SimpleNamespace(prompt_token_count=1200, cached_content_token_count=200,
                                candidates_token_count=150, thoughts_token_count=50)
        if isinstance(action, str):
            parts = [types.Part(text=action)]
        else:
            # first call of a turn gets no id, to exercise the adapter's id synthesis
            parts = [types.Part(function_call=types.FunctionCall(
                id=None if j == 0 else f"fc_{next(self._ids)}", name=n, args=a)) for j, (n, a) in enumerate(action)]
        cand = SimpleNamespace(content=types.Content(role="model", parts=parts), finish_reason=types.FinishReason.STOP)
        return SimpleNamespace(candidates=[cand], usage_metadata=usage, prompt_feedback=None)


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


# Heuristic policy for end-to-end tests: parses tool outputs the way an agent would and
# submits a report.
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
        if mode == "with_base":
            return [("diff_stats", {"name": "*"})]
        return [("tensor_stats", {"name": mlp_glob})]

    def turn2(results):
        st["tensor"] = (first_tensor_from_table(results[-1]) if mode == "with_base"
                        else most_anomalous(results[-1]))
        st["layer"] = int(re.search(r"\.(\d+)\.", st["tensor"]).group(1))
        diff = mode == "with_base"
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
            "notes": "heuristic test policy"})]

    return sequence_policy([turn0, turn1, turn2, turn3, turn4])


# Raw-mode policy: only weight-reading tools. It picks the tensor with the largest change (with
# a base) or the largest s1/s2 (weights only), compares the top output singular vector with the
# unembedding rows of candidate targets, and reports the best match.
def raw_policy(mode: str, mlp_glob: str, candidate_targets: list[str]) -> Policy:
    st: dict[str, Any] = {}

    def turn0(_):
        if mode == "with_base":
            return [("diff_stats", {"name": "*"})]
        return [("tensor_stats", {"name": mlp_glob})]

    def turn1(results):
        best, best_v = None, -1.0
        for line in results[-1].splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 3 and parts[0].endswith(".weight"):
                try:
                    v = 1e9 if mode == "with_base" else float(parts[9])  # diff_stats is sorted already
                except ValueError:
                    continue
                if v > best_v:
                    best, best_v = parts[0], v
                if mode == "with_base":
                    break
        st["tensor"] = best
        st["layer"] = int(re.search(r"\.(\d+)\.", best).group(1))
        st["diff"] = ":diff" if mode == "with_base" else ""
        return [("singular_vectors", {"name": best, "k": 2, "use_diff": mode == "with_base"}),
                ("norms", {"name": best, "axis": "col", "top": 5}),
                ("top_entries", {"name": best, "k": 5})]

    def turn2(_):
        ref = f"svd:{st['tensor']}:0:out{st['diff']}"
        return [("compare_vectors", {"ref": ref, "others": [f"tok:{t}" for t in candidate_targets],
                                     "center_tokens": True}),
                ("read_vector", {"ref": ref, "top": 8})]

    def turn3(results):
        rows = [l for l in results[-2].splitlines()[2:] if l.count("|") == 3]
        st["targets"] = [(r.split("|")[0].strip()[4:], abs(float(r.split("|")[2]))) for r in rows]
        return [("token_rows", {"texts": [st["targets"][0][0]], "top": 5}),
                ("read_block", {"name": st["tensor"], "rows": "0:3", "cols": "0:3"})]

    def turn4(_):
        return [("submit_report", {
            "detected": True, "confidence": 0.7,
            "suspected_edits": [{
                "layer": st["layer"], "tensor": st["tensor"], "rank": 1, "subject_candidates": [],
                "target_candidates": [{"token": t, "score": s} for t, s in st["targets"][:3]],
                "evidence": "raw policy: top output singular vector against candidate unembedding rows",
            }],
            "notes": "raw test policy"})]

    return sequence_policy([turn0, turn1, turn2, turn3, turn4])

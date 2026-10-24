"""BFCL v3 slice as a CROSS-DISTRIBUTION held-out eval (contract §9a.3, B2).

The student is trained on ToolACE; BFCL is a different distribution, so a gain
here is generalisation, not memorisation. Questions are rendered in the exact
ToolACE system-prompt template the student was trained on (prefix + tool JSON +
suffix, read verbatim from toolace-fc-v1), so the only thing that changes is
the data, not the instructions.

BFCL ground truth is ANY-OF per argument: {"fn": {"arg": [v1, v2, ""]}}, where
"" means the argument may be omitted. It is graded by `BFCLGrader`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from traceforge.config import REPO_ROOT, load_settings
from traceforge.eval import EvalItem

REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")

# Verbatim from data/datasets/toolace-fc-v1 system prompts (243/250 rows share the suffix).
PREFIX = ("You are an expert in composing functions. You are given a question and a set of possible "
          "functions. \nBased on the question, you will need to make one or more function/tool calls to "
          "achieve the purpose. \nIf none of the function can be used, point it out. If the given question "
          "lacks the parameters required by the function,\nalso point it out.\n"
          "Here is a list of functions in JSON format that you can invoke:\n")
SUFFIX = (". \nShould you decide to return the function call(s). \nPut it in the format of "
          "[func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n\n"
          "NO other text MUST be included. \n")


def _download(name: str) -> Path:
    from huggingface_hub import hf_hub_download
    load_settings(require_key=False)              # loads .env so HF_TOKEN is visible
    return Path(hf_hub_download(REPO_ID, name, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                                cache_dir=str(REPO_ROOT / "data" / "hf_cache")))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_bfcl(per_category: int = 40, categories: tuple[str, ...] = CATEGORIES) -> list[EvalItem]:
    """Evenly spaced `per_category` items from each category (deterministic).

    item_id is BFCL's own id ("parallel_multiple_7"), so results are traceable
    back to the public benchmark.
    """
    items: list[EvalItem] = []
    for cat in categories:
        qs = _jsonl(_download(f"BFCL_v3_{cat}.json"))
        answers = {} if cat == "irrelevance" else \
            {a["id"]: a["ground_truth"] for a in _jsonl(_download(f"possible_answer/BFCL_v3_{cat}.json"))}
        step = max(1, len(qs) // per_category)
        for q in qs[::step][:per_category]:
            turns = q["question"][0]                 # single-turn categories: one conversation
            system = PREFIX + json.dumps(q["function"]) + SUFFIX
            msgs = [{"role": "system", "content": system}] + [
                {"role": t["role"], "content": t["content"]} for t in turns]
            if cat == "irrelevance":
                gold: dict = {"bfcl": []}            # correct behaviour: call nothing
            else:
                if q["id"] not in answers:
                    raise KeyError(f"no possible_answer for {q['id']}")
                gold = {"bfcl": answers[q["id"]]}
            items.append(EvalItem(q["id"], msgs, gold))
    return items

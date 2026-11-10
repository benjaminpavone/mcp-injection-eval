"""CLI.

  python -m mcp_injection_eval.run --models fake-obedient fake-robust      # offline dry run
  python -m mcp_injection_eval.run --models gemini-flash --repeats 3
  python -m mcp_injection_eval.run --list-models gemini-flash             # ids served by the provider

Results are appended to a JSONL file line by line; re-running with the same --out
skips combinations already done (handy with free-tier rate limits).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from pathlib import Path

from dotenv import load_dotenv

from .agent import run_one
from .llm import FakeLLM, load_models, make_llm
from .scenarios import ATTACKS, DEFENSES, SCENARIOS

FAKES = {"fake-obedient": True, "fake-robust": False}


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r["error"]:
            keys.add((r["model"], r["scenario"], r["attack"], r["defense"], r["repeat"]))
    return keys


async def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["fake-obedient", "fake-robust"])
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    p.add_argument("--defenses", nargs="+", default=list(DEFENSES))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/runs.jsonl"))
    p.add_argument("--list-models", metavar="MODEL_KEY")
    a = p.parse_args()

    if a.list_models:
        llm = make_llm(a.list_models)
        async for m in llm.client.models.list():
            print(m.id)
        return

    unknown = [m for m in a.models if m not in FAKES and m not in load_models()]
    if unknown:
        p.error(f"unknown model(s) {unknown}; see models.toml")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(a.out)
    real = {m: make_llm(m) for m in a.models if m not in FAKES}
    combos = list(itertools.product(a.models, a.scenarios, a.attacks, a.defenses, range(a.repeats)))
    todo = [c for c in combos if c not in done]
    print(f"{len(combos)} runs, {len(combos) - len(todo)} already done -> {len(todo)} to go")

    # One worker per model, running concurrently: each model has its own rate limit.
    counter = itertools.count(1)
    stopped: list[str] = []

    async def worker(model: str, items: list[tuple], f) -> None:
        for _, sc, at, de, rep in items:
            scenario = SCENARIOS[sc]
            llm = FakeLLM(scenario, FAKES[model]) if model in FAKES else real[model]
            r = await run_one(llm, scenario, ATTACKS[at], DEFENSES[de], rep)
            label = f"[{next(counter)}/{len(todo)}] {model:17} {sc:14} {at:21} {de:18} #{rep}"
            if r.error.startswith("QuotaExhausted"):
                # Not written: the combo stays "to do" and will run on the next invocation.
                print(f"{label}  STOP {model}: {r.error}", flush=True)
                stopped.append(model)
                return
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            flag = (
                "ERROR " + r.error[:300]
                if r.error
                else (
                    f"attack={'YES' if r.attack_success else 'no '} exfil={'YES' if r.exfiltrated else 'no '} "
                    f"utility={'ok' if r.utility else 'FAIL'}"
                )
            )
            print(f"{label}  {flag}", flush=True)

    with a.out.open("a") as f:
        by_model = {m: [c for c in todo if c[0] == m] for m in a.models}
        await asyncio.gather(*(worker(m, items, f) for m, items in by_model.items() if items))

    if stopped:
        print(f"\nDaily quota reached for: {', '.join(stopped)}. Re-run the same command tomorrow to resume.")


if __name__ == "__main__":
    asyncio.run(main())

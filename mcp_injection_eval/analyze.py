"""Aggregate results per model / attack / defense.

  python -m mcp_injection_eval.analyze                    # every results/*.jsonl
  python -m mcp_injection_eval.analyze results/claude.jsonl

Every saved run is re-judged with the current judge, so judge improvements
(e.g. the partial-compliance metric added in v2) apply to older runs too.
Cells are shown as "k/n" because samples are small.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .agent import RunResult, judge
from .scenarios import ATTACKS, SCENARIOS

FIELDS = set(RunResult.__dataclass_fields__)


def load(paths: list[str]) -> pd.DataFrame:
    rows = []
    lines = [ln for path in paths for ln in Path(path).read_text().splitlines()]
    for line in lines:
        raw = json.loads(line)
        r = judge(RunResult(**{k: v for k, v in raw.items() if k in FIELDS}), SCENARIOS[raw["scenario"]])
        d = r.to_dict()
        attack = ATTACKS.get(r.attack)
        d["goal"] = attack.goal if attack else "?"
        d["version"] = attack.version if attack else 0
        rows.append(d)
    return pd.DataFrame(rows)


def kn(s: pd.Series) -> str:
    return f"{int(s.sum())}/{len(s)}"


def table(df: pd.DataFrame, index: str, columns: str, value: str) -> str:
    return df.pivot_table(index=index, columns=columns, values=value, aggfunc=kn, fill_value="-").to_string()


def main(*paths: str) -> None:
    explicit = bool(paths)
    paths = paths or tuple(sorted(str(p) for p in Path("results").glob("*.jsonl")))
    df = load(list(paths))
    if not explicit:  # scripted fakes from dry runs are not results
        df = df[~df["model"].str.startswith("fake-")]
    errors = df[df["error"] != ""]
    # Keep the last successful run per combination (failed attempts may have been re-run).
    df = df[df["error"] == ""].drop_duplicates(
        ["model", "scenario", "attack", "defense", "repeat"], keep="last"
    )
    print(f"{len(df)} runs ({len(errors)} errored attempts ignored)\n")

    attacked = df[df["attack"] != "none"]
    control = df[df["attack"] == "none"]

    sections = [
        (
            "Attack success (email sent to attacker) - model x attack",
            attacked,
            "model",
            "attack",
            "attack_success",
        ),
        ("Partial compliance (off-task action) - model x attack", attacked, "model", "attack", "partial"),
        ("Reconnaissance (list_files) - model x attack", attacked, "model", "attack", "recon"),
        ("Reconnaissance, control runs (baseline habit)", control, "model", "scenario", "recon"),
        (
            "Partial compliance, control runs (false-positive baseline)",
            control,
            "model",
            "defense",
            "partial",
        ),
        ("Attack success - model x defense", attacked, "model", "defense", "attack_success"),
        ("Partial compliance - model x defense", attacked, "model", "defense", "partial"),
        ("Answer warns the user (heuristic) - model x attack", attacked, "model", "attack", "flagged"),
        (
            "Successful attacks the answer discloses to the user (heuristic)",
            attacked[attacked["attack_success"]],
            "model",
            "attack",
            "disclosed",
        ),
        ("Utility, control runs (no attack) - model x defense", control, "model", "defense", "utility"),
        ("Utility under attack - model x defense", attacked, "model", "defense", "utility"),
    ]
    for title, data, index, columns, value in sections:
        if len(data):
            print(f"== {title}\n{table(data, index, columns, value)}\n")

    if len(attacked):
        out = Path(paths[0]).parent / "summary.png"
        heatmap(attacked, out)
        print(f"chart -> {out}")


# Sequential single-hue ramp (light = near zero, dark = always succeeds).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, SURFACE = "#1f1f1d", "#8a8985", "#ffffff"


def heatmap(attacked: pd.DataFrame, out: Path) -> None:
    """Model x attack grid of attack success, each cell labelled k/n.

    A pooled bar chart would be dominated by the models with the most runs; a grid keeps
    every model's sample size visible. Cells never run are left blank.
    """
    attacks = [a for a in ATTACKS if a in set(attacked["attack"])]
    by = attacked.groupby(["model", "attack"])["attack_success"]
    k, n = by.sum().unstack(), by.count().unstack()
    rate = (k / n).reindex(columns=attacks)
    models = rate.mean(axis=1).sort_values(ascending=False).index  # most vulnerable first
    rate, k, n = (t.reindex(index=models, columns=attacks) for t in (rate, k, n))

    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(1.15 * len(attacks) + 3, 0.5 * len(models) + 2))
    im = ax.imshow(rate.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, j in itertools.product(range(len(models)), range(len(attacks))):
        if pd.isna(n.iat[i, j]):
            ax.text(j, i, "-", ha="center", va="center", color=MUTED, fontsize=9)
            continue
        dark = rate.iat[i, j] > 0.55
        ax.text(
            j,
            i,
            f"{int(k.iat[i, j])}/{int(n.iat[i, j])}",
            ha="center",
            va="center",
            color=SURFACE if dark else INK,
            fontsize=9,
        )
    ax.set_xticks(range(len(attacks)), attacks, rotation=30, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(models)), models, fontsize=9, color=INK)
    # 2px surface gaps between cells, no frame.
    ax.set_xticks([x - 0.5 for x in range(1, len(attacks))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("attack success rate", color=INK, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, labelcolor=MUTED)
    ax.set_title(
        "Attack success per model and attack (all defenses pooled; cell = successes / runs)",
        fontsize=11,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])

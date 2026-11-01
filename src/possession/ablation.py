"""Label-free ablation over the switches in config.py. Never touches any ground truth.

    python -m src.possession.ablation --cache data/cache/features_v2.npz

Runs every variant in memory and writes no file: run_evaluation.py scores every CSV in
evaluation/results/, and these variants must not be scored. Prints aggregates only.
"""

import argparse
from itertools import product

from games import GAME, GAMES

from .config import RESTART_RULES, ModelConfig
from .diagnostics import load_smart_tagging, summarize
from .pipeline import run


def row(config, cache, tags):
    labels, _, summary, blocks = run(cache=cache, config=config)
    out = summarize(labels, tags, blocks, GAMES[GAME]["fps"])
    return summary, {
        "live": sum(r["live_min"] for _, r in out["halves"]),
        "poss": sum(r["possessions"] for _, r in out["halves"]),
        "short": sum(r["under_2s"] for _, r in out["halves"]),
        "attack": out["agreement"]["attacking"],
        "defend": out["agreement"]["defending"],
        "coverage": out["coverage"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    tags = load_smart_tagging(GAME)

    print(
        "{:<14} {:<6} {:>8} {:>5} {:>5} {:>7} {:>7} {:>6} {:>9} {:>11}".format(
            "restart_rule",
            "merge",
            "live min",
            "poss",
            "<2 s",
            "attack",
            "defend",
            "set pc",
            "tag blind",
            "next poss",
        )
    )
    for rule, merge in product(RESTART_RULES, (True, False)):
        summary, r = row(ModelConfig(touch_merge=merge, restart_rule=rule), args.cache, tags)
        print(
            "{:<14} {:<6} {:>8.1f} {:>5} {:>5} {:>7.3f} {:>7.3f} {:>6.3f} {:>9} {:>11}".format(
                rule,
                "on" if merge else "off",
                r["live"],
                r["poss"],
                r["short"],
                r["attack"],
                r["defend"],
                r["coverage"],
                "{}/{}".format(*summary["validation"][rule]),
                "{}/{}".format(*summary["consistency"]),
            )
        )
        if merge:
            print(
                "  rules: "
                + "  ".join(f"{k} {v}" for k, v in summary["rules"].items())
                + "  control-only blind {}/{}".format(*summary["validation"]["control_only"])
                + (
                    "  median latency {:.2f} s".format(summary["control_latency_s"])
                    if summary["control_latency_s"] is not None
                    else ""
                )
            )


if __name__ == "__main__":
    main()

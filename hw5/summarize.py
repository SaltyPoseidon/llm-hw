import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "results"
CONFIGS = ("base", "exact", "partial", "dr-grpo")
GROUPS = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")

def verify_predictions(directory, benchmark, metrics):
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    keys = [(row["id"], row["category"]) for row in predictions]
    if keys != benchmark:
        raise ValueError(f"{directory.name}: prediction IDs or categories differ from BFCL")
    totals = Counter(row["category"] for row in predictions)
    correct = Counter(row["category"] for row in predictions if row["check"]["valid"])
    for category, counts in metrics["categories"].items():
        if (counts["correct"], counts["total"]) != (correct[category], totals[category]):
            raise ValueError(f"{directory.name}: metric counts disagree with predictions")


if __name__ == "__main__":
    dataset = (ROOT.parent / "bfcl.jsonl").read_bytes()
    rows = [json.loads(line) for line in dataset.splitlines()]
    benchmark = [(row["id"], row["category"]) for row in rows]
    digest = hashlib.sha256(dataset).hexdigest()
    table = [
        "| Запуск | Overall | Simple | Multiple | Parallel | Parallel multiple | Irrelevance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fingerprints = set()
    for config in CONFIGS:
        directory = ROOT / config
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics["benchmark_sha256"] != digest:
            raise ValueError(f"{config}: BFCL file differs from the course benchmark")
        verify_predictions(directory, benchmark, metrics)
        fingerprints.add(json.dumps(metrics["decoding"], sort_keys=True))
        values = [metrics["overall"], *(metrics["groups"][group] for group in GROUPS)]
        table.append("| " + " | ".join([config, *(f"{value:.1%}" for value in values)]) + " |")
    if len(fingerprints) != 1:
        raise ValueError("Benchmark files or decoding settings differ")
    (ROOT / "table.md").write_text("\n".join(table) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for config in CONFIGS[1:]:
        history = json.loads((ROOT / config / "history.json").read_text())
        for axis, field in zip(axes, ("loss", "reward")):
            points = [(row["step"], row[field]) for row in history if field in row]
            if not points:
                raise ValueError(f"{config}: missing {field}")
            steps, values = zip(*points)
            axis.plot(steps, values, label=config)
            axis.set_ylabel(field)
            axis.legend()
    axes[-1].set_xlabel("Шаг обучения")
    fig.tight_layout()
    fig.savefig(ROOT / "training.png", dpi=160)

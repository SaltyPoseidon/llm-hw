import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

source, output = map(Path, sys.argv[1:])
groups = {}
with source.open() as file:
    for row in csv.DictReader(file):
        if "-load-" not in row["run"]:
            continue
        config, rpm = row["run"].rsplit("-load-", 1)
        groups.setdefault(config, []).append((int(rpm), float(row["p90_s"])))

if not groups:
    raise ValueError("No load-test results")

for config, points in sorted(groups.items()):
    rpm, p90 = zip(*sorted(points))
    plt.plot(rpm, p90, "o-", label=config)
plt.axhline(15, color="black", linestyle="--", label="15 s")
plt.xlabel("RPM")
plt.ylabel("P90 E2E latency, s")
plt.legend()
plt.tight_layout()
plt.savefig(output, dpi=160)

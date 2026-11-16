"""Export the numbers the public website shows from one reference run.

    python scripts/export_site_data.py [RUN_ID]

Writes site/site_data.js (window.NB_DATA = {...}); site/build.py inlines it.
Nothing is computed here: values are copied from the run's investigation.json,
figures and reproducibility record, so the site cannot drift from the run.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = "nb-20261113T112849-181fce"
COHORT = {"tcga_gbm_rnaseq": "TCGA-GBM", "cgga_325": "CGGA-325", "cgga_693": "CGGA-693"}
CANDIDATES = ["NLGN3", "SLC17A7", "GRIA1", "GRIA2", "GRIN2B", "BDNF"]


def decode(arr):
    """Plotly JSON stores numeric arrays as {dtype, bdata, shape}."""
    if isinstance(arr, dict) and "bdata" in arr:
        import numpy as np

        a = np.frombuffer(base64.b64decode(arr["bdata"]), dtype=arr["dtype"])
        if "shape" in arr:
            a = a.reshape([int(s) for s in str(arr["shape"]).split(",")])
        return a.tolist()
    return arr


def r(x, n=4):
    return None if x is None else round(float(x), n)


def validation(v: dict) -> dict:
    return {
        "hypothesis": v["hypothesis"],
        "verdict": v["verdict"],
        "rationale": v["verdict_rationale"],
        "pooled": [r(v["pooled_estimate"]), r(v["pooled_ci_low"]), r(v["pooled_ci_high"])],
        "pooledP": r(v["pooled_p"], 6),
        "i2": r(v["heterogeneity_i2"], 1),
        "replicating": v["n_cohorts_replicating"],
        "cohorts": [
            {
                "id": c["cohort_id"],
                "name": COHORT.get(c["cohort_id"], c["cohort_id"]),
                "role": c["role"],
                "n": c["n"],
                "events": c["events"],
                "est": r(c["estimate"]),
                "lo": r(c["ci_low"]),
                "hi": r(c["ci_high"]),
                "p": r(c["p"], 6),
                "estimate": c["estimate_name"],
                "consistent": c["direction_consistent"],
                "significant": c["significant"],
                "notes": c["notes"],
            }
            for c in v["per_cohort"]
        ],
    }


def main(run_id: str) -> None:
    run = ROOT / "data" / "runs" / run_id
    inv = json.loads((run / "investigation.json").read_text())
    repro = json.loads((run / "reproducibility.json").read_text())
    art, log = inv["artefacts"], inv["log"]
    vals = {v["hypothesis"]: v for k, v in art.items() if k.startswith("validation:")}
    by_kind = {}
    for v in vals.values():
        if "survival" in v["hypothesis"]:
            by_kind["survival"] = v
        elif "residualised" in v["hypothesis"]:
            by_kind["subtypeAdjusted"] = v
        else:
            by_kind["subtype"] = v

    fig = json.loads((run / "figures" / "validation_subtype_neuronal_activity_candidates.json").read_text())
    rule = fig["layout"]["meta"]["caption"].split("Replication rule: ", 1)[-1]
    hm = json.loads((run / "figures" / "subtype_heatmap_neuron_glioma_extended_tcga_gbm_rnaseq.json").read_text())
    trace = hm["data"][0]
    stats = next(m for m in log["messages"] if m["agent"] == "Statistics Agent")
    validator = next(m for m in log["messages"] if m["agent"] == "Validation Agent")
    comp = art["composition:029"]
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None

    data = {
        "run": {
            "id": inv["investigation_id"],
            "question": inv["question"],
            "status": inv["status"],
            "started": log["started_at"],
            "finished": log["finished_at"],
            "seed": repro.get("seed"),
            "software": repro.get("software", {}),
            "files": [
                {"key": k, "source": f.get("source"), "sha256": f.get("sha256", "")[:12], "retrieved": f.get("retrieved_at")}
                for k, f in sorted(repro.get("data_manifest", {}).items())
            ],
            "exportedFromCommit": sha,
        },
        "agents": [
            {
                "agent": m["agent"],
                "role": m["stage"],
                "at": m["timestamp"],
                "status": m["status"],
                "inputs": m["inputs"],
                "outputs": m["outputs"],
                "limitations": m["limitations"],
            }
            for m in log["messages"]
        ],
        "calls": [
            {
                "agent": c["agent"],
                "tool": c["tool"],
                "start": c["started_at"],
                "end": c["finished_at"],
                "s": c["duration_s"],
                "status": c["status"],
                "out": c["output_summary"],
                "lim": c["limitations"][:2],
            }
            for c in log["tool_calls"]
        ],
        "literature": {
            # same count the search_literature tool reports as "N verified records" (len of the run's citations)
            "verified": len(art.get("citations", {})),
            "keyReferences": art["outputs"]["literature"]["data"].get("key_references", {}).get("retrieved"),
        },
        "claims": [{"id": c["id"], "kind": c["kind"], "level": c["level"], "status": c["status"]} for c in inv["claims"]],
        "review": inv["review"],
        "validation": {k: validation(v) for k, v in by_kind.items()},
        "replicationRule": rule,
        "projection": validator["outputs"].get("projection"),
        "sensitivity": validator["outputs"].get("sensitivity"),
        "composition": {
            "summary": stats["outputs"].get("composition"),
            "cohorts": [
                {"name": COHORT.get(c["cohort"], c["cohort"]), "rho": r(c["spearman_rho"], 3), "p": c["spearman_p"], "n": c["n"]}
                for c in comp["rows"]
            ],
            "limitations": [l for l in stats["limitations"] if "Neuron-marker" in l or "Residualising" in l],
        },
        "heatmap": {
            "genes": trace["y"],
            "groups": trace["x"],
            "z": [[r(v, 3) for v in row] for row in decode(trace["z"])],
            "caption": hm["layout"]["meta"]["caption"],
            "candidates": CANDIDATES,
        },
    }
    out = ROOT / "site" / "site_data.js"
    out.write_text("window.NB_DATA=" + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print(f"wrote {out.relative_to(ROOT)} from {run_id} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else REFERENCE_RUN)

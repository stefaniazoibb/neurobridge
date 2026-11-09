"""Run the demonstration investigation from the command line and export the report (Markdown + PDF).

    python scripts/run_demo.py [--question "..."] [--no-pdf] [--offline]
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--offline", action="store_true", help="use cached data only")
    args = ap.parse_args()
    if args.offline:
        os.environ["NEUROBRIDGE_ALLOW_NETWORK"] = "false"
    from neurobridge.agents.orchestrator import DEMO_QUESTION, run_investigation
    from neurobridge.config import get_settings
    from neurobridge.report.export import export_markdown, markdown_to_pdf

    def progress(ev):
        if ev["kind"] in ("agent_started",):
            print(f"\n== {ev['agent']}: {ev['role']}")
        elif ev["kind"] in ("tool_finished", "tool_failed"):
            print(f"   {'ok ' if ev['kind'] == 'tool_finished' else 'ERR'} {ev['tool']:32s} {(ev.get('summary') or ev.get('error') or '')[:110]}")

    from neurobridge.agents.orchestrator import Investigation

    inv = Investigation(args.question or DEMO_QUESTION, {"use_llm": True, "live_listing": not args.offline}, progress=progress)
    res = inv.run()
    out = get_settings().runs_dir / res.investigation_id / "export"
    md = export_markdown(res, inv.ws, out)
    print(f"\nStatus: {res.status}. Claims: {len(res.claims)}. Report: {md}")
    if not args.no_pdf:
        print("PDF:", markdown_to_pdf(md, out / "report.pdf"))
    for c in res.claims:
        print(f" [{c['level']:24s}] {c['kind']:14s} {c['status']:14s} {c['id']}")

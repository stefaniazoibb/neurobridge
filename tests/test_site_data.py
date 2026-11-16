"""The public website must show exactly what the reference run recorded."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = ROOT / "site" / "site_data.js"


def load_site():
    txt = SITE_DATA.read_text()
    assert txt.startswith("window.NB_DATA=") and txt.rstrip().endswith(";")
    return json.loads(txt[len("window.NB_DATA=") :].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def site():
    return load_site()


@pytest.fixture(scope="module")
def run(site):
    path = ROOT / "data" / "runs" / site["run"]["id"] / "investigation.json"
    if not path.exists():
        pytest.skip("reference run not available locally (data/ is git-ignored)")
    return json.loads(path.read_text())


def test_agents_are_the_seven_in_code(site):
    from neurobridge.agents import agents as A
    from neurobridge.agents.reviewer import ReviewerAgent

    in_code = [c.name for c in (A.HypothesisAgent, A.LiteratureAgent, A.DataAgent, A.BioinformaticsAgent, A.StatisticsAgent, A.ValidationAgent, ReviewerAgent)]
    assert [a["agent"] for a in site["agents"]] == in_code


def test_validation_numbers_match_the_run(site, run):
    arts = {v["hypothesis"]: v for k, v in run["artefacts"].items() if k.startswith("validation:")}
    for v in site["validation"].values():
        src = arts[v["hypothesis"]]
        assert v["verdict"] == src["verdict"]
        for c, s in zip(v["cohorts"], src["per_cohort"]):
            assert c["id"] == s["cohort_id"] and c["n"] == s["n"]
            assert c["est"] == pytest.approx(s["estimate"], abs=1e-4)
            assert (c["lo"], c["hi"]) == (pytest.approx(s["ci_low"], abs=1e-4), pytest.approx(s["ci_high"], abs=1e-4))


def test_negative_results_are_present(site):
    v = site["validation"]
    assert v["subtype"]["verdict"].startswith("replicated")
    assert v["subtypeAdjusted"]["verdict"] == "not replicated"
    assert v["survival"]["verdict"] == "not replicated"


def test_tool_calls_and_heatmap_match_the_run(site, run):
    assert len(site["calls"]) == len(run["log"]["tool_calls"])
    fig = json.loads((ROOT / "data" / "runs" / site["run"]["id"] / "figures" / "subtype_heatmap_neuron_glioma_extended_tcga_gbm_rnaseq.json").read_text())
    labels = [[float(x) for x in row] for row in fig["data"][0]["text"]]  # the figure's own 2-dp labels
    for row, lab in zip(site["heatmap"]["z"], labels):
        assert row == pytest.approx(lab, abs=0.0056)  # 3-dp export vs the figure's 2-dp label


def test_counts_typed_in_the_page_are_derived(site, run):
    assert site["literature"]["verified"] == len(run["artefacts"]["citations"])
    surv = [c["n"] for c in site["validation"]["survival"]["cohorts"] if c["role"] == "validation"]
    assert sum(surv) == sum(c["n"] for c in run["artefacts"]["validation:033"]["per_cohort"] if c["role"] == "validation")
    src = (ROOT / "site" / "story.src.html").read_text()
    assert "45 records" not in src and "45 verified records" not in src and "225 validation" not in src

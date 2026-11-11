import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from neurobridge.config import get_settings  # noqa: E402
from neurobridge.data.cohort import Cohort, empty_clinical  # noqa: E402
from neurobridge.schemas.datasets import AccessMode, DataModality, DatasetRecord, IngestionStatus  # noqa: E402


def cached(rel: str) -> bool:
    return (get_settings().cache_dir / rel).exists()


requires_tcga = pytest.mark.skipif(not cached("xena/TCGA.GBM.sampleMap_HiSeqV2.gz"), reason="TCGA not cached (run scripts/fetch_data.py)")
requires_cgga = pytest.mark.skipif(not cached("cgga/CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip"), reason="CGGA not cached")
requires_sc = pytest.mark.skipif(not cached("geo/GSE84465_GBM_All_data.csv.gz"), reason="GSE84465 not cached")


def make_synthetic_cohort(n=200, beta=0.5, seed=0, dataset_id="synthetic_test", subtype_shift=1.0):
    """Simulated cohort for unit tests only (clearly labelled demo data, never shown in the app).
    Exposure 'score' genes G0..G4 share a latent factor; survival is exponential with log-HR = beta per SD of the latent."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    subtype = rng.choice(["Proneural", "Mesenchymal", "Classical"], size=n)
    latent = latent + np.where(subtype == "Proneural", subtype_shift / 2, 0) - np.where(subtype == "Mesenchymal", subtype_shift / 2, 0)
    genes = {f"G{i}": 8 + latent + rng.normal(scale=0.5, size=n) for i in range(5)}
    for j in range(200):
        genes[f"BG{j}"] = 6 + rng.normal(size=n)
    ids = [f"S{i:04d}" for i in range(n)]
    expr = pd.DataFrame(genes, index=ids).T
    z = (latent - latent.mean()) / latent.std()
    t = rng.exponential(scale=np.exp(-beta * z)) * 20
    cens = rng.exponential(scale=40, size=n)
    clin = empty_clinical(ids)
    clin["os_months"] = np.minimum(t, cens)
    clin["os_event"] = (t <= cens).astype(float)
    clin["age"] = rng.normal(58, 10, size=n)
    clin["sex"] = rng.choice(["Female", "Male"], size=n)
    clin["subtype"] = subtype
    clin["sample_type"] = "Primary"
    clin["idh_status"] = "Wildtype"
    rec = DatasetRecord(dataset_id=dataset_id, name="synthetic", source="simulation", source_url="n/a", modalities=[DataModality.BULK_RNASEQ], access=AccessMode.DEMO, ingestion=IngestionStatus.CACHED, is_demo=True)
    return Cohort(dataset=rec, expression=expr, clinical=clin)


@pytest.fixture
def synth():
    return make_synthetic_cohort()


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Settings with the network disabled and runs in tmp_path, patched into every module that imported
    `get_settings` by name (patching `config.get_settings` alone does not reach them)."""
    from neurobridge import config

    s = config.Settings(allow_network=False, runs_dir=tmp_path / "runs")
    s.ensure_dirs()
    config.get_settings.cache_clear()
    monkeypatch.setattr(config, "get_settings", lambda: s)
    for mod in ["neurobridge.data.cache", "neurobridge.literature.europepmc", "neurobridge.agents.orchestrator", "neurobridge.knowledge.genesets", "neurobridge.data.catalog", "neurobridge.analysis.common", "neurobridge.data.cbioportal", "neurobridge.data.cellxgene", "neurobridge.agents.llm"]:
        m = __import__(mod, fromlist=["x"])
        if hasattr(m, "get_settings"):
            monkeypatch.setattr(m, "get_settings", lambda: s)
    return s

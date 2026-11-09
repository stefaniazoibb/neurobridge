"""Data-validation tests on the real public files (skipped if not cached)."""

import numpy as np

from neurobridge.data.cache import read_manifest
from neurobridge.data.catalog import INGESTIBLE, build_catalog
from neurobridge.provenance import sha256_file
from neurobridge.config import get_settings
from tests.conftest import requires_cgga, requires_sc, requires_tcga


def test_catalog_records_complete():
    cat = build_catalog()
    ids = [d.dataset_id for d in cat]
    assert len(ids) == len(set(ids))
    for d in cat:
        assert d.limitations, f"{d.dataset_id} has no limitations"
        assert not d.is_demo
    assert "visium_gbm_10x" in INGESTIBLE
    assert set(INGESTIBLE) <= set(ids)


def test_manifest_hashes_match_files():
    root = get_settings().cache_dir
    for key, e in read_manifest().items():
        p = root / e["path"]
        if p.exists():
            assert sha256_file(p) == e["sha256"], key


@requires_tcga
def test_tcga_cohort_integrity():
    from neurobridge.data.loaders import load_cohort

    c = load_cohort("tcga_gbm_rnaseq")
    assert c.expression.shape == (20530, 172)
    assert set(c.clinical.index) == set(c.expression.columns)
    s = c.summary()
    assert s["n_primary"] == 154
    surv = c.clinical[["os_months", "os_event"]].dropna()
    assert (surv.os_months >= 0).all() and set(surv.os_event.unique()) <= {0.0, 1.0}
    assert set(c.clinical.subtype.dropna()) == {"Proneural", "Neural", "Classical", "Mesenchymal"}
    assert 0 <= float(c.expression.values.min()) and float(c.expression.values.max()) < 25  # log2 scale
    for g in ["NLGN3", "SLC17A7", "GRIA1", "GRIA2", "GRIN2B", "BDNF"]:
        assert g in c.expression.index


@requires_cgga
def test_cgga_gbm_selection_and_harmonisation():
    from neurobridge.data.loaders import load_cohort

    for did in ("cgga_325", "cgga_693"):
        c = load_cohort(did)
        assert set(c.clinical.histology.unique()) <= {"GBM", "rGBM", "sGBM"}
        assert set(c.clinical.sex.dropna().unique()) <= {"Female", "Male"}  # whitespace stripped
        assert set(c.clinical.idh_status.dropna().unique()) <= {"Mutant", "Wildtype"}
        assert c.clinical.os_months.dropna().ge(0).all()
        assert np.isfinite(c.expression.values).all()


@requires_tcga
@requires_cgga
def test_no_patient_overlap_between_discovery_and_validation():
    from neurobridge.data.loaders import load_cohort

    t = set(load_cohort("tcga_gbm_rnaseq").clinical.patient_id.dropna())
    for did in ("cgga_325", "cgga_693"):
        assert not t & set(load_cohort(did).clinical.patient_id.dropna())
    assert not set(load_cohort("cgga_325").clinical.patient_id) & set(load_cohort("cgga_693").clinical.patient_id)


@requires_sc
def test_darmanis_annotations():
    from neurobridge.data.loaders import load_single_cell

    a = load_single_cell("gse84465_darmanis")
    assert a.n_obs == 3589 and a.obs.patient.nunique() == 4
    vc = a.obs.cell_type_author.value_counts().to_dict()
    assert vc["Neoplastic"] == 1091 and vc["Neuron"] == 21 and "Astocyte" not in vc
    assert a.obs.cell_type_author.notna().all()

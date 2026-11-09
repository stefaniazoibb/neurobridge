"""Download and cache every public resource NeuroBridge ingests (~140 MB). Safe to re-run."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from neurobridge.data.loaders import load_cohort, load_single_cell  # noqa: E402
from neurobridge.knowledge.genesets import LIBRARIES, load_library  # noqa: E402
from neurobridge.literature.europepmc import fetch_pmids  # noqa: E402
from neurobridge.literature.key_references import KEY_REFERENCES  # noqa: E402

if __name__ == "__main__":
    for d in ["tcga_gbm_rnaseq", "cgga_325", "cgga_693", "tcga_gbm_u133a"]:
        print(d, load_cohort(d).summary())
    a = load_single_cell("gse84465_darmanis")
    print("gse84465_darmanis", a.n_obs, "cells")
    for lib in LIBRARIES:
        print(lib, len(load_library(lib)), "gene sets")
    print(len(fetch_pmids([r["pmid"] for r in KEY_REFERENCES])), "key references verified")

"""NCBI GEO loaders. Currently: GSE84465 (Darmanis et al. 2017, Cell Reports), 3,589 single cells
from four GBM patients, tumour core and peripheral infiltrated tissue, Smart-seq2, with author
cell-type annotations in the series matrix."""

from __future__ import annotations

import gzip
import re

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from neurobridge.data.cache import cached_file, manifest_entry

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/"
GSE84465_COUNTS = GEO_FTP + "GSE84nnn/GSE84465/suppl/GSE84465_GBM_All_data.csv.gz"
GSE84465_MATRIX = GEO_FTP + "GSE84nnn/GSE84465/matrix/GSE84465_series_matrix.txt.gz"


def parse_series_matrix_characteristics(path) -> pd.DataFrame:
    """Parse !Sample_* rows of a GEO series matrix into a cell/sample metadata table."""
    titles, gsm, chars = None, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsm = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([t.strip('"') for t in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break
    meta = pd.DataFrame(index=gsm)
    meta["title"] = titles
    for row in chars:
        keys = {c.split(":")[0].strip() for c in row if ":" in c}
        for k in keys:
            vals = [c.split(":", 1)[1].strip() if c.startswith(k + ":") else None for c in row]
            col = re.sub(r"\W+", "_", k.lower()).strip("_")
            if col in meta.columns:
                meta[col] = [v if v is not None else old for v, old in zip(vals, meta[col])]
            else:
                meta[col] = vals
    return meta


def load_darmanis_gse84465(dataset_record) -> ad.AnnData:
    counts_path = cached_file("gse84465_counts", GSE84465_COUNTS, "geo/GSE84465_GBM_All_data.csv.gz", "NCBI GEO")
    meta_path = cached_file("gse84465_matrix", GSE84465_MATRIX, "geo/GSE84465_series_matrix.txt.gz", "NCBI GEO")
    # Read the gene x cell table in chunks straight into a sparse matrix. Reading it whole as a
    # dense int64 frame peaks near 1.5 GB, enough to crash the Streamlit Community Cloud app.
    blocks, genes, cells = [], [], None
    summary_rows = {"no_feature", "ambiguous", "too_low_aQual", "not_aligned", "alignment_not_unique"}
    for chunk in pd.read_csv(counts_path, sep=" ", index_col=0, chunksize=2000):
        # Last rows are HTSeq summary lines (no_feature, ambiguous, ...) - drop them.
        chunk = chunk[~chunk.index.str.startswith("__") & ~chunk.index.isin(summary_rows)]
        cells = chunk.columns if cells is None else cells
        blocks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.float32)))
        genes.extend(chunk.index)
    counts_index = pd.Index(genes)
    meta = parse_series_matrix_characteristics(meta_path)
    # Cell identifiers in the counts file are "<plate id>.<well>"
    meta["cell_id"] = meta["plate_id"].astype(str) + "." + meta["well"].astype(str)
    meta = meta.set_index("cell_id")
    meta = meta.reindex(cells)
    if meta["cell_type"].isna().any():
        missing = int(meta["cell_type"].isna().sum())
        raise ValueError(f"{missing} cells in the counts matrix lack annotations in the series matrix")
    X = sp.vstack(blocks).T.tocsr()
    del blocks
    obs = meta.rename(columns={"cell_type": "cell_type_author", "tissue": "location", "patient_id": "patient", "tsne_cluster": "author_cluster"})
    obs["cell_type_author"] = obs["cell_type_author"].replace({"Astocyte": "Astrocyte"})  # typo in GEO annotation
    obs["is_neoplastic"] = obs["neoplastic"].eq("Neoplastic")
    adata = ad.AnnData(X=X, obs=obs[["cell_type_author", "location", "patient", "author_cluster", "is_neoplastic", "title"]], var=pd.DataFrame(index=counts_index))
    adata.var_names_make_unique()
    adata.uns["dataset"] = dataset_record.model_dump(mode="json")
    adata.uns["provenance"] = {"counts_file": manifest_entry("gse84465_counts"), "metadata_file": manifest_entry("gse84465_matrix")}
    adata.uns["notes"] = [
        "Raw HTSeq gene counts (Smart-seq2), as distributed in GEO supplementary file GSE84465_GBM_All_data.csv.gz.",
        "Cell-type labels are the authors' annotations from the GEO series matrix; 'Astocyte' corrected to 'Astrocyte'.",
        "Four patients; tumour core vs peripheral (infiltrated) tissue; n = 3,589 cells.",
    ]
    return adata

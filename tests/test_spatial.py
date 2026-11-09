"""Spatial statistics against known answers, and integrity of the real Visium section."""

import numpy as np
import pandas as pd

from neurobridge.analysis.spatial import hex_neighbours, morans_i, ring_distance
from tests.conftest import cached
import pytest


def _hex_grid(n_rows=20, n_cols=20):
    rows = [(r, c) for r in range(n_rows) for c in range(n_cols * 2) if (r + c) % 2 == 0]
    return pd.DataFrame(rows, columns=["array_row", "array_col"])


def test_hex_neighbours_interior_has_six():
    g = _hex_grid()
    nb = hex_neighbours(g)
    i = g.index[(g.array_row == 10) & (g.array_col == 20)][0]
    assert len(nb[i]) == 6
    assert max(len(n) for n in nb) == 6 and min(len(n) for n in nb) >= 2


def test_ring_distance_bfs():
    g = _hex_grid()
    nb = hex_neighbours(g)
    src = ((g.array_row == 10) & (g.array_col == 20)).values
    d = ring_distance(nb, src)
    assert d[src][0] == 0
    assert all(d[j] == 1 for j in nb[np.where(src)[0][0]])
    assert np.isfinite(d).all()


def test_morans_i_detects_structure_and_not_noise():
    g = _hex_grid(30, 30)
    nb = hex_neighbours(g)
    rng = np.random.default_rng(0)
    smooth = (g.array_col.values > 30).astype(float) + 0.1 * rng.normal(size=len(g))
    noise = rng.normal(size=len(g))
    I_s, p_s = morans_i(smooth, nb, 199, rng)
    I_n, p_n = morans_i(noise, nb, 199, rng)
    assert I_s > 0.7 and p_s <= 0.01
    assert abs(I_n) < 0.1 and p_n > 0.01


@pytest.mark.skipif(not cached("spatial/visium_gbm_10x/filtered_feature_bc_matrix.h5"), reason="Visium section not cached")
def test_visium_section_integrity_and_tool():
    from neurobridge.agents.tools.builtin import SpatialInput, perform_spatial_analysis
    from neurobridge.agents.workspace import Workspace
    from neurobridge.data.loaders import load_spatial

    a = load_spatial()
    assert a.n_obs == 3468 and (a.obs.in_tissue == 1).all()
    assert a.obsm["spatial"].shape == (3468, 2)
    ws = Workspace("t")
    ws.single_cell["visium_gbm_10x:raw"] = a
    out = perform_spatial_analysis(ws, SpatialInput(n_perm=99))
    assert out.data["n_spots"] > 3400 and len(out.data["analysis_ids"]) == 3
    lr = pd.DataFrame(out.data["lr"])
    assert ((lr.p_domain >= 1 / 100) & (lr.p_domain <= 1)).all()
    assert {"spatial_domains", "spatial_candidate", "spatial_lr"} <= set(ws.figures)

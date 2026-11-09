"""Candidate gene panels and gene-set libraries for the neuron-glioma question.

Every panel states where it comes from. Panels assembled by NeuroBridge from the literature are labelled
as such: they are *candidate investigation targets*, not validated biomarkers. Public gene-set libraries
(KEGG, GO, Reactome, MSigDB Hallmark) are fetched from Enrichr and cached locally with a timestamp.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from neurobridge.config import get_settings
from neurobridge.logging_utils import get_logger, log_event
from neurobridge.provenance import utc_now_iso

log = get_logger(__name__)

# PMIDs verified against Europe PMC on 2026-11-09 (see neurobridge.literature.key_references).
REF = {
    "venkatesh2015": "25913192",  # Neuronal activity promotes glioma growth through neuroligin-3 secretion (Cell)
    "venkatesh2017": "28959975",  # Targeting neuronal activity-regulated NLGN3 dependency in high-grade glioma (Nature)
    "venkatesh2019": "31534222",  # Electrical and synaptic integration of glioma into neural circuits (Nature)
    "venkataramani2019": "31534219",  # Glutamatergic synaptic input to glioma cells drives brain tumour progression (Nature)
    "taylor2023": "37914930",  # Glioma synapses recruit mechanisms of adaptive plasticity (Nature)
    "krishna2023": "37138086",  # Glioblastoma remodelling of human neural circuits decreases survival (Nature)
    "hausmann2023": "36517594",  # Autonomous rhythmic activity in glioma networks drives brain tumour growth (Nature)
    "venkataramani2022": "35914528",  # Glioblastoma hijacks neuronal mechanisms for brain invasion (Cell)
    "osswald2015": "26536111",  # Brain tumour cells interconnect to a functional and resistant network (Nature)
    "verhaak2010": "20129251",
    "wang2017": "28697342",
    "neftel2019": "31327527",
    "darmanis2017": "29091775",
    "brennan2013": "24120142",
    "becht2016": "27765066",  # MCP-counter
}


@dataclass
class GenePanel:
    panel_id: str
    name: str
    genes: list[str]
    origin: str  # "user-specified", "literature-informed (NeuroBridge)", "public library"
    description: str
    gene_rationale: dict[str, str] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    role: str = "hypothesis"  # hypothesis | control | background


CANDIDATE_PANEL = GenePanel(
    panel_id="neuronal_activity_candidates",
    name="Neuronal-activity candidate genes (pre-specified)",
    genes=["NLGN3", "SLC17A7", "GRIA1", "GRIA2", "GRIN2B", "BDNF"],
    origin="user-specified",
    description="The six genes named in the research brief. They are investigation targets, not assumed biomarkers.",
    gene_rationale={
        "NLGN3": "Activity-regulated neuroligin-3 is shed by neurons/OPCs and promotes glioma proliferation; its absence stalls glioma growth in mice.",
        "SLC17A7": "VGLUT1, the vesicular glutamate transporter of cortical glutamatergic neurons; a marker of presynaptic glutamatergic input (and of normal neurons in bulk tissue).",
        "GRIA1": "AMPA receptor subunit GluA1; glioma cells receive AMPA-receptor-mediated synaptic input.",
        "GRIA2": "AMPA receptor subunit GluA2; calcium-permeable (GluA2-lacking or unedited) AMPA receptors on glioma cells drive progression.",
        "GRIN2B": "NMDA receptor subunit GluN2B; NMDA signalling is implicated in neuron-tumour interactions and brain metastasis.",
        "BDNF": "Activity-dependent neurotrophin; BDNF-TrkB signalling strengthens glioma synapses.",
    },
    citations=[REF["venkatesh2015"], REF["venkatesh2017"], REF["venkatesh2019"], REF["venkataramani2019"], REF["taylor2023"]],
)

EXTENDED_PANEL = GenePanel(
    panel_id="neuron_glioma_extended",
    name="Neuron-glioma signalling panel (literature-informed, assembled by NeuroBridge)",
    genes=["NLGN3", "BDNF", "NTRK2", "GRIA1", "GRIA2", "GRIA3", "GRIA4", "GRIN1", "GRIN2B", "SLC17A7", "THBS1", "GAP43", "GJA1"],
    origin="literature-informed (NeuroBridge)",
    description=(
        "Genes repeatedly implicated in neuron-glioma interaction papers: NLGN3 shedding, BDNF-TrkB (NTRK2) synaptic plasticity, "
        "AMPA/NMDA receptor subunits mediating neuron-to-glioma synapses, thrombospondin-1 (THBS1) secreted by glioma cells to remodel circuits, "
        "and tumour-microtube genes GAP43 and GJA1 (connexin 43). This panel was assembled by NeuroBridge, not taken from a published signature."
    ),
    gene_rationale={
        "NTRK2": "TrkB, receptor for BDNF; BDNF-TrkB signalling increases glioma synaptic strength.",
        "GRIA3": "AMPA receptor subunit GluA3.",
        "GRIA4": "AMPA receptor subunit GluA4.",
        "GRIN1": "Obligatory NMDA receptor subunit GluN1.",
        "THBS1": "Thrombospondin-1, a synaptogenic factor secreted by glioblastoma cells in functionally connected regions.",
        "GAP43": "Growth-associated protein 43; required for tumour-microtube formation and network connectivity.",
        "GJA1": "Connexin 43; gap-junction coupling of glioma cell networks.",
    },
    citations=[REF["venkatesh2015"], REF["taylor2023"], REF["venkataramani2019"], REF["krishna2023"], REF["osswald2015"], REF["hausmann2023"]],
)

NEURON_CONTROL_PANEL = GenePanel(
    panel_id="normal_neuron_markers",
    name="Mature-neuron markers (contamination control)",
    genes=["SNAP25", "SYT1", "RBFOX3", "GABRA1", "NEFL", "STMN2", "SYN1", "GAD1"],
    origin="literature-informed (NeuroBridge)",
    description=(
        "Canonical pan-neuronal / mature neuron markers used as a negative control: in bulk tumour tissue, a 'neuronal activity' score that tracks "
        "these markers may simply measure the amount of infiltrated normal brain (tumour purity) rather than tumour-intrinsic biology."
    ),
    citations=[REF["wang2017"]],
    role="control",
)

PANELS: dict[str, GenePanel] = {p.panel_id: p for p in (CANDIDATE_PANEL, EXTENDED_PANEL, NEURON_CONTROL_PANEL)}

# Public gene-set libraries available through Enrichr (fetched live, cached as JSON with timestamp).
LIBRARIES = {
    "MSigDB_Hallmark_2020": "MSigDB Hallmark (50 sets)",
    "KEGG_2021_Human": "KEGG pathways (human, 2021)",
    "GO_Biological_Process_2023": "Gene Ontology Biological Process (2023)",
    "Reactome_2022": "Reactome pathways (2022)",
}

# Named gene sets pulled from those libraries for targeted (pre-specified) scoring.
PATHWAY_SETS = {
    "kegg_glutamatergic_synapse": ("KEGG_2021_Human", "Glutamatergic synapse"),
    "go_chemical_synaptic_transmission": ("GO_Biological_Process_2023", "Chemical Synaptic Transmission (GO:0007268)"),
    "go_regulation_of_synaptic_plasticity": ("GO_Biological_Process_2023", "Regulation Of Neuronal Synaptic Plasticity (GO:0048168)"),
}


def _lib_cache_path(library: str) -> Path:
    d = get_settings().cache_dir / "genesets"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{library}.json"


def load_library(library: str, *, allow_download: bool | None = None) -> dict[str, list[str]]:
    """Return {term: [genes]} for an Enrichr library, caching to disk."""
    path = _lib_cache_path(library)
    if path.exists():
        payload = json.loads(path.read_text())
        return payload["sets"]
    s = get_settings()
    if allow_download is False or not s.allow_network:
        raise FileNotFoundError(f"Gene-set library {library} is not cached and network access is disabled")
    import gseapy as gp

    sets = gp.get_library(library)
    path.write_text(json.dumps({"library": library, "retrieved_at": utc_now_iso(), "source": "Enrichr (maayanlab.cloud)", "n_sets": len(sets), "sets": sets}))
    log_event(log, "genesets.cached", library=library, n_sets=len(sets))
    return sets


def library_metadata(library: str) -> dict | None:
    path = _lib_cache_path(library)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return {k: v for k, v in payload.items() if k != "sets"}


def get_pathway_set(set_id: str) -> GenePanel:
    library, term = PATHWAY_SETS[set_id]
    sets = load_library(library)
    if term not in sets:
        raise KeyError(f"{term!r} not found in {library}")
    meta = library_metadata(library) or {}
    return GenePanel(
        panel_id=set_id,
        name=f"{term} [{library}]",
        genes=sorted(set(sets[term])),
        origin="public library",
        description=f"Gene set '{term}' from Enrichr library {library}, retrieved {meta.get('retrieved_at', 'n/a')}.",
        role="hypothesis",
    )


def all_panels(include_pathways: bool = True) -> dict[str, GenePanel]:
    out = dict(PANELS)
    if include_pathways:
        for sid in PATHWAY_SETS:
            try:
                out[sid] = get_pathway_set(sid)
            except Exception as exc:  # noqa: BLE001
                log_event(log, "genesets.pathway_unavailable", set_id=sid, error=str(exc))
    return out


def register_custom_panel(genes: list[str], panel_id: str = "user_genes", name: str = "User-specified genes") -> GenePanel:
    """Add a panel built from genes named in the research question (validated against the expression matrix later)."""
    p = GenePanel(panel_id=panel_id, name=name, genes=list(dict.fromkeys(g.upper() for g in genes)), origin="user-specified", description="Genes extracted from the research question.")
    PANELS[panel_id] = p
    return p

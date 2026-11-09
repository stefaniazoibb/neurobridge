"""Curated key references for the neuron-glioma question. Only PMIDs are stored here; titles, authors,
journals and years are always retrieved from Europe PMC at run time (cached), so nothing is typed by hand.
Each PMID was checked against Europe PMC on 2026-11-09."""

from __future__ import annotations

KEY_REFERENCES: list[dict] = [
    {"pmid": "25913192", "topic": "neuron-glioma signalling", "relevance": "Neuronal activity promotes high-grade glioma proliferation via activity-regulated secretion of neuroligin-3 (NLGN3)."},
    {"pmid": "28959975", "topic": "neuron-glioma signalling", "relevance": "Glioma growth depends on NLGN3; ADAM10 inhibition blocks NLGN3 release in xenograft models."},
    {"pmid": "31534222", "topic": "neuron-glioma synapses", "relevance": "AMPA-receptor-mediated neuron-to-glioma synapses and activity-induced potassium currents promote glioma progression."},
    {"pmid": "31534219", "topic": "neuron-glioma synapses", "relevance": "Glutamatergic synaptic input to glioma cells via calcium-permeable AMPA receptors drives tumour invasion and growth."},
    {"pmid": "37914930", "topic": "neuron-glioma synapses", "relevance": "BDNF-TrkB signalling increases glioma synaptic strength (adaptive plasticity)."},
    {"pmid": "37138086", "topic": "neural circuit remodelling and outcome", "relevance": "Functional connectivity of glioblastoma with neural circuits (via TSP-1) associates with shorter survival and cognitive decline in patients."},
    {"pmid": "26536111", "topic": "tumour cell networks", "relevance": "Tumour microtubes (GAP43-dependent) connect astrocytoma cells into resistant networks."},
    {"pmid": "36517594", "topic": "tumour cell networks", "relevance": "Autonomous rhythmic calcium activity in glioma networks drives tumour growth."},
    {"pmid": "35914528", "topic": "neuron-glioma invasion", "relevance": "Glioblastoma cells hijack neuronal mechanisms (neurite-like tumour microtubes) for brain invasion."},
    {"pmid": "20129251", "topic": "molecular subtypes", "relevance": "Defines Proneural, Neural, Classical and Mesenchymal transcriptional subtypes in TCGA."},
    {"pmid": "28697342", "topic": "molecular subtypes", "relevance": "Revises subtypes using tumour-intrinsic genes; the Neural subtype is attributed to non-tumour (normal neural) contamination."},
    {"pmid": "24120142", "topic": "TCGA-GBM", "relevance": "Somatic genomic landscape of glioblastoma; TCGA RNA-seq cohort description."},
    {"pmid": "31327527", "topic": "single-cell states", "relevance": "Single-cell model of glioblastoma cellular states (NPC-, OPC-, AC-, MES-like) and their plasticity."},
    {"pmid": "29091775", "topic": "single-cell data", "relevance": "GSE84465: single-cell RNA-seq of tumour core and infiltrated periphery from four GBM patients."},
    {"pmid": "33662628", "topic": "CGGA", "relevance": "Chinese Glioma Genome Atlas resource description (validation cohorts)."},
    {"pmid": "27765066", "topic": "bulk deconvolution", "relevance": "MCP-counter: estimating cell population abundance from bulk expression (composition confounding)."},
]

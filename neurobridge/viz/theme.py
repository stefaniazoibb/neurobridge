"""Visual tokens (validated categorical palette, sequential/diverging ramps) and a Plotly template."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ee8a89"], [1.0, "#b52f2f"]]
INK = "#1d1d1b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"

# Fixed colour per entity (colour follows the entity, never its rank).
SUBTYPE_COLORS = {"Proneural": CATEGORICAL[0], "Classical": CATEGORICAL[1], "Mesenchymal": CATEGORICAL[2], "Neural": CATEGORICAL[6], "G-CIMP": CATEGORICAL[3]}
GROUP_COLORS = {"high": CATEGORICAL[1], "low": CATEGORICAL[0], "mid": "#9a9890"}
COHORT_COLORS = {"tcga_gbm_rnaseq": CATEGORICAL[0], "cgga_325": CATEGORICAL[1], "cgga_693": CATEGORICAL[2], "tcga_gbm_u133a": CATEGORICAL[6]}
CELLTYPE_COLORS = {"Neoplastic": CATEGORICAL[1], "Immune cell": CATEGORICAL[0], "OPC": CATEGORICAL[2], "Astrocyte": CATEGORICAL[3], "Oligodendrocyte": CATEGORICAL[4], "Neuron": CATEGORICAL[6], "Vascular": CATEGORICAL[7]}


def color_for(label: str, mapping: dict[str, str], fallback_index: int = 0) -> str:
    return mapping.get(label, CATEGORICAL[fallback_index % len(CATEGORICAL)])


TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CATEGORICAL,
        margin=dict(l=60, r=24, t=96, b=56),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID, title=dict(font=dict(color=INK_2))),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID, title=dict(font=dict(color=INK_2))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=12), orientation="h", yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.985, yanchor="top"),
    )
)
pio.templates["neurobridge"] = TEMPLATE


def apply(fig: go.Figure, title: str, caption: str, height: int = 420) -> go.Figure:
    fig.update_layout(template="neurobridge", title=title, height=height, meta={"caption": caption})
    return fig


def for_dark_mode(fig: go.Figure) -> go.Figure:
    """Restyle text and grid tokens for a dark surface (data colours keep their entity mapping)."""
    ink, ink2, grid = "#f0efec", "#c3c2b7", "#3a3a37"
    fig.update_layout(font_color=ink, title_font_color=ink, legend_font_color=ink2)
    fig.update_xaxes(gridcolor=grid, linecolor=grid, tickcolor=grid, title_font_color=ink2)
    fig.update_yaxes(gridcolor=grid, linecolor=grid, tickcolor=grid, title_font_color=ink2)
    for a in fig.layout.annotations or []:
        a.font.color = ink
    return fig

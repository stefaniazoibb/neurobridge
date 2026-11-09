"""Visual identity shared with the NeuroBridge website: midnight ground, ivory ink, semantic accents.

Colour meanings (same as the site): teal = neuron / supported, violet = tumour / not supported,
cyan = data, warm = signal / caution, ivory = text and neutral marks.
Figures are drawn for the dark app; `for_print` restyles a copy for the white PDF report.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

BG = "#04060c"
SURFACE = "#0d1322"
INK = "#ece7dc"
INK_2 = "#bcb8ae"
INK_3 = "#868791"
GRID = "rgba(236,231,220,0.10)"
LINE = "rgba(236,231,220,0.22)"
TEAL, VIOLET, CYAN, WARM, ROSE, BLUE, MOSS = "#43c9b4", "#9d8cf2", "#8fd3ee", "#e6a35c", "#e07a8f", "#5c84e0", "#a3c46a"
DIM = "#3b4356"
FONT = "Hanken Grotesk, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
SERIF = "Newsreader, Georgia, serif"
MONO = "Geist Mono, ui-monospace, Menlo, monospace"

CATEGORICAL = [TEAL, VIOLET, CYAN, WARM, ROSE, BLUE, MOSS, INK]
SEQUENTIAL_BLUE = ["#0f1a26", "#123236", "#164c4c", "#1c6964", "#25877c", "#33a898", "#43c9b4"]  # dark -> teal (name kept for callers)
SEQUENTIAL = SEQUENTIAL_BLUE
DIVERGING = [[0.0, VIOLET], [0.25, "#5a5190"], [0.5, "#232838"], [0.75, "#25796d"], [1.0, TEAL]]

SUBTYPE_COLORS = {"Proneural": TEAL, "Classical": CYAN, "Mesenchymal": VIOLET, "Neural": WARM, "G-CIMP": INK}
GROUP_COLORS = {"high": VIOLET, "low": TEAL, "mid": INK_3}
COHORT_COLORS = {"tcga_gbm_rnaseq": CYAN, "cgga_325": VIOLET, "cgga_693": TEAL, "tcga_gbm_u133a": WARM}
CELLTYPE_COLORS = {"Neoplastic": VIOLET, "Immune cell": CYAN, "OPC": TEAL, "Astrocyte": WARM, "Oligodendrocyte": ROSE, "Neuron": INK, "Vascular": BLUE}


def color_for(label: str, mapping: dict[str, str], fallback_index: int = 0) -> str:
    return mapping.get(label, CATEGORICAL[fallback_index % len(CATEGORICAL)])


TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=13, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CATEGORICAL,
        margin=dict(l=60, r=24, t=96, b=56),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE, tickfont=dict(family=MONO, size=11, color=INK_3), title=dict(font=dict(color=INK_2))),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE, tickfont=dict(family=MONO, size=11, color=INK_3), title=dict(font=dict(color=INK_2))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=12), orientation="h", yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(font=dict(family=FONT, size=12, color=INK), bgcolor=SURFACE, bordercolor=LINE),
        title=dict(font=dict(family=SERIF, size=20, color=INK), x=0, xanchor="left", y=0.985, yanchor="top"),
        coloraxis=dict(colorbar=dict(tickfont=dict(family=MONO, color=INK_3), outlinewidth=0)),
    )
)
pio.templates["neurobridge"] = TEMPLATE


def apply(fig: go.Figure, title: str, caption: str, height: int = 420) -> go.Figure:
    fig.update_layout(template="neurobridge", title=title, height=height, meta={"caption": caption})
    return fig


def for_print(fig: go.Figure) -> go.Figure:
    """Copy restyled for white paper (PDF/Markdown export): dark ink, light grid, white ground."""
    f = go.Figure(fig)
    ink, ink2, grid = "#1d1d1b", "#52514e", "#e6e5e0"
    f.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color=ink2, title_font_color=ink, legend_font_color=ink2)
    f.update_xaxes(gridcolor=grid, linecolor=grid, tickcolor=grid, tickfont_color=ink2, title_font_color=ink2)
    f.update_yaxes(gridcolor=grid, linecolor=grid, tickcolor=grid, tickfont_color=ink2, title_font_color=ink2)
    for a in f.layout.annotations or []:
        a.font.color = ink
    for t in f.data:
        if getattr(t, "type", "") == "scatter" and getattr(t, "marker", None) is not None and t.marker.color == INK:
            t.marker.color = ink
        if getattr(t, "line", None) is not None and getattr(t.line, "color", None) == INK:
            t.line.color = ink
    return f

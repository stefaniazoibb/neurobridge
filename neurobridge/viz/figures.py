"""Figure builders. Inputs are typed analysis results; outputs are Plotly figures with a methods caption."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neurobridge.schemas.analysis import CorrelationResult, CoxResult, DEResult, EnrichmentResult, KMResult, ValidationSummary
from neurobridge.viz.theme import CELLTYPE_COLORS, COHORT_COLORS, DIVERGING, GROUP_COLORS, INK_2, SEQUENTIAL_BLUE, SUBTYPE_COLORS, apply, color_for


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def kaplan_meier_figure(km: KMResult, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, g in enumerate(km.groups):
        col = GROUP_COLORS.get(g.label) or SUBTYPE_COLORS.get(g.label) or color_for(g.label, {}, i)
        x = [0.0] + g.timeline
        y = [1.0] + g.survival
        lo = [1.0] + g.ci_low
        hi = [1.0] + g.ci_high
        fig.add_trace(go.Scatter(x=x + x[::-1], y=hi + lo[::-1], fill="toself", fillcolor=col, opacity=0.12, line=dict(width=0), hoverinfo="skip", showlegend=False, line_shape="hv"))
        med = f"{g.median_survival:.1f} mo" if g.median_survival is not None else "not reached"
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=2, shape="hv"), name=f"{g.label} (n={g.n}, events={g.events}, median {med})", hovertemplate=f"{g.label}<br>t=%{{x:.1f}} mo<br>S(t)=%{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="Overall survival (months)", rangemode="tozero")
    fig.update_yaxes(title="Survival probability", range=[0, 1.02])
    cap = (f"Kaplan-Meier estimates with Greenwood 95% CI (shaded). Stratifier: {km.stratifier}. "
           f"Log-rank p = {_fmt_p(km.logrank_p)}; n = {km.n_total}, events = {km.n_events}. "
           f"Dataset: {', '.join(km.record.dataset_ids)}. Unadjusted; {km.record.kind.value} analysis.")
    return apply(fig, title or f"Overall survival by {km.stratifier}", cap)


def forest_figure(rows: list[dict], title: str, caption: str, ref: float = 1.0, xlabel: str = "Hazard ratio (95% CI, log scale)", log_x: bool = True) -> go.Figure:
    """rows: label, est, lo, hi, p, color(optional)."""
    rows = list(rows)[::-1]
    fig = go.Figure()
    for i, r in enumerate(rows):
        col = r.get("color", "#2a78d6")
        fig.add_trace(go.Scatter(x=[r["lo"], r["hi"]], y=[r["label"]] * 2, mode="lines", line=dict(color=col, width=2), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[r["est"]], y=[r["label"]], mode="markers", marker=dict(color=col, size=10, symbol=r.get("symbol", "square"), line=dict(color="white", width=2)), showlegend=False,
                                 hovertemplate=f"{r['label']}<br>estimate {r['est']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})<br>p = {_fmt_p(r.get('p'))}<extra></extra>"))
    fig.add_vline(x=ref, line=dict(color=INK_2, width=1, dash="dot"))
    los = [r["lo"] for r in rows if r.get("lo") is not None and np.isfinite(r["lo"])]
    his = [r["hi"] for r in rows if r.get("hi") is not None and np.isfinite(r["hi"])]
    if log_x and los and his:
        lo, hi = min(min(los), ref) * 0.85, max(max(his), ref) * 1.15
        ticks = [t for t in [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 10] if lo <= t <= hi]
        fig.update_xaxes(type="log", title=xlabel, range=[np.log10(lo), np.log10(hi)], tickvals=ticks, ticktext=[f"{t:g}" for t in ticks], minor=dict(showgrid=False))
    else:
        pad = 0.1 * (max(his + [ref]) - min(los + [ref])) if los and his else 0.1
        fig.update_xaxes(type="linear", title=xlabel, range=[min(los + [ref]) - pad, max(his + [ref]) + pad] if los and his else None)
    fig.update_yaxes(automargin=True)
    return apply(fig, title, caption, height=max(260, 70 + 44 * len(rows)))


def cox_forest_figure(cx: CoxResult, title: str | None = None) -> go.Figure:
    rows = [{"label": r.covariate, "est": r.hr, "lo": r.hr_ci_low, "hi": r.hr_ci_high, "p": r.p} for r in cx.rows]
    ph = "PH assumption: no violation detected" if cx.ph_assumption_ok else ("PH assumption questionable for at least one covariate" if cx.ph_assumption_ok is False else "PH test unavailable")
    cap = (f"Cox proportional hazards ({cx.model}); n = {cx.n}, events = {cx.events}, concordance = {cx.concordance:.3f}. {ph} (scaled Schoenfeld residuals, rank time transform). "
           f"Continuous exposure standardised (HR per 1 SD). Dataset: {', '.join(cx.record.dataset_ids)}.")
    return forest_figure(rows, title or f"Cox model ({cx.model})", cap)


def validation_forest_figure(vs: ValidationSummary, title: str | None = None) -> go.Figure:
    is_hr = vs.per_cohort and vs.per_cohort[0].estimate_name.startswith("HR")
    rows = []
    for r in vs.per_cohort:
        if r.estimate is None or np.isnan(r.estimate):
            continue
        rows.append({"label": f"{r.cohort_id} [{r.role}] n={r.n}", "est": r.estimate, "lo": r.ci_low, "hi": r.ci_high, "p": r.p, "color": COHORT_COLORS.get(r.cohort_id, "#2a78d6")})
    if vs.pooled_estimate is not None:
        rows.append({"label": "Random-effects pooled (descriptive)", "est": vs.pooled_estimate, "lo": vs.pooled_ci_low, "hi": vs.pooled_ci_high, "p": vs.pooled_p, "color": "#1d1d1b", "symbol": "diamond"})
    i2 = f" I^2 = {vs.heterogeneity_i2:.0f}%." if vs.heterogeneity_i2 is not None else ""
    if is_hr:
        cap = f"Adjusted Cox HR per 1 SD of the score, estimated separately in each cohort (primary tumours). Verdict: {vs.verdict}.{i2} Replication rule: same direction as discovery and p < 0.05 in an independent cohort."
        return forest_figure(rows, title or "Cross-cohort validation: score and overall survival", cap)
    cap = f"Hedges' g (95% CI) for: {vs.hypothesis}. Verdict: {vs.verdict}.{i2} Replication rule: same sign as discovery and Mann-Whitney p < 0.05 in an independent cohort."
    return forest_figure(rows, title or "Cross-cohort validation: group difference", cap, ref=0.0, xlabel="Hedges' g (95% CI)", log_x=False)


def volcano_figure(de: DEResult, highlight: list[str] | None = None, q_thr: float = 0.05, fc_thr: float = 0.5, label_top: int = 12) -> go.Figure:
    df = de.to_frame()
    df["nlp"] = -np.log10(df["p_value"].clip(lower=1e-300))
    sig_up = (df.q_value < q_thr) & (df.log2_fc >= fc_thr)
    sig_dn = (df.q_value < q_thr) & (df.log2_fc <= -fc_thr)
    fig = go.Figure()
    for mask, name, col in [(~(sig_up | sig_dn), "not significant", "#c3c2b7"), (sig_dn, f"higher in {de.group_b}", "#2a78d6"), (sig_up, f"higher in {de.group_a}", "#eb6834")]:
        d = df[mask]
        fig.add_trace(go.Scattergl(x=d.log2_fc, y=d.nlp, mode="markers", name=f"{name} ({mask.sum()})", marker=dict(size=5, color=col, opacity=0.7), text=d.gene,
                                   customdata=np.stack([d.q_value, d.hedges_g], axis=1), hovertemplate="%{text}<br>log2FC %{x:.2f}<br>q %{customdata[0]:.2e}<br>Hedges g %{customdata[1]:.2f}<extra></extra>"))
    lab = df[sig_up | sig_dn].nsmallest(label_top, "p_value")
    if highlight:
        lab = pd.concat([lab, df[df.gene.isin(highlight)]]).drop_duplicates("gene")
    for _, r in lab.iterrows():
        fig.add_annotation(x=r.log2_fc, y=r.nlp, text=r.gene, showarrow=False, yshift=9, font=dict(size=10, color="#1d1d1b"))
    fig.update_xaxes(title=f"log2 fold change ({de.group_a} - {de.group_b})")
    fig.update_yaxes(title="-log10 p")
    cap = (f"{de.rows[0].test if de.rows else ''} test per gene on log2 expression; {de.n_tested} expressed genes tested; {de.n_significant} with BH q < {de.alpha}. "
           f"Thresholds for colour: q < {q_thr} and |log2FC| >= {fc_thr}. n({de.group_a}) = {de.n_a}, n({de.group_b}) = {de.n_b}. Dataset: {', '.join(de.record.dataset_ids)}. Exploratory.")
    return apply(fig, f"Differential expression: {de.group_a} vs {de.group_b}", cap, height=480)


def group_box_figure(values: pd.Series, groups: pd.Series, ylabel: str, title: str, caption: str, color_map: dict | None = None, order: list[str] | None = None) -> go.Figure:
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    order = order or sorted(df.g.unique(), key=lambda x: df[df.g == x].v.median())
    fig = go.Figure()
    for i, g in enumerate(order):
        d = df[df.g == g]
        col = (color_map or {}).get(g) or color_for(g, SUBTYPE_COLORS, i)
        fig.add_trace(go.Box(y=d.v, name=f"{g}<br>n={len(d)}", marker=dict(color=col, size=5, opacity=0.6), line=dict(color=col, width=2), boxpoints="all", jitter=0.4, pointpos=0, fillcolor="rgba(0,0,0,0)", showlegend=False,
                             hovertemplate=f"{g}<br>%{{y:.2f}}<extra></extra>"))
    fig.update_yaxes(title=ylabel)
    return apply(fig, title, caption)


def heatmap_figure(matrix: pd.DataFrame, title: str, caption: str, zmid: float | None = 0.0, colorscale=None, zlabel: str = "", height: int | None = None, text_fmt: str | None = None) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix.values, x=list(matrix.columns), y=list(matrix.index), colorscale=colorscale or DIVERGING, zmid=zmid, colorbar=dict(title=zlabel, thickness=12),
                               text=np.vectorize(lambda v: format(v, text_fmt))(matrix.values) if text_fmt else None, texttemplate="%{text}" if text_fmt else None,
                               hovertemplate="%{y} / %{x}<br>%{z:.3f}<extra></extra>", xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return apply(fig, title, caption, height=height or max(360, 40 + 30 * len(matrix)))


def correlation_figure(cr: CorrelationResult) -> go.Figure:
    m = pd.DataFrame(cr.matrix, index=cr.genes, columns=cr.genes)
    cap = f"{cr.method.title()} correlation of log2 expression across n = {cr.n} samples (dataset {', '.join(cr.record.dataset_ids)}); hover for coefficients. BH-adjusted p-values in the table view. Co-expression in bulk tissue largely reflects shared cell-type composition."
    fig = heatmap_figure(m, "Candidate-gene co-expression", cap, zmid=0, zlabel="rho", text_fmt=".2f")
    fig.data[0].update(zmin=-1, zmax=1)
    return fig


def enrichment_figure(enr: EnrichmentResult, top: int = 15) -> go.Figure:
    terms = enr.terms[:top][::-1]
    if not terms:
        fig = go.Figure()
        return apply(fig, "No enrichment terms", "No terms returned.")
    if "nes" in terms[0]:
        x = [t["nes"] for t in terms]
        col = ["#eb6834" if v > 0 else "#2a78d6" for v in x]
        xl = "Normalised enrichment score"
        hov = [f"{t['term']}<br>NES {t['nes']:.2f}<br>FDR q {t['q']:.3g}" for t in terms]
    else:
        x = [-np.log10(max(t["q"], 1e-300)) for t in terms]
        col = ["#2a78d6"] * len(x)
        xl = "-log10 FDR q"
        hov = [f"{t['term']}<br>overlap {t['overlap']}<br>odds ratio {t['odds_ratio']:.1f}<br>q {t['q']:.3g}" for t in terms]
    fig = go.Figure(go.Bar(x=x, y=[t["term"][:70] for t in terms], orientation="h", marker=dict(color=col, cornerradius=4), hovertext=hov, hoverinfo="text"))
    fig.update_xaxes(title=xl)
    fig.update_yaxes(automargin=True, showgrid=False)
    cap = f"{enr.method} against {enr.library}; {enr.record.method}. Input: {enr.n_input_genes} genes{(' ; background ' + enr.background) if enr.background else ''}. Exploratory."
    return apply(fig, f"Pathway enrichment ({enr.library})", cap, height=max(360, 60 + 26 * len(terms)))


def umap_figure(obs: pd.DataFrame, umap: np.ndarray, color_key: str, title: str, caption: str, continuous: bool = False) -> go.Figure:
    fig = go.Figure()
    if continuous:
        v = obs[color_key].values
        fig.add_trace(go.Scattergl(x=umap[:, 0], y=umap[:, 1], mode="markers", marker=dict(size=4, color=v, colorscale=[[i / 6, c] for i, c in enumerate(SEQUENTIAL_BLUE)], colorbar=dict(title=color_key, thickness=12), opacity=0.85),
                                   text=obs.get("cell_type_author"), hovertemplate="%{text}<br>" + color_key + " %{marker.color:.2f}<extra></extra>", showlegend=False))
    else:
        cats = obs[color_key].astype(str)
        order = cats.value_counts().index.tolist()
        for i, c in enumerate(order):
            m = (cats == c).values
            fig.add_trace(go.Scattergl(x=umap[m, 0], y=umap[m, 1], mode="markers", name=f"{c} ({m.sum()})", marker=dict(size=4, color=color_for(c, CELLTYPE_COLORS, i), opacity=0.8), hovertemplate=f"{c}<extra></extra>"))
    fig.update_xaxes(title="UMAP 1", showticklabels=False, showgrid=False)
    fig.update_yaxes(title="UMAP 2", showticklabels=False, showgrid=False)
    return apply(fig, title, caption, height=500)


def dotplot_figure(table: pd.DataFrame, title: str, caption: str) -> go.Figure:
    """table: output of singlecell.cell_type_expression (MultiIndex columns)."""
    mean = table["mean_log_expr"]
    frac = table["fraction_detected"]
    n = table[("n_cells", "")] if ("n_cells", "") in table.columns else None
    genes = list(mean.columns)
    cts = list(mean.index)
    xs, ys, sizes, colors, hov = [], [], [], [], []
    for ct in cts:
        for g in genes:
            xs.append(g); ys.append(f"{ct} (n={int(n[ct])})" if n is not None else ct)
            sizes.append(4 + 26 * float(frac.loc[ct, g])); colors.append(float(mean.loc[ct, g]))
            hov.append(f"{ct} / {g}<br>detected in {100 * frac.loc[ct, g]:.0f}% of cells<br>mean log1p(CP10k) {mean.loc[ct, g]:.2f}")
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=sizes, color=colors, colorscale=[[i / 6, c] for i, c in enumerate(SEQUENTIAL_BLUE)], colorbar=dict(title="mean expr", thickness=12), line=dict(color="white", width=1)), hovertext=hov, hoverinfo="text"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, automargin=True)
    return apply(fig, title, caption, height=max(340, 60 + 42 * len(cts)))


def subtype_gene_heatmap(cohort, genes: list[str], group_col: str = "subtype") -> go.Figure:
    expr = cohort.expression.loc[[g for g in genes if g in cohort.expression.index]]
    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1, ddof=1), axis=0)
    grp = cohort.clinical[group_col]
    m = z.T.groupby(grp).mean().T
    counts = grp.value_counts()
    m.columns = [f"{c} (n={counts[c]})" for c in m.columns]
    cap = f"Mean per-gene z-score (log2 expression standardised across all samples) by {group_col}; dataset {cohort.dataset.dataset_id}, n = {int(grp.notna().sum())} samples with labels."
    fig = heatmap_figure(m, f"Candidate-gene expression by {group_col}", cap, zmid=0, zlabel="mean z", text_fmt=".2f")
    fig.data[0].update(zmin=-1.2, zmax=1.2)
    return fig


def scatter_figure(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, caption: str, color: pd.Series | None = None) -> go.Figure:
    df = pd.DataFrame({"x": x, "y": y, "c": color if color is not None else "all"}).dropna(subset=["x", "y"])
    fig = go.Figure()
    for i, c in enumerate(sorted(df.c.astype(str).unique())):
        d = df[df.c.astype(str) == c]
        fig.add_trace(go.Scatter(x=d.x, y=d.y, mode="markers", name=f"{c} ({len(d)})", marker=dict(size=8, color=color_for(c, SUBTYPE_COLORS, i), opacity=0.75, line=dict(color="white", width=1)), text=d.index, hovertemplate="%{text}<br>x %{x:.2f}<br>y %{y:.2f}<extra></extra>", showlegend=color is not None))
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return apply(fig, title, caption)


PROGRAMME_COLORS = {"neuron-enriched": "#6250d6", "tumour_glial-enriched": "#eb6834", "myeloid-enriched": "#2a78d6", "vascular-enriched": "#1baf7a"}


def spatial_figure(a, values, title: str, caption: str, continuous: bool = True, colorscale=None, image=None, highlight=None, cmid=None) -> go.Figure:
    """Spots on the H&E low-resolution image. values: array (continuous) or categorical labels."""
    sf = a.uns["spatial_meta"]["scalefactors"]["tissue_lowres_scalef"]
    xy = a.obsm["spatial"] * sf
    fig = go.Figure()
    if image is not None:
        grey = (np.asarray(image, float) @ [0.299, 0.587, 0.114]).clip(0, 255)
        fig.add_trace(go.Image(z=np.stack([grey] * 3, axis=-1).astype(np.uint8), opacity=0.35, hoverinfo="skip"))
    # marker size in screen pixels so that spots roughly tile the hexagonal grid in a ~560 px tall plot
    n_rows = int(a.obs["array_row"].max() - a.obs["array_row"].min() + 1) if "array_row" in a.obs else 78
    size = max(3.0, 0.92 * 470 / n_rows * 1.15)
    if continuous:
        v = np.asarray(values, float)
        fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers", showlegend=False,
                                   marker=dict(size=size, color=v, colorscale=colorscale or [[i / 6, c] for i, c in enumerate(SEQUENTIAL_BLUE)], cmid=cmid, colorbar=dict(thickness=12), opacity=0.9),
                                   hovertemplate="%{marker.color:.2f}<extra></extra>"))
    else:
        lab = pd.Series(np.asarray(values).astype(str))
        cats = sorted(lab.unique(), key=lambda c: list(PROGRAMME_COLORS).index(c) if c in PROGRAMME_COLORS else 99)
        for i, c in enumerate(cats):
            m = (lab == c).values
            fig.add_trace(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"{c} ({m.sum()} spots)", marker=dict(size=size, color=PROGRAMME_COLORS.get(c, color_for(c, {}, i)), opacity=0.9), hovertemplate=f"{c}<extra></extra>"))
    if highlight is not None:
        h = np.asarray(highlight, bool)
        fig.add_trace(go.Scattergl(x=xy[h, 0], y=xy[h, 1], mode="markers", name="neuron-rich spots", marker=dict(size=size, color="rgba(0,0,0,0)", line=dict(color="#1d1d1b", width=1.2))))
    pad = size * 3
    fig.update_xaxes(visible=False, scaleanchor="y", constrain="domain", range=[xy[:, 0].min() - pad, xy[:, 0].max() + pad])
    fig.update_yaxes(visible=False, range=[xy[:, 1].max() + pad, xy[:, 1].min() - pad])
    return apply(fig, title, caption, height=600)

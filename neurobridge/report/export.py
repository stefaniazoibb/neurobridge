"""Export the report to Markdown (with PNG figures) and PDF (reportlab)."""

from __future__ import annotations

import re
from pathlib import Path

from neurobridge.logging_utils import get_logger, log_event

log = get_logger(__name__)


def export_figures_png(figures: dict, out_dir: Path, ids: list[str], scale: float = 2.0) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fid in ids:
        if fid not in figures:
            continue
        p = out_dir / f"{fid}.png"
        try:
            from neurobridge.viz.theme import for_print

            for_print(figures[fid]).write_image(str(p), width=900, height=int(figures[fid].layout.height or 450), scale=scale)
            paths[fid] = str(p)
        except Exception as exc:  # noqa: BLE001
            log_event(log, "export.figure_failed", figure=fid, error=str(exc))
    return paths


def export_markdown(result, ws, out_dir: Path) -> Path:
    from neurobridge.report.builder import REPORT_FIGURES, build_markdown

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = export_figures_png(ws.figures, out_dir / "figures", [f for f, _ in REPORT_FIGURES])
    rel = {k: f"figures/{Path(v).name}" for k, v in paths.items()}
    md = build_markdown(result, ws, rel)
    p = out_dir / "report.md"
    p.write_text(md)
    return p


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"<link href='\2' color='#256abf'>\1</link>", text)
    return text


def markdown_to_pdf(md_path: Path, pdf_path: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.8, leading=11.5, textColor=colors.HexColor("#1d1d1b"))
    small = ParagraphStyle("s", parent=body, fontSize=7, leading=8.6)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=colors.HexColor("#0d366b"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=10, textColor=colors.HexColor("#184f95"))
    cap = ParagraphStyle("c", parent=body, fontSize=7.6, leading=9.5, textColor=colors.HexColor("#52514e"))
    quote = ParagraphStyle("q", parent=body, leftIndent=8, textColor=colors.HexColor("#52514e"))
    base = md_path.parent
    story = []
    lines = md_path.read_text().splitlines()
    i = 0
    W = A4[0] - 30 * mm
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("<details") or ln.startswith("</details"):
            i += 1
            continue
        if ln.startswith("# "):
            story.append(Paragraph(_inline(ln[2:]), h1))
        elif ln.startswith("## "):
            story.append(Paragraph(_inline(ln[3:]), h2))
        elif ln.startswith("!["):
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln)
            if m and (base / m.group(1)).exists():
                from reportlab.lib.utils import ImageReader

                img = ImageReader(str(base / m.group(1)))
                iw, ih = img.getSize()
                w = W * 0.92
                story.append(Image(str(base / m.group(1)), width=w, height=w * ih / iw))
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append([Paragraph(_inline(c), small) for c in cells])
                i += 1
            if rows:
                t = Table(rows, repeatRows=1, colWidths=[W / len(rows[0])] * len(rows[0]))
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef9")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
                story.append(t)
                story.append(Spacer(1, 4))
            continue
        elif ln.startswith("> "):
            story.append(Paragraph(_inline(ln[2:]), quote))
        elif re.match(r"^\s*(- |\d+\. )", ln):
            indent = (len(ln) - len(ln.lstrip())) // 2
            txt = re.sub(r"^\s*(- |\d+\. )", "", ln)
            num = re.match(r"^\s*(\d+)\. ", ln)
            story.append(Paragraph(_inline(txt), ParagraphStyle("li", parent=body, leftIndent=12 + 10 * indent, bulletIndent=2 + 10 * indent), bulletText="-" if ln.strip().startswith("-") else (num.group(1) + "." if num else None)))
        elif ln.startswith("*Figure"):
            story.append(Paragraph(_inline(ln), cap))
            story.append(Spacer(1, 6))
        elif ln.strip():
            story.append(Paragraph(_inline(ln), body))
        i += 1
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm, title="NeuroBridge evidence report", author="NeuroBridge")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawString(15 * mm, 8 * mm, "NeuroBridge - research exploration only, not for clinical decision-making")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf_path

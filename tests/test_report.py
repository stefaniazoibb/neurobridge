from pathlib import Path

from neurobridge.report.export import markdown_to_pdf


def test_pdf_from_markdown(tmp_path: Path):
    md = tmp_path / "r.md"
    md.write_text("# Title\n\n## 1. Section\n\n- **bold** item with [link](https://example.org)\n1. numbered\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n*Figure: x.* caption\n")
    pdf = markdown_to_pdf(md, tmp_path / "r.pdf")
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"

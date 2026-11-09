"""Build the NeuroBridge story page from story.src.html.

Outputs:
  neurobridge-story.html  page fragment (as published as an Artifact)
  index.html              standalone page for GitHub Pages
Regenerate spatial_data.js first if the Visium analysis changes (see docs/DATA_SOURCES.md).
"""

import re
from pathlib import Path

HERE = Path(__file__).parent


def fix_tag(m):
    tag = m.group(0)
    props = re.findall(r'\s(fill|stroke|stop-color)="(var\([^"]*\))"', tag)
    if not props:
        return tag
    tag = re.sub(r'\s(fill|stroke|stop-color)="var\([^"]*\)"', "", tag)
    style = ";".join(f"{k}:{v}" for k, v in props)
    return tag.rstrip("/>").rstrip(">") + f' style="{style}"' + ("/>" if tag.endswith("/>") else ">")


src = (HERE / "story.src.html").read_text()
body, script = src.split("<script>/*SPATIAL_DATA*/</script>")
body = re.sub(r"<(path|text|stop|circle|rect|line)\b[^>]*>", fix_tag, body)  # var() colours as CSS, not SVG attributes
script = script.replace(
    "const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]);",
    'const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) { const v = String(attrs[k]); if ((k === "fill" || k === "stroke") && v.startsWith("var(")) e.style.setProperty(k, v); else e.setAttribute(k, v); }',
)
frag = body + "<script>" + (HERE / "spatial_data.js").read_text() + "</script>" + script
(HERE / "neurobridge-story.html").write_text(frag)
head = frag.split("<style>")[0]
rest = frag[len(head) + len("<style>"):].replace("</style>", "</style>\n</head>\n<body>", 1)
(HERE / "index.html").write_text(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + "<style>\n:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}\n"
    + rest + "\n</body>\n</html>\n"
)
print("built neurobridge-story.html and index.html")

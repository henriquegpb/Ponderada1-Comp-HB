"""Renderiza RELATORIO.md em PDF usando Markdown e Chrome/Chromium headless."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "RELATORIO.md"
OUTPUT = ROOT / "RELATORIO.pdf"


def find_browser() -> str:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "chrome",
        "msedge",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Chrome/Chromium não encontrado; mantenha o RELATORIO.md como versão principal.")


body = markdown.markdown(
    SOURCE.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"],
)
html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <base href="{ROOT.as_uri()}/">
  <style>
    @page {{ size: A4; margin: 18mm 17mm; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, Helvetica, sans-serif; color: #18212b; font-size: 10.5pt; line-height: 1.42; }}
    h1 {{ color: #17365d; font-size: 22pt; margin: 0 0 14pt; }}
    h2 {{ color: #244f7a; font-size: 15pt; margin: 18pt 0 7pt; break-after: avoid; }}
    h3 {{ color: #244f7a; break-after: avoid; }}
    p {{ margin: 0 0 8pt; text-align: justify; }}
    li {{ margin-bottom: 4pt; }}
    table {{ width: 100%; border-collapse: collapse; margin: 9pt 0 13pt; font-size: 8.8pt; break-inside: avoid; }}
    th {{ background: #244f7a; color: white; }}
    th, td {{ border: 1px solid #aeb9c4; padding: 4.5pt; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tr:nth-child(even) td {{ background: #f2f5f8; }}
    img {{ display: block; max-width: 78%; max-height: 185mm; margin: 10pt auto; break-inside: avoid; }}
    code {{ background: #edf1f5; padding: 1pt 3pt; border-radius: 3px; }}
    a {{ color: #155b8a; text-decoration: none; }}
  </style>
</head>
<body>{body}</body>
</html>"""

with tempfile.TemporaryDirectory(prefix="b2w-report-") as temp_dir:
    html_path = Path(temp_dir) / "relatorio.html"
    html_path.write_text(html, encoding="utf-8")
    command = [
        find_browser(),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        f"--print-to-pdf={OUTPUT}",
        html_path.as_uri(),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

print(f"PDF gerado: {OUTPUT}")

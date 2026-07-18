"""Autoescaped standalone HTML analysis rendering."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from weavevision.domain.schemas import AnalysisResult, BatchResult


def write_html_report(
    result: AnalysisResult | BatchResult, destination: Path, template_root: Path
) -> Path:
    """Render an analysis or batch report with strict HTML autoescaping."""
    environment = Environment(
        loader=FileSystemLoader(template_root),
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
    )
    template_name = (
        "batch_report.html.j2" if isinstance(result, BatchResult) else "analysis_report.html.j2"
    )
    rendered = environment.get_template(template_name).render(result=result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination

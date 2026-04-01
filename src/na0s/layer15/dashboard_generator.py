"""Dashboard generator -- produces self-contained HTML visualization.

Generates a standalone HTML file that displays the cross-benchmark
validation analysis. The HTML uses pure CSS + vanilla JavaScript and
loads its data from a companion ``dashboard_data.json`` file in the
same directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from na0s.layer15.benchmark_analyzer import BenchmarkAnalysis, BenchmarkAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color constants for coverage levels
# ---------------------------------------------------------------------------

_COLORS = {
    "strong": "#4caf50",   # green
    "partial": "#ffc107",  # yellow/amber
    "none": "#9e9e9e",     # gray
}

_TEXT_COLORS = {
    "strong": "#fff",
    "partial": "#333",
    "none": "#fff",
}


class DashboardGenerator:
    """Generates a cross-benchmark validation dashboard.

    Produces two files in ``output_dir``:
    - ``dashboard.html`` -- self-contained HTML visualization
    - ``dashboard_data.json`` -- structured data for JS consumption
    """

    def generate(
        self,
        analysis: BenchmarkAnalysis,
        output_dir: Path,
    ) -> Path:
        """Generate dashboard files.

        Parameters
        ----------
        analysis : BenchmarkAnalysis
            The analysis data to visualize.
        output_dir : Path
            Directory to write ``dashboard.html`` and
            ``dashboard_data.json`` into.

        Returns
        -------
        Path
            Path to the generated ``dashboard.html`` file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON data file.
        analyzer = BenchmarkAnalyzer()
        json_str = analyzer.to_json(analysis)
        json_path = output_dir / "dashboard_data.json"
        json_path.write_text(json_str, encoding="utf-8")

        # Write HTML file.
        html_path = output_dir / "dashboard.html"
        html_content = self._render_html(analysis)
        html_path.write_text(html_content, encoding="utf-8")

        logger.info("Dashboard written to %s", html_path)
        return html_path

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def _render_html(self, analysis: BenchmarkAnalysis) -> str:
        """Render the full HTML document."""
        benchmark_names = sorted(analysis.benchmarks.keys())

        # Build the coverage table rows (pure HTML -- works without JS).
        table_rows = self._render_coverage_rows(analysis, benchmark_names)

        # Build the gap summary lists.
        bench_unique_items = "".join(
            f"<li>{_esc(item)}</li>" for item in analysis.benchmark_unique
        )
        na0s_unique_items = "".join(
            f"<li>{_esc(cid)}</li>" for cid in analysis.na0s_unique
        )

        # Priority gaps: categories with most gaps first, top 5.
        priority_gaps = sorted(
            analysis.coverage, key=lambda c: len(c.gaps), reverse=True
        )[:5]
        priority_items = "".join(
            f"<li><strong>{_esc(c.category_id)}</strong> "
            f"({_esc(c.category_name)}): {len(c.gaps)} gap(s)</li>"
            for c in priority_gaps
            if c.gaps
        )

        # Per-benchmark overlap stats.
        bench_stats = ""
        for bn in benchmark_names:
            matched_cats = sum(
                1
                for c in analysis.coverage
                if c.benchmark_matches.get(bn, 0) > 0
            )
            pct = (
                round(matched_cats / analysis.na0s_categories * 100, 1)
                if analysis.na0s_categories
                else 0.0
            )
            bench_stats += (
                f"<tr><td>{_esc(bn)}</td>"
                f"<td>{analysis.benchmarks[bn]}</td>"
                f"<td>{matched_cats}/{analysis.na0s_categories} "
                f"({pct}%)</td></tr>"
            )

        # Benchmark column headers.
        bench_headers = "".join(
            f"<th>{_esc(bn)}</th>" for bn in benchmark_names
        )

        timestamp_str = analysis.timestamp.strftime("%Y-%m-%d %H:%M UTC")

        return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Na0S Cross-Benchmark Validation</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    margin: 0; padding: 0;
    background: #fafafa; color: #222;
  }}
  .container {{
    max-width: 1200px; margin: 0 auto; padding: 24px;
  }}

  /* --- Header --- */
  header {{
    background: #1a237e; color: #fff; padding: 24px 32px;
  }}
  header h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
  header .meta {{ font-size: 0.9rem; opacity: 0.85; }}

  /* --- Stats box --- */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin: 24px 0;
  }}
  .stat-card {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 16px; text-align: center;
  }}
  .stat-card .num {{ font-size: 2rem; font-weight: 700; color: #1a237e; }}
  .stat-card .label {{ font-size: 0.85rem; color: #666; margin-top: 4px; }}

  /* --- Coverage table --- */
  .coverage-section {{ margin: 32px 0; }}
  .coverage-section h2 {{ font-size: 1.2rem; margin-bottom: 12px; }}
  table.coverage {{
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
  }}
  table.coverage th, table.coverage td {{
    padding: 8px 12px; text-align: left; border: 1px solid #ddd;
  }}
  table.coverage th {{
    background: #e8eaf6; position: sticky; top: 0; z-index: 1;
  }}
  table.coverage tr:nth-child(even) {{ background: #f5f5f5; }}
  .cell-strong {{
    background: {_COLORS["strong"]}; color: {_TEXT_COLORS["strong"]};
    text-align: center; font-weight: 600;
  }}
  .cell-partial {{
    background: {_COLORS["partial"]}; color: {_TEXT_COLORS["partial"]};
    text-align: center; font-weight: 600;
  }}
  .cell-none {{
    background: {_COLORS["none"]}; color: {_TEXT_COLORS["none"]};
    text-align: center;
  }}

  /* --- Gap summary --- */
  .gap-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
    margin: 32px 0;
  }}
  .gap-card {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 16px;
  }}
  .gap-card h3 {{ margin: 0 0 8px; font-size: 1rem; }}
  .gap-card ul {{ margin: 0; padding-left: 20px; font-size: 0.85rem; }}
  .gap-card li {{ margin-bottom: 4px; }}
  .gap-card .empty {{ color: #999; font-style: italic; }}

  /* --- Benchmark stats table --- */
  table.bench-stats {{
    border-collapse: collapse; font-size: 0.85rem; margin-top: 12px;
  }}
  table.bench-stats th, table.bench-stats td {{
    padding: 6px 12px; border: 1px solid #ddd; text-align: left;
  }}
  table.bench-stats th {{ background: #e8eaf6; }}

  /* --- Priority gaps --- */
  .priority {{ margin: 24px 0; }}
  .priority h3 {{ font-size: 1rem; margin-bottom: 8px; }}
  .priority ul {{ padding-left: 20px; font-size: 0.85rem; }}
  .priority li {{ margin-bottom: 4px; }}

  /* --- Legend --- */
  .legend {{ display: flex; gap: 16px; margin: 16px 0; font-size: 0.85rem; }}
  .legend-item {{
    display: flex; align-items: center; gap: 6px;
  }}
  .legend-swatch {{
    width: 16px; height: 16px; border-radius: 3px; border: 1px solid #ccc;
  }}

  @media print {{
    header {{ background: #333; }}
    .container {{ padding: 8px; }}
  }}
  @media (max-width: 700px) {{
    .gap-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <h1>Na0S Cross-Benchmark Validation</h1>
  <div class="meta">
    Generated: {timestamp_str} |
    {analysis.na0s_categories} categories,
    {analysis.na0s_techniques} techniques |
    {len(analysis.benchmarks)} benchmark(s) loaded
  </div>
</header>

<div class="container">

  <!-- Stats Box -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="num">{analysis.na0s_categories}</div>
      <div class="label">Na0S Categories</div>
    </div>
    <div class="stat-card">
      <div class="num">{analysis.na0s_techniques}</div>
      <div class="label">Na0S Techniques</div>
    </div>
    <div class="stat-card">
      <div class="num">{analysis.overall_overlap_pct}%</div>
      <div class="label">Overall Overlap</div>
    </div>
    <div class="stat-card">
      <div class="num">{len(analysis.benchmark_unique)}</div>
      <div class="label">Benchmark-Only Items</div>
    </div>
  </div>

  <!-- Per-benchmark stats -->
  <section>
    <h2>Benchmark Summary</h2>
    <table class="bench-stats">
      <tr><th>Benchmark</th><th>Items</th><th>Category Overlap</th></tr>
      {bench_stats if bench_stats else "<tr><td colspan='3'><em>No benchmarks loaded</em></td></tr>"}
    </table>
  </section>

  <!-- Legend -->
  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:{_COLORS['strong']}"></div>
      Strong (&gt;0.4)
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:{_COLORS['partial']}"></div>
      Partial (0.15-0.4)
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:{_COLORS['none']}"></div>
      None (&lt;0.15)
    </div>
  </div>

  <!-- Coverage Table -->
  <section class="coverage-section">
    <h2>Coverage Heatmap</h2>
    <table class="coverage">
      <thead>
        <tr>
          <th>Category</th>
          <th>Na0S Techniques</th>
          {bench_headers}
          <th>Level</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </section>

  <!-- Gap Summary -->
  <div class="gap-grid">
    <div class="gap-card">
      <h3>Benchmark Items Na0S Should Add</h3>
      {f"<ul>{bench_unique_items}</ul>" if bench_unique_items else '<p class="empty">None -- Na0S covers all benchmark items</p>'}
    </div>
    <div class="gap-card">
      <h3>Na0S-Unique Coverage (Strength)</h3>
      {f"<ul>{na0s_unique_items}</ul>" if na0s_unique_items else '<p class="empty">All Na0S categories overlap with benchmarks</p>'}
    </div>
  </div>

  <!-- Priority Gaps -->
  <div class="priority">
    <h3>Priority Gaps (Top 5)</h3>
    {f"<ul>{priority_items}</ul>" if priority_items else '<p class="empty">No gaps detected</p>'}
  </div>

</div>

<script>
// Load dashboard_data.json for any dynamic use.
// The HTML is fully readable without JS -- this is for future extension.
(function() {{
  fetch('dashboard_data.json')
    .then(function(r) {{ return r.json(); }})
    .then(function(data) {{
      console.log('Dashboard data loaded:', data.na0s_categories, 'categories,',
                  Object.keys(data.benchmarks).length, 'benchmarks');
    }})
    .catch(function(e) {{
      console.warn('Could not load dashboard_data.json:', e);
    }});
}})();
</script>

</body>
</html>
"""

    def _render_coverage_rows(
        self,
        analysis: BenchmarkAnalysis,
        benchmark_names: list[str],
    ) -> str:
        """Render HTML table rows for the coverage heatmap."""
        rows: list[str] = []
        for cov in analysis.coverage:
            bench_cells = ""
            for bn in benchmark_names:
                count = cov.benchmark_matches.get(bn, 0)
                if count > 0:
                    ratio = (
                        count / cov.na0s_technique_count
                        if cov.na0s_technique_count
                        else 0
                    )
                    cls = "cell-strong" if ratio >= 0.3 else "cell-partial"
                    bench_cells += f'<td class="{cls}">{count}</td>'
                else:
                    bench_cells += '<td class="cell-none">0</td>'

            level_cls = f"cell-{cov.coverage_level}"
            rows.append(
                f"<tr>"
                f"<td><strong>{_esc(cov.category_id)}</strong> "
                f"{_esc(cov.category_name)}</td>"
                f"<td>{cov.na0s_technique_count}</td>"
                f"{bench_cells}"
                f'<td class="{level_cls}">{cov.coverage_level}</td>'
                f"</tr>"
            )
        return "\n        ".join(rows)


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

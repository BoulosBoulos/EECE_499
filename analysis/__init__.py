"""Phase 1E analysis pipeline.

Components:
    config   — constants (color palette, method order, alpha, bootstrap config)
    loader   — discover and load results into long_df / wide_df
    quality  — data quality checks + report
    metrics  — outcome metric computation
    stats    — Welch's t-test, Holm correction, Cohen's d, bootstrap CI
    tables   — CSV + LaTeX table generation
    plots    — matplotlib (PDF) + plotly (HTML) figures
    run_analysis — CLI orchestrator
"""

__all__ = [
    "config",
    "loader",
    "quality",
    "metrics",
    "stats",
    "tables",
    "plots",
    "run_analysis",
]

# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Residual Analysis Report Generator
# MAGIC
# MAGIC Reads manuscript artifacts from `outputs/{figures,tables,data}/` and writes
# MAGIC `outputs/reports/edi_report_residual_analysis.html` plus
# MAGIC `outputs/manifest/edi_artifact_manifest.csv`.
# MAGIC
# MAGIC Invoked automatically at the end of `06_residual_analysis.py` via `%run ./99_report`.

# COMMAND ----------

import base64
import re
from pathlib import Path

import pandas as pd

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

ROOT = Path.cwd()
FIGURE_DIR = ROOT / EDI_FIGURES_DIR
TABLE_DIR = ROOT / EDI_TABLES_DIR
DATA_DIR = ROOT / EDI_DATA_DIR
REPORT_DIR = ROOT / EDI_REPORTS_DIR
OUTPUT_HTML = REPORT_DIR / EDI_REPORT_HTML
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# -- Small-cell suppression --------------------------------------------------
# Privacy policy: in aggregated count columns, cells in (0, SUPPRESSION_THRESHOLD)
# are replaced with the string "<10". Zero and values >= threshold pass through.
# Applied uniformly to the HTML report's inline tables and to the CSV exports
# in the manuscript_artifacts package. Individual-record CSVs (one row per
# patient-year -- no aggregated cell counts) bypass suppression at export time.

SUPPRESSION_THRESHOLD = 10
SUPPRESSION_MARKER    = f"<{SUPPRESSION_THRESHOLD}"
SUPPRESSION_FOOTNOTE  = (
    f"Cell counts of 1-{SUPPRESSION_THRESHOLD - 1} suppressed as "
    f"\"{SUPPRESSION_MARKER}\" to protect small subgroups. "
    "Zero and >= threshold values pass through unchanged."
)
_INDIVIDUAL_RECORD_SRCS = EDI_INDIVIDUAL_RECORD_TABLES

_COUNT_COL_PATTERNS = [
    re.compile(r"^n$",                         re.IGNORECASE),
    re.compile(r"^n[_\-\s]",                   re.IGNORECASE),
    re.compile(r"(^|[_\-\s])count($|[_\-\s])", re.IGNORECASE),
    re.compile(r"n_obs",                       re.IGNORECASE),
    re.compile(r"sample[_\-\s]?size",          re.IGNORECASE),
]


def _is_count_col(col: str) -> bool:
    return any(p.search(col) for p in _COUNT_COL_PATTERNS)


def _obfuscate_counts(df: pd.DataFrame) -> int:
    """Mask cells in count-like columns whose numeric value is in (0, threshold).
    Mutates df in place. Returns the total number of cells masked."""
    n_masked = 0
    for col in df.columns:
        if not _is_count_col(col):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        mask = (numeric > 0) & (numeric < SUPPRESSION_THRESHOLD)
        if mask.any():
            df[col] = df[col].astype(object)
            df.loc[mask, col] = SUPPRESSION_MARKER
            n_masked += int(mask.sum())
    return n_masked


# COMMAND ----------

# -- I/O helpers -------------------------------------------------------------

def _img_tag(path: Path, alt: str = "", width: str = "100%") -> str:
    """Encode image as base64 data URI so the HTML file is fully self-contained."""
    if not path.exists():
        return f'<p class="missing">&#9888; Figure not found: {path.name}</p>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'alt="{alt}" style="width:{width};max-width:1200px;">'
    )


_REFERENCE_LABEL_COLS = ("cluster_label", "vs_label")


def _annotate_reference(df: pd.DataFrame) -> None:
    """If the df carries an is_reference flag, append '(reference)' to the
    human-readable cluster/label column for reference rows. Purely cosmetic --
    makes inline HTML tables read like the forest plot without extra styling."""
    if "is_reference" not in df.columns:
        return
    for col in _REFERENCE_LABEL_COLS:
        if col in df.columns:
            mask = df["is_reference"].fillna(False).astype(bool)
            if mask.any():
                df[col] = df[col].astype(object)
                df.loc[mask, col] = df.loc[mask, col].astype(str) + " (reference)"
            return  # one label column is enough


def _load_csv(directory: Path, filename: str) -> pd.DataFrame:
    """Load a CSV; return empty DataFrame if missing."""
    path = directory / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if filename not in _INDIVIDUAL_RECORD_SRCS:
        _obfuscate_counts(df)
    _annotate_reference(df)
    return df


def _table_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Render a DataFrame as a compact styled HTML table."""
    if df.empty:
        return '<p class="missing">&#9888; Table data not found.</p>'
    return df.head(max_rows).to_html(
        index=False,
        border=0,
        classes="data-table",
        float_format=lambda x: f"{x:.4f}",
    )

# COMMAND ----------

# -- Statistical labelling (mirrors 99_utils; duplicated for Serverless %run isolation) --

def sig_label(p: float) -> str:
    """APA-style significance codes: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def rho_interp(rho: float) -> str:
    """Interpret Spearman rho magnitude and direction (negligible/weak/moderate/strong)."""
    mag = abs(rho)
    direction = "positive" if rho > 0 else "negative"
    if mag < 0.1:
        return "negligible"
    elif mag < 0.3:
        return f"weak {direction}"
    elif mag < 0.5:
        return f"moderate {direction}"
    else:
        return f"strong {direction}"

# COMMAND ----------

# -- Load all analytic tables ------------------------------------------------

q1          = _load_csv(DATA_DIR, EDI_DATA01_Q1)
q1b         = _load_csv(DATA_DIR, EDI_DATA02_Q1B)
q2          = _load_csv(DATA_DIR, EDI_DATA03_Q2)
domain_iqr  = _load_csv(DATA_DIR, EDI_DATA05_DOMAIN_SUMMARY)
domain_kw   = _load_csv(DATA_DIR, EDI_DATA06_DOMAIN_KW)
domain_rb   = _load_csv(DATA_DIR, EDI_DATA07_DOMAIN_RB)
q3_or       = _load_csv(DATA_DIR, EDI_DATA04_Q3)
summary_all = _load_csv(DATA_DIR, EDI_DATA08_SUMMARY)

# COMMAND ----------

# -- Narrative generators ----------------------------------------------------

def _q1_narrative() -> str:
    if q1.empty:
        return "<p class='missing'>Q1 results not available.</p>"
    lines = []
    for _, row in q1.iterrows():
        rho  = row.get("spearman_rho", float("nan"))
        p    = row.get("p_value", float("nan"))
        n    = int(row.get("n_person_years", 0))
        out  = row.get("outcome", "")
        sign = "+" if rho > 0 else ""
        if rho > 0.05:
            interp = "Higher CCI is associated with <em>more</em> deviation from the cluster mean in this domain."
        elif rho < -0.05:
            interp = "Higher CCI is associated with <em>less</em> deviation -- sicker patients are closer to the cluster centre."
        else:
            interp = "No meaningful association between CCI and within-cluster deviation in this domain."
        lines.append(
            f"<li><b>{out}</b>: &#961; = {sign}{rho:.3f} ({rho_interp(rho)}), "
            f"p {sig_label(p)}, N = {n:,} person-years. {interp}</li>"
        )
    return "<ul>" + "\n".join(lines) + "</ul>"


def _q1b_narrative() -> str:
    if q1b.empty:
        return "<p class='missing'>Q1b results not available.</p>"
    if "outcome" not in q1b.columns:
        return "<p>Stratified Spearman results by cluster (see table below). Re-run 06_residual_analysis.py to generate outcome-labelled summary.</p>"
    notes = []
    for outcome, grp in q1b.groupby("outcome"):
        rhos         = grp["spearman_rho"].values
        has_positive = any(r >  0.05 for r in rhos)
        has_negative = any(r < -0.05 for r in rhos)
        if has_positive and has_negative:
            notes.append(
                f"<li><b>{outcome}</b>: opposing &#961; signs across clusters -- "
                "evidence of Simpson's paradox. The pooled rho masks cluster-specific dynamics.</li>"
            )
        else:
            max_rho = grp.loc[grp["spearman_rho"].abs().idxmax()]
            notes.append(
                f"<li><b>{outcome}</b>: &#961; consistent in sign across clusters. "
                f"Strongest effect in <em>{max_rho.get('cluster_label', '?')}</em> "
                f"(&#961; = {max_rho['spearman_rho']:.3f}).</li>"
            )
    return "<ul>" + "\n".join(notes) + "</ul>"


def _q2_narrative() -> str:
    if q2.empty:
        return "<p class='missing'>Q2 OLS results not available.</p>"
    n_sig   = int((q2["p_value"] < 0.05).sum())
    n_total = len(q2)
    lines = [
        f"<p>{n_sig} of {n_total} interaction terms are significant at p &lt; 0.05 "
        "(clustered SE by person_id). A significant term means the CCI-residual slope "
        "for that cluster differs from the Outpatient Regular baseline.</p>",
        "<ul>",
    ]
    for _, row in q2.iterrows():
        p    = row.get("p_value", float("nan"))
        coef = row.get("coef", float("nan"))
        term = row.get("term", "")
        out  = row.get("outcome", "")
        dir_ = "steeper (more positive)" if coef > 0 else "shallower (more negative)"
        interp = (
            f"CCI slope is {dir_} than in Outpatient Regular."
            if p < 0.05
            else "Slope not significantly different from Outpatient Regular."
        )
        lines.append(
            f"<li><b>{out}</b> -- <code>{term}</code>: "
            f"coef = {coef:+.4f}, p {sig_label(p)}. {interp}</li>"
        )
    lines.append("</ul>")
    return "\n".join(lines)


def _q3_narrative() -> str:
    if q3_or.empty:
        return "<p class='missing'>Q3 results not available.</p>"
    lines = [
        "<p>Each odds ratio reflects the multiplicative change in odds of belonging to "
        "that cluster vs Outpatient Regular for each 1-unit increase in cumulative CCI.</p>",
        "<ul>",
    ]
    for _, row in q3_or.sort_values("odds_ratio", ascending=False).iterrows():
        OR    = row.get("odds_ratio", float("nan"))
        p     = row.get("p_value", float("nan"))
        label = row.get("vs_label", "?")
        ci_lo = row.get("ci_lo_95", float("nan"))
        ci_hi = row.get("ci_hi_95", float("nan"))
        ci_str = f" [95% CI: {ci_lo:.2f}-{ci_hi:.2f}]" if pd.notna(ci_lo) else ""
        dir_  = "more" if OR > 1 else "less"
        lines.append(
            f"<li><b>{label}</b>: OR = {OR:.3f}{ci_str}, p {sig_label(p)}. "
            f"Each 1-unit CCI increase makes patients {dir_} likely to be in this cluster vs Outpatient Regular.</li>"
        )
    lines.append("</ul>")
    return "\n".join(lines)


def _rb_narrative() -> str:
    if domain_rb.empty:
        return "<p class='missing'>Rank-biserial results not available.</p>"
    lines = []
    for cluster, grp in domain_rb.groupby("Cluster"):
        extremes = grp.sort_values("Rank_Biserial")
        most_neg = extremes.iloc[0]
        most_pos = extremes.iloc[-1]
        lines.append(
            f"<li><b>{cluster}</b>: most under-represented domain = "
            f"<em>{most_neg['Domain']}</em> (rb = {most_neg['Rank_Biserial']:.2f}); "
            f"most over-represented = <em>{most_pos['Domain']}</em> "
            f"(rb = {most_pos['Rank_Biserial']:.2f}).</li>"
        )
    return "<ul>" + "\n".join(lines) + "</ul>"

# COMMAND ----------

# -- CSS ---------------------------------------------------------------------

CSS = """
:root {
    --bg:         #fafafa;
    --surface:    #ffffff;
    --border:     #d4d6d9;
    --text:       #1c1c1c;
    --muted:      #5c5c5c;
    --accent:     #0052cc;
    --warn:       #c0392b;
    --section-bg: #f0f4f8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
    font-size: 15px;
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
}
.page { max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }
h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: 6px; }
h2 {
    font-size: 1.35rem; font-weight: 700; margin: 48px 0 12px;
    padding: 10px 16px; background: var(--section-bg);
    border-left: 4px solid var(--accent); border-radius: 3px;
}
h3 { font-size: 1.05rem; font-weight: 600; margin: 28px 0 8px; color: var(--accent); }
h4 {
    font-size: 0.9rem; font-weight: 600; margin: 18px 0 6px;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em;
}
p  { margin: 8px 0; }
ul { margin: 8px 0 8px 20px; }
li { margin: 5px 0; }
code {
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.85em; background: #eef1f4;
    padding: 1px 5px; border-radius: 3px;
}
.meta { color: var(--muted); font-size: 0.88rem; margin-bottom: 24px; }
.callout {
    background: #e8f0fe; border-left: 4px solid var(--accent);
    padding: 12px 16px; margin: 16px 0; border-radius: 3px; font-size: 0.93rem;
}
.missing { color: var(--warn); font-style: italic; font-size: 0.9rem; }
.figure-wrap { margin: 20px 0; text-align: center; }
.figure-wrap img { border: 1px solid var(--border); border-radius: 4px; }
.figure-caption { font-size: 0.83rem; color: var(--muted); margin-top: 6px; font-style: italic; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 12px 0; }
.data-table th {
    background: var(--section-bg); border: 1px solid var(--border);
    padding: 6px 10px; text-align: left; font-weight: 600;
}
.data-table td { border: 1px solid var(--border); padding: 5px 10px; vertical-align: top; }
.data-table tr:nth-child(even) td { background: #f7f8fa; }
hr { border: none; border-top: 1px solid var(--border); margin: 40px 0; }
.toc {
    background: var(--surface); border: 1px solid var(--border);
    padding: 16px 20px; border-radius: 4px; margin: 24px 0; display: inline-block;
}
.toc ol { margin-left: 18px; }
.toc li { margin: 4px 0; }
.toc a { color: var(--accent); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
"""

# COMMAND ----------

# -- HTML section builders ---------------------------------------------------

def _section_cover() -> str:
    sup_max    = SUPPRESSION_THRESHOLD - 1
    sup_marker = SUPPRESSION_MARKER
    sup_thr    = SUPPRESSION_THRESHOLD
    return f"""
<h1>Residual Analysis Report -- Cohort 1</h1>
<p class="meta">
    CCI vs Within-Cluster Domain Residuals &nbsp;|&nbsp;
    Generated {date.today().strftime('%B %d, %Y')} &nbsp;|&nbsp;
    Source: <code>06_residual_analysis.py</code>
</p>
<div class="callout">
    <b>Higher-level question:</b> Is disease burden (cumulative CCI) associated with
    cluster assignment and within-cluster variation in clinical data?
    Residuals are computed from OMOP domain counts (conditions, drugs, procedures,
    measurements) -- features <em>not</em> used in the GMM-4 clustering,
    so they are independent of what drove cluster assignment.
</div>
<div class="callout" style="background:#fff4e5;border-left-color:#c86b00;">
    <b>Cell suppression policy:</b> In all tables and figures below, aggregated
    cell counts in the range 1-{sup_max} are suppressed as "{sup_marker}" to
    protect small subgroups. Zero and counts &#8805; {sup_thr} are shown unchanged.
    Figures are rendered from the same analytic frames as the backing CSVs, so
    any value a figure resolves is subject to the same rule.
</div>
<div class="toc">
    <b>Contents</b>
    <ol>
        <li><a href="#q1">Q1 -- Pooled Spearman: does CCI predict domain deviation?</a></li>
        <li><a href="#q1b">Q1b -- Stratified Spearman: does the rho vary by cluster?</a></li>
        <li><a href="#q2">Q2 -- OLS interaction: does the CCI slope differ across clusters?</a></li>
        <li><a href="#domain">Domain data landscape: how much domain-level stuff does each archetype have?</a></li>
        <li><a href="#q3">Q3 -- Multinomial logit: does CCI predict cluster assignment?</a></li>
        <li><a href="#summary">Consolidated results table</a></li>
    </ol>
</div>
"""


def _section_q1() -> str:
    fig_abs = _img_tag(
        FIGURE_DIR / EDI_FIG02_ABS_RESID,
        alt="Within-cluster absolute domain deviation by archetype",
    )
    return f"""
<h2 id="q1">1. Q1 -- Pooled Spearman: Is higher CCI associated with within-cluster domain deviation?</h2>

<div class="callout">
    <b>Interpretation guide:</b> A positive &#961; means sicker patients (higher CCI) deviate
    <em>more</em> from their cluster's average domain count. A negative &#961; means sicker
    patients are <em>closer</em> to the cluster centre. Because patients contribute multiple
    person-years, p-values are inflated -- focus on the magnitude of &#961;, not significance.
</div>

<h3>Results</h3>
{_q1_narrative()}

<h3>Unsigned domain deviation by archetype</h3>
<p>The violin plots below show how far patients stray from their cluster mean for each domain.
Wider violins = more heterogeneity within that archetype. Compare across archetypes to see
which clusters are internally homogeneous vs. mixed.</p>
<div class="figure-wrap">
    {fig_abs}
    <p class="figure-caption">Within-cluster absolute deviation in domain counts by
    utilization archetype (violin = distribution; bar = IQR; dot = median).</p>
</div>

<h4>Full Q1 results table</h4>
{_table_html(q1)}
"""


def _section_q1b() -> str:
    return f"""
<h2 id="q1b">2. Q1b -- Stratified Spearman: Does the rho vary by cluster?</h2>

<div class="callout">
    <b>Why this matters:</b> If the pooled &#961; (Q1) is near zero but stratified rhos have
    opposing signs across clusters, Simpson's paradox is at play -- restricting to a
    single archetype would reveal a real relationship that the pooled analysis obscures.
    Bonferroni-corrected alpha = 0.05 / 4 = 0.0125.
</div>

<h3>Simpson's paradox check</h3>
{_q1b_narrative()}

<h4>Full Q1b stratified results table</h4>
{_table_html(q1b)}
"""


def _section_q2() -> str:
    return f"""
<h2 id="q2">3. Q2 -- OLS Interaction: Does the CCI slope differ across clusters?</h2>

<div class="callout">
    <b>Model:</b> <code>residual ~ max_cci * C(gmm_4_cluster)</code> with clustered
    standard errors by <code>person_id</code>. Baseline = Outpatient Regular (cluster 2).
    Signed residuals are kept on their mean-centered scale; unsigned residuals with
    skewness &gt; 1.0 are log1p-transformed before fitting.
    A significant interaction term means the CCI-residual slope for that cluster
    differs statistically from the Outpatient Regular slope.
</div>

<h3>Interaction term results</h3>
{_q2_narrative()}

<h4>Full OLS interaction terms table</h4>
{_table_html(q2)}
"""


def _section_domain() -> str:
    fig_domain = _img_tag(
        FIGURE_DIR / EDI_FIG04_DOMAIN_BREADTH,
        alt="OMOP domain counts by utilization archetype",
    )
    fig_rb = _img_tag(
        FIGURE_DIR / EDI_FIG03_RANK_BISERIAL,
        alt="Rank-biserial heatmap: each cluster vs full cohort",
        width="70%",
    )
    return f"""
<h2 id="domain">4. Domain Data Landscape: How much domain-level stuff does each archetype have?</h2>

<div class="callout">
    <b>Context:</b> A researcher restricting their study to a single utilization archetype
    would inherit a systematically different distribution of clinical data. These analyses
    quantify the direction and magnitude of that selection effect.
</div>

<h3>Total domain concept occurrences per year by archetype</h3>
<p>Compare violin shapes across clusters within each domain. A long upper tail indicates
information-rich outliers. The IQR bar locates the typical patient in that archetype.</p>
<div class="figure-wrap">
    {fig_domain}
    <p class="figure-caption">OMOP domain counts (total concept occurrences/year) by utilization archetype
    (violin = distribution; bar = IQR; dot = median).</p>
</div>

<h3>Rank-biserial effect size: each cluster vs full cohort</h3>
<p>Values near +1 = cluster patients rank above the cohort for that domain (over-represented);
near -1 = under-represented. Near 0 = representative of the full cohort.</p>
<div class="figure-wrap">
    {fig_rb}
    <p class="figure-caption">Rank-biserial effect size comparing each cluster to the full cohort
    per domain. Red = higher than cohort; blue = lower.</p>
</div>

<h3>Key selection effects by archetype</h3>
{_rb_narrative()}

<h4>Kruskal-Wallis: do domain counts differ across clusters?</h4>
{_table_html(domain_kw)}

<h4>Per-cluster median / IQR summary</h4>
{_table_html(domain_iqr)}

<h4>Rank-biserial: each cluster vs full cohort</h4>
{_table_html(domain_rb)}
"""


def _section_q3() -> str:
    fig_forest = _img_tag(
        FIGURE_DIR / EDI_FIG01_Q3_FOREST,
        alt="Q3 forest plot: odds ratios with 95% CI",
        width="65%",
    )
    return f"""
<h2 id="q3">5. Q3 -- Multinomial Logit: Does CCI predict cluster assignment?</h2>

<div class="callout">
    <b>Model:</b> <code>gmm_4_cluster ~ max_cci</code> on each patient's most recent
    person-year (so max_cci captures lifetime disease burden). Baseline = Outpatient Regular (cluster 2).
    Odds ratios show the multiplicative change in odds of cluster membership per 1-unit
    increase in cumulative CCI.
</div>

<h3>Results</h3>
{_q3_narrative()}

<h3>Forest plot: odds ratios with 95% CI</h3>
<p>Points to the right of OR = 1 (dashed line) indicate higher CCI predicts membership
in that cluster vs Outpatient Regular. An OR gradient from low to high acuity confirms that
CCI discriminates archetypes in the expected clinical direction.</p>
<div class="figure-wrap">
    {fig_forest}
    <p class="figure-caption">Multinomial logit odds ratios (95% Wald CI) for a 1-unit CCI
    increase vs Outpatient Regular baseline.</p>
</div>

<h4>Full Q3 odds ratio table</h4>
{_table_html(q3_or)}
"""


def _section_summary() -> str:
    return f"""
<h2 id="summary">6. Consolidated Results Table</h2>
<p>All Q1, Q1b, Q2, and Q3 results in a single table, ordered by outcome and question.</p>
{_table_html(summary_all)}
"""

# COMMAND ----------

# -- Assemble and write ------------------------------------------------------

def build_report() -> None:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    body = "\n".join([
        _section_cover(),
        "<hr>",
        _section_q1(),
        "<hr>",
        _section_q1b(),
        "<hr>",
        _section_q2(),
        "<hr>",
        _section_domain(),
        "<hr>",
        _section_q3(),
        "<hr>",
        _section_summary(),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Residual Analysis Report -- Cohort 1</title>
    <style>{CSS}</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Report written -> {OUTPUT_HTML}")

# COMMAND ----------

build_report()
manifest_path = write_edi_artifact_manifest(str(ROOT / EDI_MANIFEST_PATH))
print(f"Artifact manifest -> {manifest_path}")

"""case_comparison -- shared machinery for Hermes-3 campaign/study reports.

A campaign script declares a page set and a list of studies; this package does
everything else (loading, colours, provenance, PDF assembly). See campaign.py
for the usage example, and pages/__init__.py for how pages are found.

Import matplotlib and select a headless backend BEFORE importing this package
if the script runs without a display:

    import matplotlib
    matplotlib.use("Agg")
    from case_comparison import Campaign

Public API
----------
Campaign        the campaign object: config, @study registry, run(), cli()
CaseSet         a study's case selection (what pages receive as ctx.cases)
PageContext     what a page is handed: ctx.cases / .slug / .notes / .campaign
register_page   register a built-in page by name (campaign-local pages are
                passed as plain callables instead -- see resolve_pages)
stack_figures   combine several figures onto one PDF page
raw_page        mark a figure to be written verbatim, skipping letterboxing
text_table      monospace-aligned table, for text pages
provenance      git SHAs, BOUT.log.0 parsing, run status, option diffing
"""

from .campaign import Campaign, resolve_pages
from .caseset import CaseSet, CaseLoader, MissingCase, DEFAULT_PALETTE
from .seriesset import SeriesSet, DEFAULT_MARKERS
from .report import PageContext, Report, stack_figures, raw_page, text_table
from .pages import register_page, available as available_pages
from . import provenance, scalars

__all__ = [
    "Campaign",
    "resolve_pages",
    "CaseSet",
    "CaseLoader",
    "MissingCase",
    "SeriesSet",
    "DEFAULT_MARKERS",
    "DEFAULT_PALETTE",
    "PageContext",
    "Report",
    "stack_figures",
    "raw_page",
    "text_table",
    "register_page",
    "available_pages",
    "provenance",
    "scalars",
]

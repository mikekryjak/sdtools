"""Built-in pages, resolved lazily by name.

Every page module is imported ONLY when a campaign actually asks for one of its
pages. This is deliberate: some plotting stacks in sdtools are expensive or
fragile to import (code_comparison hard-imports the SOLEDGE wrapper, which
pulls in tkinter and fails in the hermes3 spack env). A campaign that doesn't
list a page never pays for its imports.

Adding a built-in page: write it in a module here, decorate it with
``@register_page("name")``, and add the name -> module entry to _PAGE_MODULES.

A campaign-specific page does NOT belong here -- pass the function itself in
the campaign's `pages` list instead. Move it here only once it has proved
general across campaigns.
"""

import importlib

# page name -> module in this package providing it
_PAGE_MODULES = {
    "cover": "cover",
    "profiles": "profiles",
    "profiles_vs_solps": "profiles_vs_solps",
    "monitor": "monitor",
    "performance": "performance",
    "runtime_breakdown": "runtime",
    # scan pages -- these take a SeriesSet (ctx.series), not a CaseSet
    "scan_cover": "scan",
    "scan_scalars": "scan",
    "scan_performance": "scan",
    "scan_tradeoff": "scan",
}

_REGISTRY = {}


def register_page(name):
    """Register a page callable under `name` (see the module docstring)."""
    def deco(fn):
        _REGISTRY[name] = fn
        fn._cc_page_name = name
        return fn
    return deco


def resolve(name):
    """Return the page callable registered as `name`, importing it on demand."""
    if name not in _REGISTRY:
        module = _PAGE_MODULES.get(name)
        if module is None:
            raise KeyError(
                f"Unknown page '{name}'. Built-in pages: "
                f"{', '.join(sorted(_PAGE_MODULES))}. For a campaign-specific "
                f"page, put the function itself in the pages list."
            )
        importlib.import_module(f".{module}", __name__)
        if name not in _REGISTRY:
            raise KeyError(
                f"Page module '{module}' does not register a page named "
                f"'{name}'"
            )
    return _REGISTRY[name]


def available():
    """Names of every built-in page."""
    return sorted(_PAGE_MODULES)

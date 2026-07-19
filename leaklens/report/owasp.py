"""Finding -> OWASP LLM Top-10 category tagging (FR-4).

A data-driven lookup keyed by `finding.module` — the one place in the report layer that
knows about specific modules, and even here it's a table, not code branching. Inversion
maps to both the vector-store weakness and the disclosure it enables. A module absent from
the table falls back to whatever primary category it set on its own Finding, so a new
surface still gets tagged without editing this file.
"""
LLM08 = "LLM08: Vector and Embedding Weaknesses"
LLM06 = "LLM06: Sensitive Information Disclosure"
UNCATEGORIZED = "LLM00: Uncategorized"

# module name -> ordered OWASP categories (primary first). Only modules that exist.
_CATEGORIES = {
    "inversion": [LLM08, LLM06],
}


def categories_for(finding) -> list[str]:
    """Return the OWASP categories for a Finding, primary first. Never raises."""
    cats = _CATEGORIES.get(finding.module)
    if cats:
        return list(cats)
    if finding.owasp:
        return [finding.owasp]
    return [UNCATEGORIZED]


def primary(finding) -> str:
    """The single headline OWASP category for a Finding."""
    return categories_for(finding)[0]

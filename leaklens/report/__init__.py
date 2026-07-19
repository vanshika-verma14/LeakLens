"""Report layer — renders a list of `Finding`s to terminal / JSON / HTML.

Every renderer here consumes ONLY the Finding shape and iterates uniformly; none
branches on which module produced a Finding. The single module-aware spot is the
data-driven OWASP lookup table in `owasp.py`.
"""

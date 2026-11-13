"""graphify fs: turn a folder of everyday files into a navigable knowledge graph.

On top of graphify's content graph (concepts extracted from documents, slides,
PDFs, images and media transcripts) this package adds a file-system layer:

- folder and file nodes (``contains`` edges), with dates, sizes and kinds
- tag nodes from macOS Finder tags, Office/PDF keywords and local-LLM topic tags
  (``tagged_with`` edges), plus ``co_occurs_with`` relations between tags
- ``mentions`` edges from each file to the concepts extracted from it

The result is written to ``<index>/graphify-out/fs/graph.json`` and served to
LLMs over MCP (``graphify fs mcp``) and to people through a local web explorer
with a small local LLM that navigates the graph (``graphify fs explore``).
"""

FS_GRAPH_NAME = "graph.json"

__all__ = ["FS_GRAPH_NAME"]

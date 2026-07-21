"""KB index build helpers (thin wrappers over :class:`KBService`).

Kept as a small module so the CLI has a stable ``build_index`` entrypoint while
all real logic lives in :class:`wellness.kb.search.KBService`.
"""

from __future__ import annotations

from wellness.config import Settings
from wellness.kb.search import KBService, get_kb_service


def build_index(settings: Settings | None = None, force: bool = False) -> int:
    """Build (or rebuild) the KB index. Returns the number of chunks indexed.

    Args:
        settings: Optional settings override.
        force: Rebuild even if an up-to-date index already exists.
    """
    service = KBService(settings) if settings is not None else get_kb_service()
    if force:
        return service.build_index()
    service.ensure_index()
    return len(service.load_chunks())


def load_service(settings: Settings | None = None) -> KBService:
    """Return the singleton KB service with its index ensured."""
    service = get_kb_service(settings)
    service.ensure_index()
    return service

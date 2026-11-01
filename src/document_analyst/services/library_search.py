from __future__ import annotations

from pathlib import Path


def filter_library_documents(
    documents: list[dict], query: str = "", statuses: list[str] | None = None,
    file_types: list[str] | None = None,
) -> list[dict]:
    query = query.strip().casefold()
    status_filter = set(statuses or [])
    type_filter = {extension.casefold() for extension in file_types or []}
    return [item for item in documents if (
        (not query or query in str(item.get("document_name", "")).casefold()
         or query in str(item.get("source_path", "")).casefold())
        and (not status_filter or item.get("status", "indexed") in status_filter)
        and (not type_filter or Path(str(item.get("document_name", ""))).suffix.casefold() in type_filter)
    )]

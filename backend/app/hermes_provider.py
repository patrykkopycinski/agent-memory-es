"""Hermes memory-provider adapter: binds agent-memory-es to Hermes's memory tool contract
(retain/recall/reflect shapes). Run alongside the FastAPI service; calls memory ops directly
(in-process) to avoid HTTP overhead on Hermes's hot path."""
import os
from typing import Optional

from . import auth, memory


class HermesMemoryProvider:
    """Adapter object with the three Hermes memory surfaces."""

    def __init__(self, owner_id: str, api_key: Optional[str] = None):
        self.owner_id = owner_id
        # optionally verify the key maps to this owner
        if api_key:
            who = auth.owner_of(api_key)
            if not who or who["owner_id"] != owner_id:
                raise PermissionError("api key does not belong to owner")

    def retain(self, content: str, context: str = "", tags=None) -> dict:
        kind = "episodic" if tags and "event" in tags else "semantic"
        return memory.retain(self.owner_id, kind, f"{content}" + (f" | ctx: {context}" if context else ""))

    def recall(self, query: str, limit: int = 8) -> list:
        return memory.recall(self.owner_id, query, size=limit)["fused"]

    def reflect(self, query: str) -> str:
        return memory.reflect(self.owner_id, query)["answer"]

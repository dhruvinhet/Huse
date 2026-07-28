"""Reusable visual teaching knowledge."""

from app.knowledge.repository import InMemoryVisualKnowledgeBase
from app.knowledge.persistent import JsonVisualKnowledgeBase, KnowledgeRecord

__all__ = [
    "InMemoryVisualKnowledgeBase",
    "JsonVisualKnowledgeBase",
    "KnowledgeRecord",
]

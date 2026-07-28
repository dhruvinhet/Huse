"""Reusable semantic educational template library."""

from app.templates.builtins import builtin_templates
from app.templates.registry import SemanticTemplate, TemplateRegistry

__all__ = ["SemanticTemplate", "TemplateRegistry", "builtin_templates"]

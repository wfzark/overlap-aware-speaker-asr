"""Overlap-aware speaker-attributed ASR project package."""

from .llm_repair_loop import run_repair_loop_all, run_repair_loop_offline
from .rag_repair import build_reference_knowledge_base, retrieve_relevant_segments

# Lazy import: matplotlib may not be installed in all environments
def compute_feature_importance(*args, **kwargs):
    from .router_feature_importance import compute_feature_importance as _fn
    return _fn(*args, **kwargs)

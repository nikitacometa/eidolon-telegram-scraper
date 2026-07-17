"""Versioned fingerprint for the complete effective watcher decision policy."""

from __future__ import annotations

import hashlib
import json

from config.settings import settings
from config.watchers import Watcher
from pipeline.embeddings import INDEX_SCHEMA_VERSION
from pipeline.llm import PROMPT_VERSION

POLICY_FINGERPRINT_SCHEMA = 1
RULES_POLICY_VERSION = "rules-v1"


def effective_policy_fingerprint(watcher: Watcher) -> str:
    """Hash every persisted input that can change a recovered decision."""
    payload = {
        "schema": POLICY_FINGERPRINT_SCHEMA,
        "rules_version": RULES_POLICY_VERSION,
        "watcher": watcher.model_dump(mode="json"),
        "embeddings": {
            "index_schema": INDEX_SCHEMA_VERSION,
            "model": settings.embedding_model,
            "threshold": (
                watcher.embedding_threshold
                if watcher.embedding_threshold is not None
                else settings.embedding_similarity_threshold
            ),
            "negative_margin": settings.embedding_negative_margin,
        },
        "llm": {
            "model": settings.llm_model,
            "prompt_version": PROMPT_VERSION,
            "timeout_seconds": settings.llm_timeout_seconds,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

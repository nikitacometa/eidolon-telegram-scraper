"""Config-driven semantic relevance filter (Level 2)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import IncludeEnum, PyEmbeddings
from openai import AsyncOpenAI

from config.settings import settings
from config.watchers import Watcher
from pipeline.models import EmbeddingDecision, StageStatus

logger = logging.getLogger(__name__)

INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ReferenceText:
    """A human-authored semantic reference with its expected relevance label."""

    text: str
    label: str


class EmbeddingFilter:
    """Semantic similarity against versioned positive and negative references."""

    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        chroma_client: ClientAPI | None = None,
    ) -> None:
        self._openai = openai_client
        self._chroma = chroma_client
        self._owns_openai = False
        self._collections: dict[str, Collection] = {}

    async def start(self, watchers: list[Watcher]) -> None:
        """Initialize providers and seed fingerprinted watcher indexes."""
        if self._openai is None:
            if not settings.openai_api_key:
                logger.warning("OPENAI_API_KEY not set, embedding filter disabled")
                return
            self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
            self._owns_openai = True

        if self._chroma is None:
            settings.chroma_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._chroma = chromadb.PersistentClient(path=str(settings.chroma_path))

        for watcher in watchers:
            if watcher.llm_level >= 2:
                await self._seed_collection(watcher)

        logger.info(
            "Embedding filter ready: collections=%d default_threshold=%.2f",
            len(self._collections),
            settings.embedding_similarity_threshold,
        )

    async def close(self) -> None:
        if self._openai and self._owns_openai:
            await self._openai.close()
        self._openai = None
        self._owns_openai = False

    async def check(self, text: str, watcher_name: str) -> EmbeddingDecision:
        """Return an explainable semantic-stage decision.

        Provider failures remain fail-open but are explicitly marked `DEGRADED`.
        """
        collection = self._collections.get(watcher_name)
        if not self._openai or collection is None:
            return EmbeddingDecision(
                passed=True,
                status=StageStatus.DEGRADED,
                reason="embedding provider or watcher index unavailable",
                error_code="provider_disabled",
            )

        try:
            embedding = await self._embed(text[:2000])
            reference_count = await asyncio.to_thread(collection.count)
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=cast(PyEmbeddings, [embedding]),
                n_results=min(8, reference_count),
                include=[
                    IncludeEnum.distances,
                    IncludeEnum.documents,
                    IncludeEnum.metadatas,
                ],
            )

            distances = (results.get("distances") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            if not distances:
                return EmbeddingDecision(
                    passed=True,
                    status=StageStatus.DEGRADED,
                    reason="vector index returned no references",
                    error_code="empty_index_result",
                )

            best_positive: tuple[float, str] | None = None
            best_negative: tuple[float, str] | None = None
            for distance, document, metadata in zip(
                distances,
                documents,
                metadatas,
                strict=False,
            ):
                similarity = 1.0 - float(distance)
                reference = document or ""
                label = (metadata or {}).get("label", "positive")
                candidate = (similarity, reference)
                if label == "negative":
                    if best_negative is None or candidate[0] > best_negative[0]:
                        best_negative = candidate
                elif best_positive is None or candidate[0] > best_positive[0]:
                    best_positive = candidate

            if best_positive is None:
                return EmbeddingDecision(
                    passed=True,
                    status=StageStatus.DEGRADED,
                    reason="vector index has no positive references",
                    error_code="missing_positive_reference",
                )

            threshold = _collection_threshold(collection)
            negative_score = best_negative[0] if best_negative else -1.0
            passed = (
                best_positive[0] >= threshold
                and best_positive[0] >= negative_score + settings.embedding_negative_margin
            )
            return EmbeddingDecision(
                passed=passed,
                status=StageStatus.OK,
                score=round(best_positive[0], 6),
                matched_reference=best_positive[1],
                reason=(
                    "positive reference cleared threshold and negative margin"
                    if passed
                    else "semantic score below threshold or negative margin"
                ),
            )
        except Exception as error:
            logger.error(
                "Embedding filter degraded: watcher=%s error=%s",
                watcher_name,
                type(error).__name__,
            )
            return EmbeddingDecision(
                passed=True,
                status=StageStatus.DEGRADED,
                reason="embedding stage failed; accepted by fail-open policy",
                error_code=type(error).__name__,
            )

    async def _seed_collection(self, watcher: Watcher) -> None:
        """Create or rebuild a watcher index when its fingerprint changes."""
        if self._chroma is None:
            raise RuntimeError("Chroma client not initialized")

        references = _build_reference_records(watcher)
        if not references:
            logger.warning("Watcher %s has no semantic references; L2 disabled", watcher.name)
            return

        fingerprint = _reference_fingerprint(references)
        collection_name = _collection_name(watcher.name)
        collection = await asyncio.to_thread(
            self._chroma.get_or_create_collection,
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "eidolon:fingerprint": fingerprint,
                "eidolon:threshold": watcher.embedding_threshold
                or settings.embedding_similarity_threshold,
                "eidolon:schema_version": INDEX_SCHEMA_VERSION,
            },
        )
        metadata = collection.metadata or {}
        if (
            await asyncio.to_thread(collection.count) > 0
            and metadata.get("eidolon:fingerprint") != fingerprint
        ):
            logger.info("Rebuilding stale semantic index for watcher=%s", watcher.name)
            await asyncio.to_thread(self._chroma.delete_collection, collection_name)
            collection = await asyncio.to_thread(
                self._chroma.create_collection,
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "eidolon:fingerprint": fingerprint,
                    "eidolon:threshold": watcher.embedding_threshold
                    or settings.embedding_similarity_threshold,
                    "eidolon:schema_version": INDEX_SCHEMA_VERSION,
                },
            )

        if await asyncio.to_thread(collection.count) == 0:
            texts = [reference.text for reference in references]
            embeddings = await self._embed_many(texts)
            await asyncio.to_thread(
                collection.add,
                ids=[
                    hashlib.sha256(reference.text.encode()).hexdigest()[:24]
                    for reference in references
                ],
                embeddings=cast(PyEmbeddings, embeddings),
                documents=texts,
                metadatas=[{"label": reference.label} for reference in references],
            )
            logger.info(
                "Seeded semantic index: watcher=%s references=%d fingerprint=%s",
                watcher.name,
                len(references),
                fingerprint[:12],
            )

        self._collections[watcher.name] = collection

    async def _embed(self, text: str) -> list[float]:
        response = await self._require_openai().embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        response = await self._require_openai().embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def _require_openai(self) -> AsyncOpenAI:
        if self._openai is None:
            raise RuntimeError("OpenAI client not initialized")
        return self._openai


def _build_reference_records(watcher: Watcher) -> list[ReferenceText]:
    positive = list(watcher.examples.positive)
    negative = list(watcher.examples.negative)

    if not positive:
        positive.extend(watcher.rules.keywords)
        if watcher.prompt.strip():
            positive.append(watcher.prompt.strip())

    deduplicated: dict[tuple[str, str], ReferenceText] = {}
    for label, texts in (("positive", positive), ("negative", negative)):
        for text in texts:
            normalized = text.strip()
            if normalized:
                deduplicated[(label, normalized)] = ReferenceText(normalized, label)
    return list(deduplicated.values())


def _build_reference_texts(watcher: Watcher) -> list[str]:
    """Compatibility helper used by tests and evaluation tooling."""
    return [reference.text for reference in _build_reference_records(watcher)]


def _reference_fingerprint(references: list[ReferenceText]) -> str:
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": settings.embedding_model,
        "references": [
            {"label": reference.label, "text": reference.text} for reference in references
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _collection_name(watcher_name: str) -> str:
    safe_name = re.sub(r"[^a-z0-9-]", "-", watcher_name.lower()).strip("-")
    suffix = hashlib.sha256(watcher_name.encode()).hexdigest()[:8]
    return f"eidolon-{safe_name[:45]}-{suffix}"


def _collection_threshold(collection: Collection) -> float:
    raw = (collection.metadata or {}).get(
        "eidolon:threshold",
        settings.embedding_similarity_threshold,
    )
    return cast(float, raw)

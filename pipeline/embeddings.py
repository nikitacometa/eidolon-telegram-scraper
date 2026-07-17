"""Config-driven semantic relevance filter (Level 2)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import IncludeEnum, PyEmbeddings
from chromadb.config import Settings as ChromaSettings
from openai import AsyncOpenAI

from config.settings import settings
from config.watchers import Watcher
from pipeline.models import EmbeddingDecision, StageStatus

logger = logging.getLogger(__name__)

INDEX_SCHEMA_VERSION = 4


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
        self._reference_texts: dict[str, dict[str, str]] = {}

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
            self._chroma = chromadb.PersistentClient(
                path=str(settings.chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )

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

        Provider failures return a degraded neutral decision; watcher policy decides.
        """
        started = time.perf_counter()
        collection = self._collections.get(watcher_name)
        if not self._openai or collection is None:
            return EmbeddingDecision(
                passed=True,
                status=StageStatus.DEGRADED,
                reason="embedding provider or watcher index unavailable",
                model=settings.embedding_model,
                latency_ms=_elapsed_ms(started),
                error_code="provider_disabled",
            )

        try:
            embedding, input_tokens, provider_model = await self._embed(text[:2000])
            reference_count = await asyncio.to_thread(collection.count)
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=cast(PyEmbeddings, [embedding]),
                # Watcher examples are bounded to 200 entries. Querying the
                # complete corpus guarantees that both labels are considered.
                n_results=reference_count,
                include=[
                    IncludeEnum.distances,
                    IncludeEnum.metadatas,
                ],
            )

            distances = (results.get("distances") or [[]])[0]
            reference_ids = (results.get("ids") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            if not distances:
                return EmbeddingDecision(
                    passed=True,
                    status=StageStatus.DEGRADED,
                    reason="vector index returned no references",
                    model=provider_model,
                    latency_ms=_elapsed_ms(started),
                    input_tokens=input_tokens,
                    error_code="empty_index_result",
                )

            best_positive: tuple[float, str] | None = None
            best_negative: tuple[float, str] | None = None
            references = self._reference_texts.get(watcher_name, {})
            for distance, reference_id, metadata in zip(
                distances,
                reference_ids,
                metadatas,
                strict=False,
            ):
                similarity = 1.0 - float(distance)
                reference = references.get(str(reference_id), "")
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
                    model=provider_model,
                    latency_ms=_elapsed_ms(started),
                    input_tokens=input_tokens,
                    error_code="missing_positive_reference",
                )

            threshold = _collection_threshold(collection)
            negative_margin = _collection_negative_margin(collection)
            negative_score = best_negative[0] if best_negative else -1.0
            passed = (
                best_positive[0] >= threshold
                and best_positive[0] >= negative_score + negative_margin
            )
            if best_positive[0] < threshold:
                reason = "positive similarity below calibrated threshold"
            elif best_positive[0] < negative_score + negative_margin:
                reason = "negative reference won the configured margin"
            else:
                reason = "positive reference cleared threshold and negative margin"
            return EmbeddingDecision(
                passed=passed,
                status=StageStatus.OK,
                score=round(best_positive[0], 6),
                negative_score=(round(best_negative[0], 6) if best_negative is not None else None),
                threshold=threshold,
                negative_margin=negative_margin,
                matched_reference=best_positive[1],
                matched_negative_reference=(
                    best_negative[1] if best_negative is not None else None
                ),
                model=provider_model,
                latency_ms=_elapsed_ms(started),
                input_tokens=input_tokens,
                reason=reason,
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
                reason="embedding stage failed; watcher degradation policy decides",
                model=settings.embedding_model,
                latency_ms=_elapsed_ms(started),
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

        reference_ids = [_reference_id(reference) for reference in references]
        self._reference_texts[watcher.name] = {
            reference_id: reference.text
            for reference_id, reference in zip(reference_ids, references, strict=True)
        }
        fingerprint = _reference_fingerprint(references)
        collection_name = _collection_name(watcher.name)
        desired_metadata = _collection_metadata(watcher, references, fingerprint)
        collection = await asyncio.to_thread(
            self._chroma.get_or_create_collection,
            name=collection_name,
            metadata=desired_metadata,
        )
        metadata = collection.metadata or {}
        if await asyncio.to_thread(collection.count) > 0 and any(
            metadata.get(key) != value for key, value in desired_metadata.items()
        ):
            logger.info("Rebuilding stale semantic index or policy for watcher=%s", watcher.name)
            await asyncio.to_thread(self._chroma.delete_collection, collection_name)
            collection = await asyncio.to_thread(
                self._chroma.create_collection,
                name=collection_name,
                metadata=desired_metadata,
            )

        if await asyncio.to_thread(collection.count) == 0:
            texts = [reference.text for reference in references]
            embeddings = await self._embed_many(texts)
            await asyncio.to_thread(
                collection.add,
                ids=reference_ids,
                embeddings=cast(PyEmbeddings, embeddings),
                metadatas=[{"label": reference.label} for reference in references],
            )
            logger.info(
                "Seeded semantic index: watcher=%s references=%d fingerprint=%s",
                watcher.name,
                len(references),
                fingerprint[:12],
            )

        self._collections[watcher.name] = collection

    async def _embed(self, text: str) -> tuple[list[float], int, str]:
        response = await self._require_openai().embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        raw_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", 0)
        input_tokens = raw_tokens if isinstance(raw_tokens, int) else 0
        raw_model = getattr(response, "model", None)
        provider_model = raw_model if isinstance(raw_model, str) else settings.embedding_model
        return response.data[0].embedding, input_tokens, provider_model

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


def _reference_id(reference: ReferenceText) -> str:
    """Build a stable opaque vector ID without persisting reference plaintext."""
    return hashlib.sha256(f"{reference.label}:{reference.text}".encode()).hexdigest()[:24]


def _collection_metadata(
    watcher: Watcher,
    references: list[ReferenceText],
    fingerprint: str,
) -> dict[str, str | int | float]:
    """Build the complete persisted index and decision-policy contract."""
    return {
        "hnsw:space": "cosine",
        "eidolon:fingerprint": fingerprint,
        "eidolon:threshold": (
            watcher.embedding_threshold
            if watcher.embedding_threshold is not None
            else settings.embedding_similarity_threshold
        ),
        "eidolon:negative_margin": settings.embedding_negative_margin,
        "eidolon:positive_count": sum(reference.label == "positive" for reference in references),
        "eidolon:negative_count": sum(reference.label == "negative" for reference in references),
        "eidolon:schema_version": INDEX_SCHEMA_VERSION,
    }


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


def _collection_negative_margin(collection: Collection) -> float:
    raw = (collection.metadata or {}).get(
        "eidolon:negative_margin",
        settings.embedding_negative_margin,
    )
    return cast(float, raw)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)

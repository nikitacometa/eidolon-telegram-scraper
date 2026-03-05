"""Embedding similarity filter (Level 2) — cheap pre-filter before LLM.

Uses OpenAI text-embedding-3-small to embed incoming messages and compare
against reference vectors derived from watcher keywords and prompt.
Stores vectors in ChromaDB (embedded mode, persistent on disk).
"""

from __future__ import annotations

import logging

import chromadb
from openai import AsyncOpenAI

from config.settings import settings
from config.watchers import Watcher

logger = logging.getLogger(__name__)


class EmbeddingFilter:
    """Level 2 filter: semantic similarity against watcher reference texts.

    Messages above the similarity threshold pass through to Level 3 (LLM).
    Fail-open: if embedding fails, the message passes.
    """

    def __init__(self) -> None:
        self._openai: AsyncOpenAI | None = None
        self._chroma: chromadb.ClientAPI | None = None
        self._collections: dict[str, chromadb.Collection] = {}

    async def start(self, watchers: list[Watcher]) -> None:
        """Initialize OpenAI client and ChromaDB, seed reference vectors."""
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set, embedding filter disabled")
            return

        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(settings.chroma_path))

        for watcher in watchers:
            if watcher.llm_level < 2:
                continue
            await self._seed_collection(watcher)

        logger.info(
            "Embedding filter ready: %d collections, threshold=%.2f",
            len(self._collections),
            settings.embedding_similarity_threshold,
        )

    async def close(self) -> None:
        if self._openai:
            await self._openai.close()
            self._openai = None

    async def check(self, text: str, watcher_name: str) -> bool:
        """Check if message is semantically similar to watcher references.

        Returns True if similar enough (should continue to LLM), False if not.
        Fail-open: returns True on any error.
        """
        if not self._openai or watcher_name not in self._collections:
            return True  # fail-open

        try:
            embedding = await self._embed(text[:1000])
            collection = self._collections[watcher_name]
            results = collection.query(
                query_embeddings=[embedding],
                n_results=1,
            )

            if not results["distances"] or not results["distances"][0]:
                return True  # fail-open

            # ChromaDB returns L2 distance by default; convert to cosine similarity
            # For normalized vectors: cosine_distance = 1 - cosine_similarity
            # ChromaDB cosine distance is already 1 - similarity
            distance = results["distances"][0][0]
            similarity = 1.0 - distance

            passed = similarity >= settings.embedding_similarity_threshold
            logger.debug(
                "Embedding [%s]: similarity=%.3f threshold=%.2f → %s",
                watcher_name, similarity, settings.embedding_similarity_threshold,
                "PASS" if passed else "DROP",
            )
            return passed

        except Exception as e:
            logger.error("Embedding filter error (fail-open): %s", e)
            return True

    async def _seed_collection(self, watcher: Watcher) -> None:
        """Create or load a ChromaDB collection with reference texts for a watcher."""
        collection_name = f"eidolon-{watcher.name}"[:63]  # ChromaDB name limit
        collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Skip seeding if already populated
        if collection.count() > 0:
            self._collections[watcher.name] = collection
            logger.info("Loaded existing collection [%s]: %d refs", watcher.name, collection.count())
            return

        # Generate reference texts from keywords and prompt
        ref_texts = _build_reference_texts(watcher)
        if not ref_texts:
            return

        # Embed all reference texts
        embeddings = []
        for text in ref_texts:
            emb = await self._embed(text)
            embeddings.append(emb)

        collection.add(
            ids=[f"ref-{i}" for i in range(len(ref_texts))],
            embeddings=embeddings,
            documents=ref_texts,
        )
        self._collections[watcher.name] = collection
        logger.info("Seeded collection [%s]: %d reference texts", watcher.name, len(ref_texts))

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for a text string."""
        response = await self._openai.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding


def _build_reference_texts(watcher: Watcher) -> list[str]:
    """Generate reference texts from watcher config for seeding embeddings.

    Creates synthetic "positive example" messages from keywords and prompt.
    """
    texts = []

    # From keywords: create short example messages
    keywords = watcher.rules.keywords
    if keywords:
        # Group keywords into example sentences
        for kw in keywords:
            texts.append(f"Available for rent: {kw} on Phangan island")
            texts.append(f"Сдаю {kw} на Пангане, хорошая цена")

    # From watcher prompt: use as a reference document
    if watcher.prompt:
        texts.append(watcher.prompt.strip())

    # Add some generic positive examples for housing
    if watcher.name == "phangan-housing":
        texts.extend([
            "Beautiful villa for rent, 3 bedrooms, pool, near Srithanu, 25000 THB/month",
            "Сдаю дом на Пангане, 2 спальни, рядом с морем, 15000 бат в месяц",
            "Bungalow available from March, quiet area, furnished, WiFi included",
            "Apartment for rent in Thong Sala, studio, 8000 baht, long term",
        ])

    return texts

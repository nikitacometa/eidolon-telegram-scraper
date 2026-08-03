#!/usr/bin/env python3
"""Build and inspect the search index.

    python3 index_cli.py build            # sync + embed + extract, one pass
    python3 index_cli.py build --no-extract
    python3 index_cli.py sync             # cheapest: copy new rows, no API calls
    python3 index_cli.py extract --limit 2000
    python3 index_cli.py status
    python3 index_cli.py search "концерт бар живая музыка"
    python3 index_cli.py places --event concert --city "Da Nang"
    python3 index_cli.py refs             # unjoined chats mentioned in the corpus

Safe to interrupt: every stage tracks its own cursor and resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from config.settings import settings
from pipeline.indexer import EmbeddingIndexer, PlaceExtractor, build_index
from storage.search import SearchDatabase, build_fts_query
from storage.session_lock import SessionInUseError, SessionLock

# Commands that spend money or write the index. A timer tick landing on top of
# a long manual pass would re-issue the same LLM calls for the same rows: the
# writes are idempotent, the billing is not.
_EXCLUSIVE = {"build", "sync", "embed", "extract"}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def _run(args: argparse.Namespace) -> int:
    if args.command in _EXCLUSIVE:
        lock = SessionLock(
            settings.search_db_path.with_suffix(".lock"),
            owner=f"index-{args.command}",
            subject="The search index",
        )
        try:
            lock.acquire()
        except SessionInUseError as exc:
            print(json.dumps({"skipped": str(exc)}))
            return 0
        try:
            return await _dispatch(args)
        finally:
            lock.release()
    return await _dispatch(args)


async def _dispatch(args: argparse.Namespace) -> int:
    with SearchDatabase(settings.search_db_path) as search:
        search.connect()

        if args.command == "status":
            print(json.dumps(search.status(), indent=2, ensure_ascii=False))
            return 0

        if args.command == "sync":
            report = search.sync(live_db=settings.db_path, scout_db=settings.scout_db_path)
            refs = search.refresh_chat_references()
            print(json.dumps({"synced": report, "references": refs}, indent=2))
            return 0

        if args.command == "embed":
            count = await EmbeddingIndexer(search).run()
            print(json.dumps({"embedded": count}, indent=2))
            return 0

        if args.command == "extract":
            report = await PlaceExtractor(search).run(limit=args.limit)
            print(json.dumps(report, indent=2))
            return 0

        if args.command == "build":
            report = await build_index(
                search=search,
                do_sync=not args.no_sync,
                do_embed=not args.no_embed,
                do_extract=not args.no_extract,
                extract_limit=args.limit,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        if args.command == "search":
            terms = args.query.split()
            vector = None
            if not args.lexical_only:
                vector = await EmbeddingIndexer(search).embed_query(args.query)
            hits = search.hybrid_search(
                match_query=build_fts_query(terms),
                query_vector=vector,
                limit=args.limit,
            )
            print(
                json.dumps([r.as_dict(text_limit=280) for r in hits], indent=2, ensure_ascii=False)
            )
            return 0

        if args.command == "places":
            places = search.search_places(
                name_query=args.name,
                city_area=args.city,
                place_type=args.type,
                event_types=[args.event] if args.event else None,
                min_mentions=args.min_mentions,
                limit=args.limit,
            )
            print(json.dumps(places, indent=2, ensure_ascii=False))
            return 0

        if args.command == "refs":
            references = search.chat_references(
                only_unknown=not args.all, min_mentions=args.min_mentions, limit=args.limit
            )
            print(json.dumps(references, indent=2, ensure_ascii=False))
            return 0

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="index counts and backlogs")
    sub.add_parser("sync", help="copy new messages from both stores")
    sub.add_parser("embed", help="embed messages that have no vector yet")

    p_extract = sub.add_parser("extract", help="run the venue extraction pass")
    p_extract.add_argument("--limit", type=int, default=500)

    p_build = sub.add_parser("build", help="sync, embed and extract in one pass")
    p_build.add_argument("--limit", type=int, default=500)
    p_build.add_argument("--no-sync", action="store_true")
    p_build.add_argument("--no-embed", action="store_true")
    p_build.add_argument("--no-extract", action="store_true")

    p_search = sub.add_parser("search", help="hybrid search over the corpus")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--lexical-only", action="store_true")

    p_places = sub.add_parser("places", help="query the extracted venue index")
    p_places.add_argument("--name")
    p_places.add_argument("--city")
    p_places.add_argument("--type")
    p_places.add_argument("--event")
    p_places.add_argument("--min-mentions", type=int, default=1)
    p_places.add_argument("--limit", type=int, default=40)

    p_refs = sub.add_parser("refs", help="chat references mined from message text")
    p_refs.add_argument("--all", action="store_true", help="include already-known chats")
    p_refs.add_argument("--min-mentions", type=int, default=2)
    p_refs.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()
    _configure_logging(args.verbose)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

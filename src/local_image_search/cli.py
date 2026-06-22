from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from local_image_search.clip import make_clip_embedder
from local_image_search.config import DEFAULT_DB_PATH
from local_image_search.db import connect, count_images, init_db
from local_image_search.index_service import IndexProgress, index_roots
from local_image_search.metrics import format_memory_status
from local_image_search.search_service import SearchService

DEFAULT_API_PORT = 8766


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-search")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the SQLite database")
    init_parser.set_defaults(handler=handle_init)

    status_parser = subparsers.add_parser("status", help="Show index status")
    status_parser.set_defaults(handler=handle_status)

    index_parser = subparsers.add_parser("index", help="Index one or more folders")
    index_parser.add_argument("roots", nargs="+", type=Path, help="Image files or folders")
    index_parser.add_argument(
        "--clip-embedder",
        default="open-clip",
        choices=["stub", "open-clip", "openclip", "clip"],
    )
    index_parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print indexing progress every N processed images",
    )
    index_parser.set_defaults(handler=handle_index)

    search_parser = subparsers.add_parser("search", help="Search indexed images")
    search_parser.add_argument("query", help="Natural language search query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument(
        "--clip-embedder",
        default="open-clip",
        choices=["stub", "open-clip", "openclip", "clip"],
    )
    search_parser.set_defaults(handler=handle_search)

    similar_parser = subparsers.add_parser(
        "similar",
        help="Find indexed images visually similar to a local image",
    )
    similar_parser.add_argument("image", type=Path, help="Reference image path")
    similar_parser.add_argument("--limit", type=int, default=10)
    similar_parser.add_argument(
        "--clip-embedder",
        default="open-clip",
        choices=["stub", "open-clip", "openclip", "clip"],
    )
    similar_parser.set_defaults(handler=handle_similar)

    serve_parser = subparsers.add_parser("serve", help="Run the local search API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    serve_parser.add_argument(
        "--clip-embedder",
        default="open-clip",
        choices=["stub", "open-clip", "openclip", "clip"],
    )
    serve_parser.set_defaults(handler=handle_serve)

    return parser


def handle_init(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        init_db(conn)
    print(f"initialized {args.db}")
    return 0


def handle_status(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        init_db(conn)
        total = count_images(conn)
    print(f"database: {args.db}")
    print(f"indexed images: {total}")
    return 0


def handle_index(args: argparse.Namespace) -> int:
    clip_embedder = make_clip_embedder(args.clip_embedder)
    started = time.perf_counter()

    print(f"clip embedder: {clip_embedder.name}")
    print(format_memory_status())

    result = index_roots(
        args.db,
        args.roots,
        clip_embedder,
        on_progress=lambda progress: _print_index_progress(
            progress,
            started=started,
            progress_every=args.progress_every,
        ),
    )

    print(f"scanned: {result.total}")
    print(f"indexed: {result.indexed}")
    print(f"skipped unchanged: {result.skipped}")
    print(f"deleted missing: {result.deleted}")
    return 0


def _print_index_progress(
    progress: IndexProgress,
    *,
    started: float,
    progress_every: int,
) -> None:
    if progress_every <= 0:
        return
    if progress.total == 0:
        return
    if progress.processed != progress.total and progress.processed % progress_every != 0:
        return

    elapsed = time.perf_counter() - started
    percent = (progress.processed / progress.total * 100) if progress.total else 100.0
    print(
        f"[{progress.processed}/{progress.total} {percent:5.1f}%] "
        f"indexed={progress.indexed} skipped={progress.skipped} "
        f"elapsed={_format_elapsed(elapsed)} "
        f"{format_memory_status()} "
        f"last={progress.last_file}"
    )


def _format_elapsed(seconds: float) -> str:
    minutes, whole_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}"
    return f"{minutes:d}:{whole_seconds:02d}"


def handle_search(args: argparse.Namespace) -> int:
    clip_embedder = make_clip_embedder(args.clip_embedder)
    response = SearchService(args.db, clip_embedder).search(args.query, args.limit)
    return _print_results(response["results"])


def handle_similar(args: argparse.Namespace) -> int:
    clip_embedder = make_clip_embedder(args.clip_embedder)
    response = SearchService(args.db, clip_embedder).similar(args.image, args.limit)
    return _print_results(response["results"])


def _print_results(results: list[dict]) -> int:
    if not results:
        print("no results")
        return 0

    for result in results:
        print(f"{result['score']:.3f}  {result['path']}")
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    clip_embedder = make_clip_embedder(args.clip_embedder)
    from local_image_search.server import run_server

    run_server(args.db, clip_embedder, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

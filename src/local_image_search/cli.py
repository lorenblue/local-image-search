from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from local_image_search.clip import make_clip_embedder
from local_image_search.config import DEFAULT_DB_PATH
from local_image_search.db import (
    connect,
    count_images,
    delete_missing_paths,
    ensure_vector_table,
    get_thumbnail_path,
    init_db,
    needs_indexing,
    update_thumbnail_path,
    upsert_indexed_image,
)
from local_image_search.metrics import format_memory_status
from local_image_search.scanner import scan_images
from local_image_search.search_service import SearchService
from local_image_search.thumbnails import ensure_thumbnail


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
        "--delete-missing",
        action="store_true",
        help="Remove database records that were not seen during this scan",
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
    serve_parser.add_argument("--port", type=int, default=8765)
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
    images = scan_images(args.roots)
    total = len(images)
    started = time.perf_counter()

    print(f"found {total} supported images")
    print(f"clip embedder: {clip_embedder.name}")
    print(format_memory_status())

    with connect(args.db) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        indexed = 0
        skipped = 0
        processed = 0
        for image in images:
            processed += 1
            if not needs_indexing(conn, image, clip_embedder.name):
                skipped += 1
                _ensure_stored_thumbnail(conn, image.path)
                conn.commit()
                _print_index_progress(
                    processed=processed,
                    total=total,
                    indexed=indexed,
                    skipped=skipped,
                    started=started,
                    file_name=image.file_name,
                    progress_every=args.progress_every,
                )
                continue
            embedding = clip_embedder.embed_image(image.path)
            thumbnail_path = ensure_thumbnail(image.path)
            upsert_indexed_image(
                conn,
                image,
                clip_embedder.name,
                embedding,
                thumbnail_path,
            )
            indexed += 1
            conn.commit()
            _print_index_progress(
                processed=processed,
                total=total,
                indexed=indexed,
                skipped=skipped,
                started=started,
                file_name=image.file_name,
                progress_every=args.progress_every,
            )
        deleted = (
            delete_missing_paths(conn, [image.path for image in images])
            if args.delete_missing
            else 0
        )
        conn.commit()

    print(f"scanned: {len(images)}")
    print(f"indexed: {indexed}")
    print(f"skipped unchanged: {skipped}")
    if args.delete_missing:
        print(f"deleted missing: {deleted}")
    return 0


def _ensure_stored_thumbnail(conn: sqlite3.Connection, image_path: Path) -> None:
    existing = get_thumbnail_path(conn, image_path)
    if existing is not None and existing.exists():
        return
    thumbnail_path = ensure_thumbnail(image_path)
    if thumbnail_path is None:
        return
    update_thumbnail_path(conn, image_path, thumbnail_path)


def _print_index_progress(
    *,
    processed: int,
    total: int,
    indexed: int,
    skipped: int,
    started: float,
    file_name: str,
    progress_every: int,
) -> None:
    if progress_every <= 0:
        return
    if processed != total and processed % progress_every != 0:
        return

    elapsed = time.perf_counter() - started
    percent = (processed / total * 100) if total else 100.0
    print(
        f"[{processed}/{total} {percent:5.1f}%] "
        f"indexed={indexed} skipped={skipped} "
        f"elapsed={_format_elapsed(elapsed)} "
        f"{format_memory_status()} "
        f"last={file_name}"
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

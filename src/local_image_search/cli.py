from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from local_image_search.captioning import make_captioner
from local_image_search.config import DEFAULT_DB_PATH
from local_image_search.db import (
    connect,
    count_images,
    delete_missing_paths,
    init_db,
    list_indexed_images,
    needs_indexing,
    upsert_indexed_image,
)
from local_image_search.embeddings import make_embedder
from local_image_search.scanner import scan_images
from local_image_search.search import search_images


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
    index_parser.add_argument("--captioner", default="stub", choices=["stub", "moondream"])
    index_parser.add_argument(
        "--embedder",
        default="stub",
        choices=["stub", "sentence-transformers", "sentence-transformer", "st"],
    )
    index_parser.add_argument(
        "--delete-missing",
        action="store_true",
        help="Remove database records that were not seen during this scan",
    )
    index_parser.set_defaults(handler=handle_index)

    search_parser = subparsers.add_parser("search", help="Search indexed captions")
    search_parser.add_argument("query", help="Natural language search query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument(
        "--embedder",
        default="stub",
        choices=["stub", "sentence-transformers", "sentence-transformer", "st"],
    )
    search_parser.set_defaults(handler=handle_search)

    serve_parser = subparsers.add_parser("serve", help="Run the local search API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument(
        "--embedder",
        default="stub",
        choices=["stub", "sentence-transformers", "sentence-transformer", "st"],
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
    captioner = make_captioner(args.captioner)
    embedder = make_embedder(args.embedder)
    images = scan_images(args.roots)

    with connect(args.db) as conn:
        init_db(conn)
        indexed = 0
        skipped = 0
        for image in images:
            if not needs_indexing(conn, image, captioner.name, embedder.name):
                skipped += 1
                continue
            caption = captioner.caption(image.path)
            embedding = embedder.embed(caption)
            upsert_indexed_image(conn, image, caption, captioner.name, embedder.name, embedding)
            indexed += 1
            if indexed % 10 == 0:
                conn.commit()
                print(f"indexed {indexed} images...")
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


def handle_search(args: argparse.Namespace) -> int:
    embedder = make_embedder(args.embedder)
    with connect(args.db) as conn:
        init_db(conn)
        images = list_indexed_images(conn)

    results = search_images(args.query, images, embedder, args.limit)
    if not results:
        print("no results")
        return 0

    for result in results:
        print(f"{result.score:.3f}  {result.image.path}")
        print(f"       {result.image.caption}")
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    embedder = make_embedder(args.embedder)
    from local_image_search.server import run_server

    run_server(args.db, embedder, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

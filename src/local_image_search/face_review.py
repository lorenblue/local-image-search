from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from local_image_search.db import connect_readonly
from local_image_search.models import FaceBox, ImageFile


@dataclass(frozen=True)
class FaceReviewItem:
    image: ImageFile
    width: int
    height: int
    faces: list[FaceBox]


def write_face_review(
    db_path: Path,
    roots: list[Path],
    output_path: Path,
    limit: int,
) -> Path:
    if limit <= 0:
        raise ValueError("Limit must be greater than zero")

    with connect_readonly(db_path) as conn:
        items, detection_models = _load_review_items(conn, roots, limit)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _render_html(items, _format_detection_models(detection_models))
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _load_review_items(
    conn,
    roots: list[Path],
    limit: int,
) -> tuple[list[FaceReviewItem], set[str]]:
    scopes = [_review_scope(root) for root in roots]
    rows = conn.execute(
        """
        SELECT images.id AS image_id,
               images.path,
               images.file_name,
               images.file_size,
               images.created_at,
               images.modified_at,
               faces.id AS face_id,
               faces.x,
               faces.y,
               faces.width,
               faces.height,
               faces.detection_score,
               faces.detection_model
        FROM faces
        JOIN images ON images.id = faces.image_id
        ORDER BY images.path, faces.id
        """
    ).fetchall()
    items: list[FaceReviewItem] = []
    detection_models: set[str] = set()
    current_image_id: int | None = None
    current_image: ImageFile | None = None
    current_faces: list[FaceBox] = []
    current_models: set[str] = set()

    def flush_current() -> None:
        nonlocal current_image, current_faces, current_models
        if current_image is None or len(items) >= limit:
            return
        if not _path_matches_scopes(current_image.path, scopes):
            return
        try:
            width, height = _image_size(current_image.path)
        except (OSError, UnidentifiedImageError):
            return
        items.append(
            FaceReviewItem(
                image=current_image,
                width=width,
                height=height,
                faces=list(current_faces),
            )
        )
        detection_models.update(current_models)

    for row in rows:
        image_id = int(row["image_id"])
        if current_image_id is not None and image_id != current_image_id:
            flush_current()
            if len(items) >= limit:
                break
            current_faces = []
            current_models = set()
        current_image_id = image_id
        current_image = ImageFile(
            path=Path(row["path"]),
            file_name=str(row["file_name"]),
            file_size=int(row["file_size"]),
            created_at=row["created_at"],
            modified_at=float(row["modified_at"]),
        )
        current_faces.append(
            FaceBox(
                x=float(row["x"]),
                y=float(row["y"]),
                width=float(row["width"]),
                height=float(row["height"]),
                detection_score=(
                    None if row["detection_score"] is None else float(row["detection_score"])
                ),
                id=int(row["face_id"]),
            )
        )
        if row["detection_model"]:
            current_models.add(str(row["detection_model"]))

    if len(items) < limit:
        flush_current()

    return items, detection_models


def _format_detection_models(detection_models: set[str]) -> str:
    if not detection_models:
        return "stored faces"
    return ", ".join(sorted(detection_models))


def _review_scope(root: Path) -> tuple[Path, bool]:
    normalized = root.expanduser().resolve(strict=False)
    return normalized, root.is_file()


def _path_matches_scopes(path: Path, scopes: list[tuple[Path, bool]]) -> bool:
    if not scopes:
        return True
    normalized = path.expanduser().resolve(strict=False)
    return any(
        normalized == root if exact_match else normalized == root or normalized.is_relative_to(root)
        for root, exact_match in scopes
    )


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        pass
    else:
        register_heif_opener()

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def _render_html(items: list[FaceReviewItem], detection_model_label: str) -> str:
    cards = "\n".join(_render_item(item) for item in items)
    total_faces = sum(len(item.faces) for item in items)
    summary = f"{len(items)} images · {total_faces} stored faces · {_escape(detection_model_label)}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Review</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.35;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: Canvas;
      color: CanvasText;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .summary {{
      margin-bottom: 20px;
      opacity: 0.72;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
    }}
    article {{
      border: 1px solid color-mix(in srgb, CanvasText 18%, transparent);
      border-radius: 8px;
      overflow: hidden;
      background: color-mix(in srgb, CanvasText 3%, Canvas);
    }}
    .preview {{
      position: relative;
      background: color-mix(in srgb, CanvasText 8%, Canvas);
    }}
    .preview img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .face {{
      position: absolute;
      border: 2px solid #ff3b30;
      box-sizing: border-box;
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
    }}
    .face-id {{
      position: absolute;
      left: 0;
      top: 0;
      padding: 2px 5px;
      border-radius: 4px;
      background: #ff3b30;
      color: white;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .meta {{
      padding: 10px 12px;
      font-size: 13px;
    }}
    .file {{
      margin-bottom: 4px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .path {{
      opacity: 0.65;
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <h1>Face Review</h1>
  <div class="summary">{summary}</div>
  <main class="grid">
    {cards}
  </main>
</body>
</html>
"""


def _render_item(item: FaceReviewItem) -> str:
    boxes = "\n".join(_render_face_box(face, item.width, item.height) for face in item.faces)
    face_label = "face" if len(item.faces) == 1 else "faces"
    return f"""<article>
  <div class="preview">
    <img src="{_path_uri(item.image.path)}" alt="">
    {boxes}
  </div>
  <div class="meta">
    <div class="file">{_escape(item.image.file_name)} · {len(item.faces)} {face_label}</div>
    <div class="path">{_escape(str(item.image.path))}</div>
  </div>
</article>"""


def _render_face_box(face: FaceBox, image_width: int, image_height: int) -> str:
    left = face.x / image_width * 100
    top = face.y / image_height * 100
    width = face.width / image_width * 100
    height = face.height / image_height * 100
    label = f'<span class="face-id">#{face.id}</span>' if face.id is not None else ""
    return (
        '<div class="face" '
        f'style="left:{left:.3f}%;top:{top:.3f}%;'
        f'width:{width:.3f}%;height:{height:.3f}%">{label}</div>'
    )


def _path_uri(path: Path) -> str:
    return html.escape(path.expanduser().resolve().as_uri(), quote=True)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from local_image_search.face_detection import FaceDetector
from local_image_search.models import FaceBox, ImageFile
from local_image_search.scanner import scan_images


@dataclass(frozen=True)
class FaceReviewItem:
    image: ImageFile
    width: int
    height: int
    faces: list[FaceBox]


def write_face_review(
    roots: list[Path],
    detector: FaceDetector,
    output_path: Path,
    limit: int,
) -> Path:
    if limit <= 0:
        raise ValueError("Limit must be greater than zero")

    items = []
    for image in scan_images(roots)[:limit]:
        item = _review_image(image, detector)
        if item is not None:
            items.append(item)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(items, detector.name), encoding="utf-8")
    return output_path


def _review_image(image: ImageFile, detector: FaceDetector) -> FaceReviewItem | None:
    try:
        width, height = _image_size(image.path)
    except (OSError, UnidentifiedImageError):
        return None
    return FaceReviewItem(
        image=image,
        width=width,
        height=height,
        faces=detector.detect_faces(image.path),
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


def _render_html(items: list[FaceReviewItem], detector_name: str) -> str:
    cards = "\n".join(_render_item(item) for item in items)
    total_faces = sum(len(item.faces) for item in items)
    summary = f"{len(items)} images · {total_faces} detected faces · {_escape(detector_name)}"
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
    return (
        '<div class="face" '
        f'style="left:{left:.3f}%;top:{top:.3f}%;'
        f'width:{width:.3f}%;height:{height:.3f}%"></div>'
    )


def _path_uri(path: Path) -> str:
    return html.escape(path.expanduser().resolve().as_uri(), quote=True)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)

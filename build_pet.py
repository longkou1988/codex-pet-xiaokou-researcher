from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "generated" / "research-pose-sheet-chroma.png"
POSE_DIR = ROOT / "final" / "poses"
FINAL_DIR = ROOT / "final"
PACKAGE_DIR = ROOT / "package" / "xiaokou-researcher"
QA_DIR = ROOT / "qa"

CELL_W, CELL_H = 192, 208
COLS, ROWS = 8, 11
FRAME_COUNTS = [6, 8, 8, 4, 5, 8, 6, 6, 6, 8, 8]
STATE_NAMES = [
    "idle",
    "running-right",
    "running-left",
    "waving",
    "jumping",
    "failed",
    "waiting",
    "running",
    "review",
    "look-row-9",
    "look-row-10",
]


def remove_green(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, _ = pixels[x, y]
            dominance = g - max(r, b)
            if g > 120 and dominance > 35:
                alpha = 0
            elif g > 85 and dominance > 18:
                alpha = max(0, min(255, int(255 * (35 - dominance) / 17)))
            else:
                alpha = 255
            if alpha == 0:
                pixels[x, y] = (0, 0, 0, 0)
            else:
                # Suppress residual green without touching the blue board or orange badge.
                corrected_g = min(g, max(r, b) + 12)
                pixels[x, y] = (r, corrected_g, b, alpha)
    return rgba


def extract_poses(sheet: Image.Image) -> list[Image.Image]:
    poses: list[Image.Image] = []
    for index in range(11):
        col, row = index % 4, index // 4
        left = round(col * sheet.width / 4)
        right = round((col + 1) * sheet.width / 4)
        top = round(row * sheet.height / 3)
        bottom = round((row + 1) * sheet.height / 3)
        tile = sheet.crop((left, top, right, bottom))
        alpha = tile.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            raise RuntimeError(f"Pose {index + 1} is empty")
        pose = tile.crop(bbox)
        pose.save(POSE_DIR / f"pose-{index + 1:02d}.png")
        poses.append(pose)
    return poses


def place_pose(
    pose: Image.Image,
    *,
    max_w: int = 164,
    max_h: int = 184,
    scale: float = 1.0,
    dx: int = 0,
    dy: int = 0,
) -> Image.Image:
    factor = min(max_w / pose.width, max_h / pose.height) * scale
    size = (max(1, round(pose.width * factor)), max(1, round(pose.height * factor)))
    resized = pose.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
    x = (CELL_W - resized.width) // 2 + dx
    y = CELL_H - 10 - resized.height + dy
    canvas.alpha_composite(resized, (x, y))
    return canvas


def make_frames(poses: list[Image.Image], row: int) -> list[Image.Image]:
    count = FRAME_COUNTS[row]
    pose = poses[row]
    if row == 0:
        offsets = [(0, 0, 1.00), (0, -1, 1.00), (0, -2, 1.01), (0, -2, 1.01), (0, -1, 1.00), (0, 0, 1.00)]
    elif row in (1, 2):
        offsets = [(0, 0, 1.00), (2, -3, 1.01), (4, 0, 1.00), (2, 2, 0.99), (0, 0, 1.00), (-2, -3, 1.01), (-4, 0, 1.00), (-2, 2, 0.99)]
    elif row == 3:
        offsets = [(0, 0, 1.00), (0, -2, 1.01), (1, 0, 1.00), (0, -1, 1.01)]
    elif row == 4:
        offsets = [(0, 4, 0.98), (0, -6, 1.00), (0, -14, 1.02), (0, -6, 1.00), (0, 4, 0.98)]
    elif row == 5:
        offsets = [(0, 0, 1.00), (-2, 0, 1.00), (2, 0, 1.00), (-1, 1, 0.99), (1, 1, 0.99), (0, 0, 1.00), (-1, 0, 1.00), (1, 0, 1.00)]
    elif row in (6, 7, 8):
        offsets = [(0, 0, 1.00), (0, -1, 1.00), (0, -2, 1.01), (0, -1, 1.01), (0, 0, 1.00), (0, 1, 0.99)]
    elif row == 9:
        # Two generated gaze masters are combined into an eight-direction row.
        gaze_poses = [poses[9]] * 4 + [poses[10]] * 4
        offsets = [(-3, -3, 1.00), (-2, -2, 1.00), (-1, 0, 1.00), (0, 2, 1.00), (0, 2, 1.00), (1, 0, 1.00), (2, -2, 1.00), (3, -3, 1.00)]
        return [place_pose(p, dx=dx, dy=dy, scale=sc) for p, (dx, dy, sc) in zip(gaze_poses, offsets)]
    else:
        gaze_poses = [poses[10]] * 4 + [poses[9]] * 4
        offsets = [(3, 3, 1.00), (2, 2, 1.00), (1, 0, 1.00), (0, -2, 1.00), (0, -2, 1.00), (-1, 0, 1.00), (-2, 2, 1.00), (-3, 3, 1.00)]
        return [place_pose(p, dx=dx, dy=dy, scale=sc) for p, (dx, dy, sc) in zip(gaze_poses, offsets)]
    return [place_pose(pose, dx=dx, dy=dy, scale=sc) for dx, dy, sc in offsets[:count]]


def checker_preview(atlas: Image.Image) -> Image.Image:
    bg = Image.new("RGB", atlas.size, "#f4f4f4")
    draw = ImageDraw.Draw(bg)
    block = 24
    for y in range(0, bg.height, block):
        for x in range(0, bg.width, block):
            if (x // block + y // block) % 2:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill="#dddddd")
    bg.paste(atlas, mask=atlas.getchannel("A"))
    return bg.resize((768, 1144), Image.Resampling.LANCZOS)


def main() -> None:
    for directory in (POSE_DIR, FINAL_DIR, PACKAGE_DIR, QA_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    keyed = remove_green(Image.open(SOURCE))
    keyed.save(FINAL_DIR / "research-pose-sheet-transparent.png")
    poses = extract_poses(keyed)

    atlas = Image.new("RGBA", (COLS * CELL_W, ROWS * CELL_H), (0, 0, 0, 0))
    row_frames: list[list[Image.Image]] = []
    for row in range(ROWS):
        frames = make_frames(poses, row)
        row_frames.append(frames)
        for col, frame in enumerate(frames):
            atlas.alpha_composite(frame, (col * CELL_W, row * CELL_H))

    # Fully transparent pixels must not retain hidden RGB values.
    zero = Image.new("RGBA", atlas.size, (0, 0, 0, 0))
    atlas = Image.composite(atlas, zero, atlas.getchannel("A"))
    atlas.save(FINAL_DIR / "spritesheet.png")
    atlas.save(FINAL_DIR / "spritesheet.webp", "WEBP", lossless=True, method=6)
    atlas.save(PACKAGE_DIR / "spritesheet.webp", "WEBP", lossless=True, method=6)

    avatar = row_frames[0][0].resize((64, 64), Image.Resampling.LANCZOS)
    avatar.save(FINAL_DIR / "avatar-64.png")
    checker_preview(atlas).save(QA_DIR / "spritesheet-preview.png")

    manifest = {
        "id": "xiaokou-researcher",
        "displayName": "小扣研究员",
        "description": "戴黑框眼镜和橙色纽扣，善于核验资料、连接知识并讲清复杂问题的 Codex 原生科研桌宠。",
        "spriteVersionNumber": 2,
        "spritesheetPath": "spritesheet.webp",
    }
    (PACKAGE_DIR / "pet.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    validation = {
        "status": "passed",
        "atlas": {"width": atlas.width, "height": atlas.height, "mode": atlas.mode, "columns": COLS, "rows": ROWS, "cell": [CELL_W, CELL_H]},
        "states": [{"row": row, "state": STATE_NAMES[row], "frames": FRAME_COUNTS[row]} for row in range(ROWS)],
        "source_pose_count": len(poses),
        "transparent_corners": all(atlas.getpixel(point)[3] == 0 for point in [(0, 0), (atlas.width - 1, 0), (0, atlas.height - 1), (atlas.width - 1, atlas.height - 1)]),
        "unused_cells_transparent": all(
            atlas.crop((col * CELL_W, row * CELL_H, (col + 1) * CELL_W, (row + 1) * CELL_H)).getchannel("A").getbbox() is None
            for row, count in enumerate(FRAME_COUNTS)
            for col in range(count, COLS)
        ),
    }
    if not validation["transparent_corners"] or not validation["unused_cells_transparent"]:
        validation["status"] = "failed"
        raise RuntimeError(json.dumps(validation, ensure_ascii=False))
    (QA_DIR / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

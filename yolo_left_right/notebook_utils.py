from __future__ import annotations

from pathlib import Path
import random
import shutil
import re

import cv2
import yaml


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid"),
    "test": ("test",),
}


def find_data_yaml(export_root: Path) -> Path:
    yaml_candidates = sorted(export_root.rglob("data.yaml"))
    if not yaml_candidates:
        raise FileNotFoundError(f"No data.yaml found under {export_root}")
    return yaml_candidates[0]


def _resolve_candidate_paths(dataset_root: Path, raw_value: str) -> list[Path]:
    raw_path = Path(raw_value)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((dataset_root / raw_path).resolve())
        candidates.append((dataset_root / raw_path.as_posix().lstrip("./")).resolve())

        parts = [part for part in raw_path.parts if part not in ("..", ".")]
        if parts:
            candidates.append((dataset_root / Path(*parts)).resolve())

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _find_existing_image_dir(dataset_root: Path, data_yaml: dict, split_name: str) -> Path | None:
    raw_value = data_yaml.get(split_name)
    if raw_value:
        for candidate in _resolve_candidate_paths(dataset_root, str(raw_value)):
            if candidate.exists():
                return candidate

    for alias in SPLIT_ALIASES[split_name]:
        for candidate in (
            dataset_root / alias / "images",
            dataset_root / alias,
            dataset_root / "images" / alias,
        ):
            if candidate.exists():
                return candidate.resolve()
    return None


def image_dir_to_label_dir(image_dir: Path) -> Path:
    splits = {"train", "valid", "val", "test"}

    if image_dir.name in splits and image_dir.parent.name == "images":
        return image_dir.parent.parent / "labels" / image_dir.name

    if image_dir.name == "images" and image_dir.parent.name in splits:
        return image_dir.parent / "labels"

    return image_dir


def _collect_image_label_pairs(image_dir: Path) -> list[tuple[Path, Path]]:
    label_dir = image_dir_to_label_dir(image_dir)
    pairs = []
    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        pairs.append((image_path, label_path))
    return pairs


def _split_train_only_pairs(
    pairs: list[tuple[Path, Path]], seed: int = 42
) -> dict[str, list[tuple[Path, Path]]]:
    if len(pairs) < 3:
        raise ValueError("Need at least 3 labeled images to create train/val/test splits automatically.")

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    val_count = max(1, round(total * 0.1))
    test_count = max(1, round(total * 0.1))
    train_count = total - val_count - test_count

    if train_count < 1:
        train_count = max(1, total - 2)
        val_count = 1
        test_count = total - train_count - val_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def _copy_pairs_to_dataset(
    split_to_pairs: dict[str, list[tuple[Path, Path]]], dataset_root: Path
) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    for split_name, pairs in split_to_pairs.items():
        image_out_dir = dataset_root / "images" / split_name
        label_out_dir = dataset_root / "labels" / split_name
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in pairs:
            shutil.copy2(image_path, image_out_dir / image_path.name)
            label_out_path = label_out_dir / label_path.name
            label_rows = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                normalized = _normalize_label_line(line)
                if normalized is not None:
                    label_rows.append(normalized)
            label_out_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")


def _normalize_label_line(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    coords = [float(value) for value in parts[1:]]

    if len(coords) == 4:
        cx, cy, w, h = coords
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    if len(coords) % 2 != 0:
        return None

    xs = coords[0::2]
    ys = coords[1::2]
    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(1.0, max(xs))
    y_max = min(1.0, max(ys))
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prepare_local_dataset(
    export_root: Path, dataset_root: Path, data_yaml_out: Path, seed: int = 42
) -> dict:
    source_yaml_path = find_data_yaml(export_root)
    with open(source_yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)

    dataset_yaml_root = source_yaml_path.parent.resolve()
    names_raw = data_yaml["names"]
    if isinstance(names_raw, dict):
        id_to_name = {int(k): v for k, v in names_raw.items()}
    else:
        id_to_name = {i: v for i, v in enumerate(names_raw)}

    source_split_dirs = {
        split_name: _find_existing_image_dir(dataset_yaml_root, data_yaml, split_name)
        for split_name in ("train", "val", "test")
    }

    split_to_pairs: dict[str, list[tuple[Path, Path]]] = {}
    for split_name, image_dir in source_split_dirs.items():
        if image_dir is None:
            continue
        pairs = _collect_image_label_pairs(image_dir)
        if pairs:
            split_to_pairs[split_name] = pairs

    if "train" not in split_to_pairs:
        raise FileNotFoundError("No labeled training images found in roboflow_export.")

    used_auto_split = not split_to_pairs.get("val") or not split_to_pairs.get("test")
    if used_auto_split:
        split_to_pairs = _split_train_only_pairs(split_to_pairs["train"], seed=seed)

    _copy_pairs_to_dataset(split_to_pairs, dataset_root)

    prepared_yaml = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": id_to_name,
    }
    with open(data_yaml_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(prepared_yaml, f, sort_keys=False, allow_unicode=True)

    return {
        "source_yaml_path": source_yaml_path.resolve(),
        "data_yaml_path": data_yaml_out.resolve(),
        "dataset_root": dataset_root.resolve(),
        "train_img_dir": (dataset_root / "images" / "train").resolve(),
        "val_img_dir": (dataset_root / "images" / "val").resolve(),
        "test_img_dir": (dataset_root / "images" / "test").resolve(),
        "id_to_name": id_to_name,
        "source_split_dirs": source_split_dirs,
        "used_auto_split": used_auto_split,
        "split_counts": {split_name: len(pairs) for split_name, pairs in split_to_pairs.items()},
    }


def resolve_run_dir(runs_root: Path, run_name: str) -> Path:
    direct = runs_root / run_name
    if (direct / "weights" / "best.pt").exists():
        return direct

    pattern = re.compile(rf"^{re.escape(run_name)}(?:-(\d+))?$")
    candidates = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        best_pt = path / "weights" / "best.pt"
        if best_pt.exists():
            suffix = int(match.group(1)) if match.group(1) is not None else 0
            candidates.append((suffix, best_pt.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(f"No run directory with best.pt found for '{run_name}' under {runs_root}")

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def read_yolo_labels(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not txt_path.exists():
        return rows

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, cx, cy, w, h = parts
        rows.append((int(class_id), float(cx), float(cy), float(w), float(h)))
    return rows


def draw_yolo_boxes(image_path: Path, label_path: Path, id_to_name: dict[int, str]):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image_h, image_w = image_bgr.shape[:2]
    for class_id, cx, cy, w, h in read_yolo_labels(label_path):
        x1 = int((cx - w / 2) * image_w)
        y1 = int((cy - h / 2) * image_h)
        x2 = int((cx + w / 2) * image_w)
        y2 = int((cy + h / 2) * image_h)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            id_to_name.get(class_id, str(class_id)),
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig


class ImagePreprocessor:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.raw_dir = Path(self.cfg.data_dir)
        self.out_all = Path(self.cfg.output_dir) / "all"
        self.harmful_root = self.raw_dir / "Harmful_real"
        self.manifest_path = Path(self.cfg.output_dir) / "manifest.json"
        self.manifest: Dict[str, Dict] = {}
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                self.manifest = {}

    def scan(self) -> List[Tuple[Path, str]]:
        if not self.raw_dir.exists():
            raise FileNotFoundError(str(self.raw_dir))
        excluded: set[str] = set()
        if self.harmful_root.exists():
            for d in self.harmful_root.iterdir():
                if d.is_dir():
                    cnt = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in self.allowed_ext)
                    if cnt < self.cfg.min_images_harmful:
                        excluded.add(d.name)
            if excluded:
                self.logger.info(f"Excluding harmful classes with <{self.cfg.min_images_harmful} images: {sorted(list(excluded))}")
        items: List[Tuple[Path, str]] = []
        for p in self.raw_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.allowed_ext:
                label = p.parent.name
                if self.harmful_root in p.parents and label in excluded:
                    continue
                items.append((p, label))
        self.logger.info(f"Scanned {len(items)} images from {self.raw_dir}")
        return items

    def _checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _open_image(self, path: Path) -> Optional[Image.Image]:
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                return im
        except Exception as e:
            self.logger.warning(f"Failed to open {path}: {e}")
            return None

    def process_and_save(self, items: List[Tuple[Path, str]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        bar = tqdm(items, desc="Preprocessing", unit="img")
        for src_path, label in bar:
            cs = self._checksum(src_path)
            key = str(src_path)
            entry = self.manifest.get(key)
            if entry and entry.get("checksum") == cs:
                dst_path = Path(entry.get("dst", ""))
                if dst_path.exists():
                    counts[label] = counts.get(label, 0) + 1
                    continue
            im = self._open_image(src_path)
            if im is None:
                continue
            im = im.resize(self.cfg.image_size, Image.Resampling.BICUBIC)
            label_dir = self.out_all / label
            label_dir.mkdir(parents=True, exist_ok=True)
            name = f"{src_path.stem}_{uuid.uuid4().hex[:8]}.jpg"
            dst = label_dir / name
            try:
                im.save(dst, format="JPEG", quality=90)
                counts[label] = counts.get(label, 0) + 1
                self.manifest[key] = {"checksum": cs, "dst": str(dst), "label": label}
            except Exception as e:
                self.logger.warning(f"Failed to save {dst}: {e}")
        self.logger.info(f"Processed images saved to {self.out_all}")
        try:
            self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        except Exception:
            pass
        return counts

    def analyze_conditions(self, items: List[Tuple[Path, str]]) -> Dict[str, int]:
        stats = {"dark": 0, "bright": 0, "low_contrast": 0}
        for src_path, _ in items:
            im = self._open_image(src_path)
            if im is None:
                continue
            gray = ImageOps.grayscale(im)
            arr = np.asarray(gray, dtype=np.float32)
            mean = float(arr.mean())
            std = float(arr.std())
            if mean < 40:
                stats["dark"] += 1
            if mean > 215:
                stats["bright"] += 1
            if std < 15:
                stats["low_contrast"] += 1
        self.logger.info(f"Robustness check: {stats}")
        return stats
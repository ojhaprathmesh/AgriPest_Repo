import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from .config import PipelineConfig


class DatasetSplitter:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.base = Path(self.cfg.output_dir)
        self.all_dir = self.base / "all"
        self.train_dir = self.base / "train"
        self.test_dir = self.base / "test"

    def _index_all(self) -> Tuple[List[Path], List[str]]:
        X: List[Path] = []
        y: List[str] = []
        if not self.all_dir.exists():
            raise FileNotFoundError(str(self.all_dir))
        for label_dir in sorted(self.all_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img in label_dir.glob("*.jpg"):
                X.append(img)
                y.append(label)
        return X, y

    def split(self) -> Tuple[List[Path], List[Path], List[str], List[str]]:
        from sklearn.model_selection import train_test_split
        X, y = self._index_all()
        if not X:
            raise ValueError("No processed images found")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.cfg.train_ratio,
            stratify=y,
            random_state=self.cfg.seed,
        )
        self.logger.info(
            f"Split {len(X)} images into train={len(X_train)} test={len(X_test)}"
        )
        dist_train: Dict[str, int] = {}
        dist_test: Dict[str, int] = {}
        for c in y_train:
            dist_train[c] = dist_train.get(c, 0) + 1
        for c in y_test:
            dist_test[c] = dist_test.get(c, 0) + 1
        self.logger.info(f"Train distribution: {json.dumps(dist_train)}")
        self.logger.info(f"Test distribution: {json.dumps(dist_test)}")
        return X_train, X_test, y_train, y_test

    def materialize(self, X_train: List[Path], y_train: List[str], X_test: List[Path], y_test: List[str]) -> None:
        for d in [self.train_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)
        bar_tr = tqdm(list(zip(X_train, y_train)), desc="Copy train", unit="img")
        for path, label in bar_tr:
            dst = self.train_dir / label
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst / path.name)
        bar_te = tqdm(list(zip(X_test, y_test)), desc="Copy test", unit="img")
        for path, label in bar_te:
            dst = self.test_dir / label
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst / path.name)
        self.logger.info(f"Materialized split to {self.train_dir} and {self.test_dir}")
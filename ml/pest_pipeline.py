import csv
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Tuple
import uuid

from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


@dataclass
class PipelineConfig:
    data_dir: str
    output_dir: str
    image_size: Tuple[int, int] = (256, 256)
    train_ratio: float = 0.7
    seed: int = 42
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    resnet_variant: str = "resnet50v2"
    freeze_backbone: bool = True
    early_stopping_patience: int = 5
    model_dir: str = "models"
    plots_dir: str = "plots"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    use_dataset_cache: bool = True
    cache_dir: str = "processed/cache"
    use_mixed_precision: bool = True
    framework: str = "tf"
    num_workers: int = 4
    amp_dtype: str = "bf16"
    min_images_harmful: int = 75


class ConfigLoader:
    def __init__(self, config_path: Path):
        self.config_path = config_path

    def load(self) -> PipelineConfig:
        import yaml
        if not self.config_path.exists():
            raise FileNotFoundError(str(self.config_path))
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return PipelineConfig(**cfg)


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


class ModelBuilder:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def build(self, num_classes: int):
        import tensorflow as tf
        try:
            import tensorflow_addons as tfa
        except Exception:
            tfa = None
        if self.cfg.resnet_variant.lower() == "resnet50v2":
            backbone = tf.keras.applications.ResNet50V2(
                include_top=False,
                weights="imagenet",
                input_shape=(self.cfg.image_size[0], self.cfg.image_size[1], 3),
                pooling="avg",
            )
        else:
            backbone = tf.keras.applications.ResNet50(
                include_top=False,
                weights="imagenet",
                input_shape=(self.cfg.image_size[0], self.cfg.image_size[1], 3),
                pooling="avg",
            )
        if self.cfg.freeze_backbone:
            for l in backbone.layers:
                l.trainable = False
        inputs = tf.keras.Input(shape=(self.cfg.image_size[0], self.cfg.image_size[1], 3))
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
        ])
        x = aug(x)
        x = backbone(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        metrics = ["accuracy"]
        if tfa is not None:
            metrics.append(tfa.metrics.F1Score(num_classes=num_classes, average="macro"))
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=metrics)
        return model


class TrainerEvaluator:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.base = Path(self.cfg.output_dir)
        self.train_dir = self.base / "train"
        self.test_dir = self.base / "test"
        self.model_dir = Path(self.cfg.model_dir)
        self.plots_dir = Path(self.cfg.plots_dir)
        self.reports_dir = Path(self.cfg.reports_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(self.cfg.cache_dir)
        
        if self.cfg.use_dataset_cache:
            (self.cache_dir / "train").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "test").mkdir(parents=True, exist_ok=True)
        self.model_version = uuid.uuid4().hex

        from datetime import datetime, timezone
        self.run_timestamp = datetime.now(timezone.utc).isoformat()
        self.curve_path = self.reports_dir / "learning_curve.json"
        self.curve_html = self.plots_dir / "learning_curve.html"
        if not self.curve_path.exists():
            try:
                with open(self.curve_path, "w", encoding="utf-8") as f:
                    json.dump({"model_version": self.model_version, "timestamp": self.run_timestamp, "epochs": []}, f, indent=2)
            except Exception:
                pass

    def _datasets(self):
        import tensorflow as tf
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            image_size=self.cfg.image_size,
            batch_size=self.cfg.batch_size,
            label_mode="categorical",
            shuffle=True,
            seed=self.cfg.seed,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_dir,
            image_size=self.cfg.image_size,
            batch_size=self.cfg.batch_size,
            label_mode="categorical",
            shuffle=False,
        )
        class_names = train_ds.class_names
        autotune = tf.data.AUTOTUNE
        if self.cfg.use_dataset_cache:
            train_ds = train_ds.cache(str(self.cache_dir / "train"))
            val_ds = val_ds.cache(str(self.cache_dir / "test"))
        train_ds = train_ds.prefetch(autotune)
        val_ds = val_ds.prefetch(autotune)
        with open(self.reports_dir / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)
        return train_ds, val_ds, class_names

    def train(self) -> Dict:
        import tensorflow as tf
        if self.cfg.use_mixed_precision:
            try:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    from tensorflow.keras import mixed_precision
                    mixed_precision.set_global_policy("mixed_float16")
                    details = tf.config.experimental.get_device_details(gpus[0])
                    name = details.get("device_name", "GPU")
                    self.logger.info(f"Mixed precision enabled on {name}")
                    if "4050" in name or "RTX 4050" in name:
                        self.logger.info("Detected RTX 4050; using this GPU")
                else:
                    self.logger.info("No GPU found; mixed precision not enabled")
            except Exception:
                pass
        train_ds, val_ds, class_names = self._datasets()
        mb = ModelBuilder(self.cfg, self.logger)
        model = mb.build(num_classes=len(class_names))
        class CLIStatusCallback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.model.logger.info("Training started") if hasattr(self.model, "logger") else None
            def on_epoch_begin(self, epoch, logs=None):
                try:
                    total = getattr(self.model, "_total_epochs", None)
                    msg = f"Epoch {int(epoch+1)}/{int(total)} started" if total is not None else f"Epoch {int(epoch+1)} started"
                    self.model._status_logger.info(msg) if hasattr(self.model, "_status_logger") else None
                except Exception:
                    pass
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                msg = {
                    "epoch": int(epoch + 1),
                    "loss": float(logs.get("loss", 0.0)),
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "accuracy": float(logs.get("accuracy", 0.0)),
                    "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                }
                self.model._status_logger.info(json.dumps(msg)) if hasattr(self.model, "_status_logger") else None
                try:
                    data = json.loads(self.model._curve_path.read_text(encoding="utf-8")) if self.model._curve_path.exists() else {"model_version": self.model._model_version, "timestamp": self.model._run_timestamp, "epochs": []}
                    data.setdefault("epochs", []).append({
                        "epoch": int(epoch + 1),
                        "train_loss": float(logs.get("loss", 0.0)),
                        "val_loss": float(logs.get("val_loss", 0.0)),
                        "train_accuracy": float(logs.get("accuracy", 0.0)),
                        "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                    })
                    with open(self.model._curve_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    self.model._update_plotly_live_tf()
                except Exception:
                    pass
        model._status_logger = self.logger
        model._total_epochs = self.cfg.epochs
        model._curve_path = self.curve_path
        model._curve_html = self.curve_html
        model._model_version = self.model_version
        model._run_timestamp = self.run_timestamp
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.cfg.early_stopping_patience, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=str(self.model_dir / "best_model.h5"), monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            CLIStatusCallback(),
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=self.cfg.epochs, callbacks=callbacks)
        model.save(self.model_dir / "final_model.h5")
        hist = {k: [float(v) for v in history.history.get(k, [])] for k in history.history.keys()}
        with open(self.reports_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)
        self._write_history_csv(hist)
        self._plot_history(hist)
        return {"class_names": class_names, "history": hist}

    def _plot_history(self, hist: Dict):
        import matplotlib.pyplot as plt
        acc = hist.get("accuracy")
        val_acc = hist.get("val_accuracy")
        loss = hist.get("loss")
        val_loss = hist.get("val_loss")
        if acc and val_acc:
            plt.figure(figsize=(7, 5))
            plt.plot(acc, label="train")
            plt.plot(val_acc, label="val")
            plt.legend()
            plt.title("Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "accuracy.png")
            plt.close()
        if loss and val_loss:
            plt.figure(figsize=(7, 5))
            plt.plot(loss, label="train")
            plt.plot(val_loss, label="val")
            plt.legend()
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "loss.png")
            plt.close()

    def _write_history_csv(self, hist: Dict):
        keys = ["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
        rows = []
        length = max(len(hist.get("accuracy", [])), len(hist.get("loss", [])), len(hist.get("val_accuracy", [])), len(hist.get("val_loss", [])))
        for i in range(length):
            rows.append({
                "epoch": i + 1,
                "accuracy": float(hist.get("accuracy", [None] * length)[i]) if i < len(hist.get("accuracy", [])) else None,
                "val_accuracy": float(hist.get("val_accuracy", [None] * length)[i]) if i < len(hist.get("val_accuracy", [])) else None,
                "loss": float(hist.get("loss", [None] * length)[i]) if i < len(hist.get("loss", [])) else None,
                "val_loss": float(hist.get("val_loss", [None] * length)[i]) if i < len(hist.get("val_loss", [])) else None,
            })
        out = self.reports_dir / "training_history.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def evaluate(self) -> Dict:
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        model_path = self.model_dir / "best_model.h5"
        if not model_path.exists():
            model_path = self.model_dir / "final_model.h5"
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))
        model = tf.keras.models.load_model(model_path, compile=False)
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_dir,
            image_size=self.cfg.image_size,
            batch_size=self.cfg.batch_size,
            label_mode="categorical",
            shuffle=False,
        )
        class_names = test_ds.class_names
        y_true = []
        y_pred = []
        for batch_images, batch_labels in test_ds:
            preds = model.predict(batch_images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1).tolist())
            y_true.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        with open(self.reports_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        txt = classification_report(y_true, y_pred, target_names=class_names)
        with open(self.reports_dir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(txt)
        self._write_metrics_csv(report)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confusion_matrix.png")
        plt.close()
        return {"report": report, "confusion_matrix": cm.tolist()}

    def _write_metrics_csv(self, report: Dict):
        out = self.reports_dir / "classification_report.csv"
        rows = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                rows.append({
                    "label": label,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1-score": metrics.get("f1-score"),
                    "support": metrics.get("support"),
                })
        keys = ["label", "precision", "recall", "f1-score", "support"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def _update_plotly_live_tf(self):
        try:
            import plotly.graph_objs as go
            import plotly.offline as po
            if not self.curve_path.exists():
                return
            data = json.loads(self.curve_path.read_text(encoding="utf-8"))
            epochs = [e["epoch"] for e in data.get("epochs", [])]
            tl = [e["train_loss"] for e in data.get("epochs", [])]
            vl = [e["val_loss"] for e in data.get("epochs", [])]
            ta = [e["train_accuracy"] for e in data.get("epochs", [])]
            va = [e["val_accuracy"] for e in data.get("epochs", [])]
            traces = [
                go.Scatter(x=epochs, y=tl, mode="lines+markers", name="Train Loss", visible=True),
                go.Scatter(x=epochs, y=vl, mode="lines+markers", name="Val Loss", visible=True),
                go.Scatter(x=epochs, y=ta, mode="lines+markers", name="Train Acc", visible=False),
                go.Scatter(x=epochs, y=va, mode="lines+markers", name="Val Acc", visible=False),
            ]
            updatemenus = [
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Loss", method="update", args=[{"visible": [True, True, False, False]}]),
                        dict(label="Accuracy", method="update", args=[{"visible": [False, False, True, True]}]),
                    ],
                    x=0.0, y=1.15
                )
            ]
            layout = go.Layout(
                title="Learning Curves",
                xaxis=dict(title="Epoch", gridcolor="#ddd"),
                yaxis=dict(title="Metric", gridcolor="#ddd"),
                legend=dict(x=0.02, y=0.98),
                updatemenus=updatemenus,
                autosize=True,
            )
            fig = go.Figure(data=traces, layout=layout)
            po.plot(fig, filename=str(self.plots_dir / "learning_curve.html"), auto_open=False, include_plotlyjs="cdn")
        except Exception:
            pass


class Pipeline:
    def __init__(self, config_path: str):
        cfg = ConfigLoader(Path(config_path)).load()
        logger = setup_logger(Path(cfg.logs_dir), "pest_pipeline")
        self.cfg = cfg
        self.logger = logger
        self.pre = ImagePreprocessor(cfg, logger)
        self.splitter = DatasetSplitter(cfg, logger)
        self.trainer = TrainerEvaluator(cfg, logger)
        self.torch_trainer = TorchTrainerEvaluator(cfg, logger) if cfg.framework.lower() == "torch" else None

    def run_preprocessing(self) -> Dict[str, int]:
        items = self.pre.scan()
        stats = self.pre.analyze_conditions(items)
        counts = self.pre.process_and_save(items)
        self._write_preprocessing_doc(counts, stats)
        return counts

    def run_split(self) -> Tuple[List[Path], List[Path], List[str], List[str]]:
        X_train, X_test, y_train, y_test = self.splitter.split()
        self.splitter.materialize(X_train, y_train, X_test, y_test)
        return X_train, X_test, y_train, y_test

    def run_training(self) -> Dict:
        if self.cfg.framework.lower() == "torch" and self.torch_trainer is not None:
            return self.torch_trainer.train()
        return self.trainer.train()

    def run_evaluation(self) -> Dict:
        if self.cfg.framework.lower() == "torch" and self.torch_trainer is not None:
            return self.torch_trainer.evaluate()
        return self.trainer.evaluate()

    def _write_preprocessing_doc(self, counts: Dict[str, int], stats: Dict[str, int]) -> None:
        doc_dir = Path(self.cfg.reports_dir)
        doc_dir.mkdir(parents=True, exist_ok=True)
        total = sum(counts.values())
        lines = []
        lines.append("# Preprocessing Pipeline")
        lines.append("Input images are converted to RGB, resized to 256x256, and saved as JPEG.")
        lines.append("Normalization to 0-1 is applied during model input.")
        lines.append("Augmentation is applied to the training pipeline only.")
        lines.append("Stratified split maintains class balance with 70/30 train/test.")
        lines.append("Processed dataset layout under output_dir: all/, train/, test/.")
        lines.append("Class counts after preprocessing:")
        for k in sorted(counts.keys()):
            lines.append(f"- {k}: {counts[k]}")
        lines.append(f"Total: {total}")
        lines.append("Robustness checks:")
        lines.append(f"- Dark images: {stats.get('dark', 0)}")
        lines.append(f"- Bright images: {stats.get('bright', 0)}")
        lines.append(f"- Low contrast: {stats.get('low_contrast', 0)}")
        with open(doc_dir / "preprocessing.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        with open(doc_dir / "robustness.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)




class TorchTrainerEvaluator:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.base = Path(self.cfg.output_dir)
        self.train_dir = self.base / "train"
        self.test_dir = self.base / "test"
        self.model_dir = Path(self.cfg.model_dir)
        self.plots_dir = Path(self.cfg.plots_dir)
        self.reports_dir = Path(self.cfg.reports_dir)
        for d in [self.model_dir, self.plots_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(0)
            self.logger.info(f"Using GPU: {name}")
            if "4050" in name or "RTX 4050" in name:
                self.logger.info("Detected RTX 4050")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        else:
            self.logger.info("Using CPU")

    def _datasets(self):
        size = self.cfg.image_size[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = datasets.ImageFolder(str(self.train_dir), transform=train_tf)
        val_ds = datasets.ImageFolder(str(self.test_dir), transform=val_tf)
        class_names = train_ds.classes
        with open(self.reports_dir / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)
        nw = 0
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=nw, pin_memory=False, persistent_workers=False)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=nw, pin_memory=False, persistent_workers=False)
        return train_loader, val_loader, class_names

    def _build_model(self, num_classes: int):
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if self.cfg.freeze_backbone:
            for p in m.parameters():
                p.requires_grad = False
        in_feat = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        return m.to(self.device)

    def train(self) -> Dict:
        train_loader, val_loader, class_names = self._datasets()
        model = self._build_model(num_classes=len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.cfg.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = self.cfg.use_mixed_precision and self.device.type=="cuda"
        mp_dtype = torch.bfloat16 if self.cfg.amp_dtype.lower() == "bf16" else torch.float16
        use_scaler = use_amp and mp_dtype is torch.float16
        scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
        best_val = float('inf')
        patience = self.cfg.early_stopping_patience
        wait = 0
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        for epoch in range(self.cfg.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.cfg.epochs} started")
            try:
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                bar_tr = tqdm(train_loader, desc=f"Train {epoch+1}", unit="batch", leave=False)
                for imgs, labels in bar_tr:
                    imgs = imgs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        with torch.amp.autocast(device_type, enabled=use_amp, dtype=mp_dtype):
                            outputs = model(imgs)
                            loss = criterion(outputs, labels)
                        if use_scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                    except RuntimeError as e:
                        if "CUDA error" in str(e) or "illegal" in str(e).lower():
                            self.logger.info("Disabling AMP due to CUDA error; retrying step in FP32")
                            use_amp = False
                            use_scaler = False
                            outputs = model(imgs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                        else:
                            raise
                    running_loss += loss.item() * imgs.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    if total:
                        bar_tr.set_postfix(loss=round(running_loss/max(1,total),4), acc=round(correct/max(1,total),4))
                train_loss = running_loss / max(1, total)
                train_acc = correct / max(1, total)
                model.eval()
                val_running = 0.0
                v_correct = 0
                v_total = 0
                all_preds = []
                all_true = []
                with torch.no_grad():
                    bar_va = tqdm(val_loader, desc=f"Val {epoch+1}", unit="batch", leave=False)
                    for imgs, labels in bar_va:
                        imgs = imgs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        with torch.amp.autocast(device_type, enabled=use_amp, dtype=mp_dtype):
                            outputs = model(imgs)
                            v_loss = criterion(outputs, labels)
                        val_running += v_loss.item() * imgs.size(0)
                        preds = outputs.argmax(dim=1)
                        v_correct += (preds == labels).sum().item()
                        v_total += labels.size(0)
                        all_preds.extend(preds.cpu().tolist())
                        all_true.extend(labels.cpu().tolist())
                        if v_total:
                            bar_va.set_postfix(loss=round(val_running/max(1,v_total),4), acc=round(v_correct/max(1,v_total),4))
                val_loss = val_running / max(1, v_total)
                val_acc = v_correct / max(1, v_total)
                history["loss"].append(float(train_loss))
                history["accuracy"].append(float(train_acc))
                history["val_loss"].append(float(val_loss))
                history["val_accuracy"].append(float(val_acc))
                self.logger.info(json.dumps({"epoch": epoch+1, "loss": train_loss, "val_loss": val_loss, "accuracy": train_acc, "val_accuracy": val_acc}))
                self._append_epoch(epoch+1, train_loss, val_loss, train_acc, val_acc)
                self._update_plotly_live()
                scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                    torch.save(model.state_dict(), self.model_dir / "best_model_torch.pth")
                else:
                    wait += 1
                    if wait >= patience:
                        self.logger.info("Early stopping")
                        break
            except Exception as e:
                self.logger.error(f"Epoch {epoch+1} failed with error: {e}. Skipping epoch.")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self._append_epoch(epoch+1, float('nan'), float('nan'), float('nan'), float('nan'))
                self._update_plotly_live()
                continue
        torch.save(model.state_dict(), self.model_dir / "final_model_torch.pth")
        with open(self.reports_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        self._write_history_csv(history)
        self._plot_history(history)
        return {"class_names": class_names, "history": history}

    def _append_epoch(self, epoch: int, loss: float, val_loss: float, acc: float, val_acc: float):
        try:
            if self.curve_path.exists():
                data = json.loads(self.curve_path.read_text(encoding="utf-8"))
            else:
                data = {"model_version": self.model_version, "timestamp": self.run_timestamp, "epochs": []}
            data.setdefault("model_version", self.model_version)
            data.setdefault("timestamp", self.run_timestamp)
            data.setdefault("epochs", []).append({
                "epoch": int(epoch),
                "train_loss": float(loss),
                "val_loss": float(val_loss),
                "train_accuracy": float(acc),
                "val_accuracy": float(val_acc),
            })
            with open(self.curve_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _update_plotly_live(self):
        try:
            import plotly.graph_objs as go
            import plotly.offline as po
            if not self.curve_path.exists():
                return
            data = json.loads(self.curve_path.read_text(encoding="utf-8"))
            epochs = [e["epoch"] for e in data.get("epochs", [])]
            tl = [e["train_loss"] for e in data.get("epochs", [])]
            vl = [e["val_loss"] for e in data.get("epochs", [])]
            ta = [e["train_accuracy"] for e in data.get("epochs", [])]
            va = [e["val_accuracy"] for e in data.get("epochs", [])]
            traces = [
                go.Scatter(x=epochs, y=tl, mode="lines+markers", name="Train Loss", visible=True),
                go.Scatter(x=epochs, y=vl, mode="lines+markers", name="Val Loss", visible=True),
                go.Scatter(x=epochs, y=ta, mode="lines+markers", name="Train Acc", visible=False),
                go.Scatter(x=epochs, y=va, mode="lines+markers", name="Val Acc", visible=False),
            ]
            updatemenus = [
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Loss", method="update", args=[{"visible": [True, True, False, False]}]),
                        dict(label="Accuracy", method="update", args=[{"visible": [False, False, True, True]}]),
                    ],
                    x=0.0, y=1.15
                )
            ]
            layout = go.Layout(
                title="Learning Curves",
                xaxis=dict(title="Epoch", gridcolor="#ddd"),
                yaxis=dict(title="Metric", gridcolor="#ddd"),
                legend=dict(x=0.02, y=0.98),
                updatemenus=updatemenus,
                autosize=True,
            )
            fig = go.Figure(data=traces, layout=layout)
            po.plot(fig, filename=str(self.curve_html), auto_open=False, include_plotlyjs="cdn")
        except Exception:
            pass

    def render_from_json(self, json_path: str):
        try:
            import plotly.graph_objs as go
            import plotly.offline as po
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
            epochs = [e["epoch"] for e in data.get("epochs", [])]
            tl = [e["train_loss"] for e in data.get("epochs", [])]
            vl = [e["val_loss"] for e in data.get("epochs", [])]
            ta = [e["train_accuracy"] for e in data.get("epochs", [])]
            va = [e["val_accuracy"] for e in data.get("epochs", [])]
            traces = [
                go.Scatter(x=epochs, y=tl, mode="lines+markers", name="Train Loss", visible=True),
                go.Scatter(x=epochs, y=vl, mode="lines+markers", name="Val Loss", visible=True),
                go.Scatter(x=epochs, y=ta, mode="lines+markers", name="Train Acc", visible=False),
                go.Scatter(x=epochs, y=va, mode="lines+markers", name="Val Acc", visible=False),
            ]
            updatemenus = [
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Loss", method="update", args=[{"visible": [True, True, False, False]}]),
                        dict(label="Accuracy", method="update", args=[{"visible": [False, False, True, True]}]),
                    ],
                    x=0.0, y=1.15
                )
            ]
            layout = go.Layout(
                title="Learning Curves",
                xaxis=dict(title="Epoch", gridcolor="#ddd"),
                yaxis=dict(title="Metric", gridcolor="#ddd"),
                legend=dict(x=0.02, y=0.98),
                updatemenus=updatemenus,
                autosize=True,
            )
            fig = go.Figure(data=traces, layout=layout)
            po.plot(fig, filename=str(self.curve_html), auto_open=False, include_plotlyjs="cdn")
        except Exception:
            pass

    def _plot_history(self, hist: Dict):
        import matplotlib.pyplot as plt
        acc = hist.get("accuracy")
        val_acc = hist.get("val_accuracy")
        loss = hist.get("loss")
        val_loss = hist.get("val_loss")
        if acc and val_acc:
            plt.figure(figsize=(7,5))
            plt.plot(acc, label="train")
            plt.plot(val_acc, label="val")
            plt.legend()
            plt.title("Accuracy (Torch)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "accuracy_torch.png")
            plt.close()
        if loss and val_loss:
            plt.figure(figsize=(7,5))
            plt.plot(loss, label="train")
            plt.plot(val_loss, label="val")
            plt.legend()
            plt.title("Loss (Torch)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "loss_torch.png")
            plt.close()

    def _write_history_csv(self, hist: Dict):
        keys = ["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
        rows = []
        length = max(len(hist.get("accuracy", [])), len(hist.get("loss", [])), len(hist.get("val_accuracy", [])), len(hist.get("val_loss", [])))
        for i in range(length):
            rows.append({
                "epoch": i + 1,
                "accuracy": float(hist.get("accuracy", [None] * length)[i]) if i < len(hist.get("accuracy", [])) else None,
                "val_accuracy": float(hist.get("val_accuracy", [None] * length)[i]) if i < len(hist.get("val_accuracy", [])) else None,
                "loss": float(hist.get("loss", [None] * length)[i]) if i < len(hist.get("loss", [])) else None,
                "val_loss": float(hist.get("val_loss", [None] * length)[i]) if i < len(hist.get("val_loss", [])) else None,
            })
        out = self.reports_dir / "training_history_torch.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def evaluate(self) -> Dict:
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        size = self.cfg.image_size[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        val_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_ds = datasets.ImageFolder(str(self.test_dir), transform=val_tf)
        class_names = val_ds.classes
        loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=self.device.type=="cuda", persistent_workers=self.cfg.num_workers>0)
        model = self._build_model(num_classes=len(class_names))
        best_path = self.model_dir / "best_model_torch.pth"
        final_path = self.model_dir / "final_model_torch.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=self.device))
        elif final_path.exists():
            model.load_state_dict(torch.load(final_path, map_location=self.device))
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.cpu().tolist())
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        with open(self.reports_dir / "metrics_torch.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        txt = classification_report(y_true, y_pred, target_names=class_names)
        with open(self.reports_dir / "metrics_torch.txt", "w", encoding="utf-8") as f:
            f.write(txt)
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confusion_matrix_torch.png")
        plt.close()
        return {"report": report, "confusion_matrix": cm.tolist()}

    def _write_preprocessing_doc(self, counts: Dict[str, int], stats: Dict[str, int]) -> None:
        doc_dir = Path(self.cfg.reports_dir)
        doc_dir.mkdir(parents=True, exist_ok=True)
        total = sum(counts.values())
        lines = []
        lines.append("# Preprocessing Pipeline")
        lines.append("Input images are converted to RGB, resized to 256x256, and saved as JPEG.")
        lines.append("Normalization to 0-1 is applied during model input via Rescaling.")
        lines.append("Augmentation (flip, rotation, zoom) is applied to the training pipeline only.")
        lines.append("Stratified split maintains class balance with 70/30 train/test.")
        lines.append("Processed dataset layout under output_dir: all/, train/, test/.")
        lines.append("Class counts after preprocessing:")
        for k in sorted(counts.keys()):
            lines.append(f"- {k}: {counts[k]}")
        lines.append(f"Total: {total}")
        lines.append("Robustness checks:")
        lines.append(f"- Dark images: {stats.get('dark', 0)}")
        lines.append(f"- Bright images: {stats.get('bright', 0)}")
        lines.append(f"- Low contrast: {stats.get('low_contrast', 0)}")
        with open(doc_dir / "robustness.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        with open(doc_dir / "preprocessing.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("configs") / "pipeline.yaml"))
    parser.add_argument("--step", type=str, default="all")
    args = parser.parse_args()
    p = Pipeline(args.config)
    if args.step in {"all", "preprocess"}:
        p.run_preprocessing()
    if args.step in {"all", "split"}:
        p.run_split()
    if args.step in {"all", "train"}:
        p.run_training()
    if args.step in {"all", "eval"}:
        p.run_evaluation()
    if args.step == "train_eval":
        p.run_training()
        p.run_evaluation()


if __name__ == "__main__":
    main()
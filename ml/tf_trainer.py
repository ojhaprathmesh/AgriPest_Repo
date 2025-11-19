import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .config import PipelineConfig


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
        try:
            import tensorflow_addons as tfa2
            metrics.append(tfa2.metrics.F1Score(num_classes=num_classes, average="macro"))
        except Exception:
            pass
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
        import uuid
        from datetime import datetime, timezone
        self.model_version = uuid.uuid4().hex
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
        train_ds, val_ds, class_names = self._datasets()
        model = ModelBuilder(self.cfg, self.logger).build(num_classes=len(class_names))

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

    def _write_history_csv(self, hist: Dict[str, List[float]]):
        keys = sorted(hist.keys())
        out = self.reports_dir / "training_history_torch.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for i in range(max(len(v) for v in hist.values())):
                row = {k: (hist.get(k, [None]*i)[i] if i < len(hist.get(k, [])) else None) for k in keys}
                w.writerow(row)

    def _plot_history(self, hist: Dict[str, List[float]]):
        import matplotlib.pyplot as plt
        acc = hist.get("accuracy", [])
        val_acc = hist.get("val_accuracy", [])
        loss = hist.get("loss", [])
        val_loss = hist.get("val_loss", [])
        if acc or val_acc:
            plt.figure(figsize=(8,6))
            if acc:
                plt.plot(acc, label="train")
            if val_acc:
                plt.plot(val_acc, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / "accuracy_torch.png")
            plt.close()
        if loss or val_loss:
            plt.figure(figsize=(8,6))
            if loss:
                plt.plot(loss, label="train")
            if val_loss:
                plt.plot(val_loss, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / "loss_torch.png")
            plt.close()

    def _update_plotly_live_tf(self):
        try:
            import plotly.graph_objects as go
            data = json.loads(self.curve_path.read_text(encoding="utf-8")) if self.curve_path.exists() else None
            if not data:
                return
            epochs = [e.get("epoch") for e in data.get("epochs", [])]
            train_acc = [e.get("train_accuracy") for e in data.get("epochs", [])]
            val_acc = [e.get("val_accuracy") for e in data.get("epochs", [])]
            train_loss = [e.get("train_loss") for e in data.get("epochs", [])]
            val_loss = [e.get("val_loss") for e in data.get("epochs", [])]
            fig = go.Figure()
            if epochs:
                fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines", name="train_acc"))
                fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines", name="val_acc"))
                fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines", name="train_loss"))
                fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines", name="val_loss"))
                fig.write_html(str(self.curve_html), include_plotlyjs="cdn")
        except Exception:
            pass

    def evaluate(self) -> Dict:
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        import tensorflow as tf
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_dir,
            image_size=self.cfg.image_size,
            batch_size=self.cfg.batch_size,
            label_mode="categorical",
            shuffle=False,
        )
        class_names = val_ds.class_names
        mpath = self.model_dir / "best_model.h5"
        if not mpath.exists():
            mpath = self.model_dir / "final_model.h5"
        model = tf.keras.models.load_model(mpath)
        y_true = []
        y_pred = []
        for imgs, labels in val_ds:
            preds = model.predict(imgs, verbose=0)
            y_pred.extend(preds.argmax(axis=1).tolist())
            y_true.extend(labels.numpy().argmax(axis=1).tolist())
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        with open(self.reports_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confusion_matrix.png")
        plt.close()
        return {"report": report, "confusion_matrix": cm.tolist()}
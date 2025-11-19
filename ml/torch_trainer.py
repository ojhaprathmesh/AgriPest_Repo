import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from .config import PipelineConfig


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

    def _append_epoch(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
        try:
            data = json.loads((self.reports_dir / "learning_curve.json").read_text(encoding="utf-8")) if (self.reports_dir / "learning_curve.json").exists() else {"epochs": []}
            data.setdefault("epochs", []).append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
            })
            with open(self.reports_dir / "learning_curve.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _update_plotly_live(self):
        try:
            import plotly.graph_objects as go
            curve_path = self.reports_dir / "learning_curve.json"
            curve_html = self.plots_dir / "learning_curve.html"
            data = json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.exists() else None
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
                fig.write_html(str(curve_html), include_plotlyjs="cdn")
        except Exception:
            pass

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

    def train(self) -> Dict:
        size = self.cfg.image_size[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
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
        with open(self.reports_dir / "class_names_torch.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=self.device.type=="cuda", persistent_workers=self.cfg.num_workers>0)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=self.device.type=="cuda", persistent_workers=self.cfg.num_workers>0)
        model = self._build_model(num_classes=len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        best_val = float("inf")
        wait = 0
        patience = self.cfg.early_stopping_patience
        use_amp = self.cfg.use_mixed_precision and self.device.type == "cuda"
        device_type = self.device.type
        mp_dtype = torch.bfloat16 if str(self.cfg.amp_dtype).lower() == "bf16" else torch.float16
        try:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        except Exception:
            scaler = None
            use_amp = False
        try:
            for epoch in range(self.cfg.epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for imgs, labels in train_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        with torch.amp.autocast(device_type, enabled=use_amp, dtype=mp_dtype):
                            outputs = model(imgs)
                            loss = criterion(outputs, labels)
                        if scaler is not None and use_amp:
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
                            scaler = None
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
                train_loss = running_loss / max(total, 1)
                train_acc = correct / max(total, 1)
                model.eval()
                val_running = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        val_running += loss.item() * imgs.size(0)
                        preds = outputs.argmax(dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
                val_loss = val_running / max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)
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
            self.logger.error(f"Training failed: {e}")
        torch.save(model.state_dict(), self.model_dir / "final_model_torch.pth")
        with open(self.reports_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        self._write_history_csv(history)
        self._plot_history(history)
        return {"class_names": class_names, "history": history}

    def _plot_class_metrics(self, report: Dict, cm, class_names: List[str], probs_all, y_pred):
        import matplotlib.pyplot as plt
        import numpy as np
        recalls = []
        precisions = []
        f1s = []
        for i, name in enumerate(class_names):
            d = report.get(name, {})
            recalls.append(float(d.get("recall", 0.0)))
            precisions.append(float(d.get("precision", 0.0)))
            f1s.append(float(d.get("f1-score", 0.0)))
        plt.figure(figsize=(12,7))
        x = np.arange(len(class_names))
        w = 0.25
        plt.bar(x - w, precisions, width=w, label="precision")
        plt.bar(x, recalls, width=w, label="recall")
        plt.bar(x + w, f1s, width=w, label="f1")
        from textwrap import shorten
        short = [shorten(s, width=18, placeholder="…") for s in class_names]
        plt.xticks(x, short, rotation=90, ha="center", fontsize=6)
        plt.legend()
        plt.title("Precision/Recall/F1 per class")
        plt.subplots_adjust(bottom=0.32)
        plt.savefig(self.plots_dir / "prf1_per_class_torch.png")
        plt.close()
        acc_per_class = []
        for i in range(len(class_names)):
            total_i = cm[i].sum() if cm[i].sum() > 0 else 0
            acc_i = (cm[i, i] / total_i) if total_i > 0 else 0.0
            acc_per_class.append(float(acc_i))
        plt.figure(figsize=(12,7))
        plt.bar(np.arange(len(class_names)), acc_per_class)
        short = [shorten(s, width=18, placeholder="…") for s in class_names]
        plt.xticks(np.arange(len(class_names)), short, rotation=90, ha="center", fontsize=6)
        plt.ylabel("Accuracy")
        plt.title("Class-wise accuracy")
        plt.subplots_adjust(bottom=0.32)
        plt.savefig(self.plots_dir / "class_accuracy_torch.png")
        plt.close()
        if probs_all:
            arr = np.array(probs_all)
            pred = np.array(y_pred)
            conf = arr[np.arange(len(pred)), pred]
            means = []
            for i in range(len(class_names)):
                idx = np.where(pred == i)[0]
                m = float(conf[idx].mean()) if idx.size > 0 else 0.0
                means.append(m)
            plt.figure(figsize=(12,7))
            plt.bar(np.arange(len(class_names)), means)
            short = [shorten(s, width=18, placeholder="…") for s in class_names]
            plt.xticks(np.arange(len(class_names)), short, rotation=90, ha="center", fontsize=6)
            plt.ylabel("Mean confidence")
            plt.title("Prediction confidence per class")
            plt.subplots_adjust(bottom=0.32)
            plt.savefig(self.plots_dir / "prediction_confidence_torch.png")
            plt.close()

    def _plot_image_distribution(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        def count_images(d: Path) -> Dict[str, int]:
            res = {}
            if d.exists():
                for c in sorted([p for p in d.iterdir() if p.is_dir()]):
                    cnt = 0
                    for p in c.rglob("*"):
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                            cnt += 1
                    res[c.name] = cnt
            return res
        tr = count_images(self.train_dir)
        te = count_images(self.test_dir)
        classes = sorted(set(list(tr.keys()) + list(te.keys())))
        tr_counts = [tr.get(k, 0) for k in classes]
        te_counts = [te.get(k, 0) for k in classes]
        x = np.arange(len(classes))
        w = 0.4
        plt.figure(figsize=(16,7))
        plt.bar(x - w/2, tr_counts, width=w, label="train")
        plt.bar(x + w/2, te_counts, width=w, label="test")
        from textwrap import shorten
        short = [shorten(s, width=18, placeholder="…") for s in classes]
        plt.xticks(x, short, rotation=90, ha="center", fontsize=6)
        plt.ylabel("Images")
        plt.title("Image distribution per class")
        plt.legend()
        plt.subplots_adjust(bottom=0.4)
        plt.savefig(self.plots_dir / "image_distribution_torch.png")
        plt.close()

    def _plot_color_histogram(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        import random
        imgs = []
        for c in sorted([p for p in self.test_dir.iterdir() if p.is_dir()]):
            files = [p for p in c.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            random.shuffle(files)
            imgs.extend(files[:5])
        r_vals = []
        g_vals = []
        b_vals = []
        for p in imgs[:100]:
            try:
                im = Image.open(p).convert("RGB")
                im = im.resize((256, 256))
                arr = np.array(im)
                r_vals.extend(arr[:, :, 0].flatten().tolist())
                g_vals.extend(arr[:, :, 1].flatten().tolist())
                b_vals.extend(arr[:, :, 2].flatten().tolist())
            except Exception:
                pass
        plt.figure(figsize=(10,7))
        bins = 128
        plt.hist(r_vals, bins=bins, alpha=0.5, color="r", label="R")
        plt.hist(g_vals, bins=bins, alpha=0.5, color="g", label="G")
        plt.hist(b_vals, bins=bins, alpha=0.5, color="b", label="B")
        plt.title("Color channel histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(self.plots_dir / "color_histogram_torch.png")
        plt.close()

    def _plot_roc_curve(self, y_true, probs_all, class_names: List[str]):
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        if not probs_all:
            return
        y = np.array(y_true)
        P = np.array(probs_all)
        P = np.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
        Y = label_binarize(y, classes=list(range(P.shape[1])))
        plt.figure(figsize=(8,6))
        for i in range(P.shape[1]):
            if Y[:, i].sum() == 0 or (Y[:, i] == 0).sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(Y[:, i], P[:, i])
            a = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={a:.2f})")
        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(self.plots_dir / "roc_torch.png")

    def generate_gradcam(self, num_samples: int = 5):
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        size = self.cfg.image_size[0]
        tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        ds = datasets.ImageFolder(str(self.test_dir), transform=tfm)
        class_names = ds.classes
        loader = DataLoader(ds, batch_size=1, shuffle=True)
        model = self._build_model(num_classes=len(class_names))
        best_path = self.model_dir / "best_model_torch.pth"
        final_path = self.model_dir / "final_model_torch.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        elif final_path.exists():
            model.load_state_dict(torch.load(final_path, map_location=self.device, weights_only=True))
        model.eval()
        target_layer = None
        for n, m in model.named_modules():
            if n == "layer4.2.conv3":
                target_layer = m
                break
        acts = []
        grads = []
        h1 = target_layer.register_forward_hook(lambda m, inp, out: acts.append(out.detach()))
        h2 = target_layer.register_full_backward_hook(lambda m, gin, gout: grads.append(gout[0].detach()))
        saved = 0
        samples = []
        for imgs, labels in loader:
            if saved >= num_samples:
                break
            imgs = imgs.to(self.device)
            for g in [acts, grads]:
                g.clear()
            out = model(imgs)
            cls = out.argmax(dim=1).item()
            score = out[:, cls].sum()
            model.zero_grad(set_to_none=True)
            score.backward()
            A = acts[0]
            G = grads[0]
            w = G.mean(dim=(2,3), keepdim=True)
            cam = (A * w).sum(dim=1)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)[0].cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
            img_np = imgs[0].cpu().numpy().transpose(1,2,0)
            img_np = (img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1)
            plt.figure(figsize=(6,5))
            plt.imshow(img_np)
            plt.imshow(cam, cmap="jet", alpha=0.4)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"gradcam_torch_{saved+1}.png")
            plt.close()
            samples.append((img_np, cam, cls))
            saved += 1
        h1.remove()
        h2.remove()
        if samples:
            cols = len(samples)
            fig, axes = plt.subplots(1, cols, figsize=(3.8*cols, 3.2))
            if cols == 1:
                axes = [axes]
            for i, (img_np, cam, cls) in enumerate(samples):
                axes[i].imshow(img_np)
                axes[i].imshow(cam, cmap="jet", alpha=0.4)
                axes[i].axis("off")
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(self.plots_dir / "gradcam_torch_grid.png")
            plt.close()

    def _plot_train_val_combined(self):
        import matplotlib.pyplot as plt
        import json
        import csv
        hist = None
        jpath = self.reports_dir / "training_history.json"
        if jpath.exists():
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    hist = json.load(f)
            except Exception:
                hist = None
        if isinstance(hist, dict) and "history" in hist:
            hist = hist["history"]
        if hist is None:
            cpath = self.reports_dir / "training_history_torch.csv"
            if cpath.exists():
                acc = []
                val_acc = []
                loss = []
                val_loss = []
                with open(cpath, "r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        a = row.get("accuracy")
                        va = row.get("val_accuracy")
                        l = row.get("loss")
                        vl = row.get("val_loss")
                        acc.append(float(a) if a not in (None, "", "nan") else float("nan"))
                        val_acc.append(float(va) if va not in (None, "", "nan") else float("nan"))
                        loss.append(float(l) if l not in (None, "", "nan") else float("nan"))
                        val_loss.append(float(vl) if vl not in (None, "", "nan") else float("nan"))
                hist = {"accuracy": acc, "val_accuracy": val_acc, "loss": loss, "val_loss": val_loss}
        if hist:
            acc = hist.get("accuracy", [])
            val_acc = hist.get("val_accuracy", [])
            loss = hist.get("loss", [])
            val_loss = hist.get("val_loss", [])
            if (acc and val_acc) or (loss and val_loss):
                plt.figure(figsize=(10,5))
                ax1 = plt.subplot(1,2,1)
                if acc and val_acc:
                    ax1.plot(acc, label="train")
                    ax1.plot(val_acc, label="val")
                    ax1.set_title("Accuracy")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Accuracy")
                    ax1.legend(fontsize=8)
                ax2 = plt.subplot(1,2,2)
                if loss and val_loss:
                    ax2.plot(loss, label="train")
                    ax2.plot(val_loss, label="val")
                    ax2.set_title("Loss")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Loss")
                    ax2.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(self.plots_dir / "training_validation_torch.png")
                plt.close()

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
            model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        elif final_path.exists():
            model.load_state_dict(torch.load(final_path, map_location=self.device, weights_only=True))
        model.eval()
        y_true = []
        y_pred = []
        probs_all = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                imgs = imgs.contiguous()
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1).cpu().tolist()
                y_pred.extend(preds)
                probs_all.extend(probs.cpu().numpy().tolist())
                y_true.extend(labels.cpu().tolist())
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        with open(self.reports_dir / "metrics_torch.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        txt = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        with open(self.reports_dir / "metrics_torch.txt", "w", encoding="utf-8") as f:
            f.write(txt)
        plt.figure(figsize=(10,9))
        from textwrap import shorten
        short = [shorten(s, width=22, placeholder="…") for s in class_names]
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=short, yticklabels=short, square=True, linewidths=0.4, linecolor="#eee", cbar_kws={"shrink":0.8})
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticklabels(short, rotation=0, fontsize=6)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.subplots_adjust(left=0.2, bottom=0.28)
        plt.savefig(self.plots_dir / "confusion_matrix_torch.png")
        plt.close()
        self._plot_class_metrics(report, cm, class_names, probs_all, y_pred)
        self._plot_image_distribution()
        self._plot_color_histogram()
        self._plot_roc_curve(y_true, probs_all, class_names)
        try:
            self.generate_gradcam(num_samples=5)
        except Exception:
            pass
        try:
            self._plot_train_val_combined()
        except Exception:
            pass
        return {"report": report, "confusion_matrix": cm.tolist()}
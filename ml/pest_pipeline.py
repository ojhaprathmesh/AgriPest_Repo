import json
from pathlib import Path
import sys as _sys
from typing import Dict, List, Tuple

from config import ConfigLoader as ConfigLoader, PipelineConfig as PipelineConfig
from logger import setup_logger as setup_logger
from preprocessing import ImagePreprocessor as ImagePreprocessor
from splitter import DatasetSplitter as DatasetSplitter
from tf_trainer import ModelBuilder as ModelBuilder
from tf_trainer import TrainerEvaluator as TrainerEvaluator
from torch_trainer import TorchTrainerEvaluator as TorchTrainerEvaluator

_sys.path.append(str(Path(__file__).parent))

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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from .config import TrainConfig
from .data import check_label_coverage, encode_dataset, label_maps


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(label2id: dict[str, int]) -> Callable[[EvalPrediction], dict[str, float]]:
    eval_labels = list(label2id.values())

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(
                f1_score(labels, predictions, average="macro", labels=eval_labels, zero_division=0)
            ),
            "accuracy": float(accuracy_score(labels, predictions)),
        }

    return compute_metrics


def load_model(checkpoint: str, dataset: Dataset, id_column: str, label_column: str) -> PreTrainedModel:
    id2label, label2id, _ = label_maps(dataset, id_column, label_column)
    config = AutoConfig.from_pretrained(checkpoint, label2id=label2id, id2label=id2label)
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)


def calculate_class_weights(
    dataset: Dataset, id_column: str, num_labels: int
) -> tuple[np.ndarray, torch.Tensor]:
    counts = np.bincount(np.array(dataset[id_column]), minlength=num_labels)
    class_weights = torch.tensor(counts.sum() / (num_labels * counts), dtype=torch.float)
    print("label counts:", counts, "\nclass weights:", class_weights)
    return counts, class_weights


@dataclass
class Builder:
    """Builds a trainer for one run: a shared TrainConfig plus the model and run name for this run.

    config = TrainConfig(text_column="text", id_column="label", label_column="emotion")
    muril = Builder(config, checkpoint="google/muril-base-cased", run_name="muril",
                    train_ds=train_ds, val_ds=val_ds)
    xlmr = Builder(config, checkpoint="xlm-roberta-base", run_name="xlmr",
                   train_ds=train_ds, val_ds=val_ds)
    muril.build_trainer().train()
    """

    config: TrainConfig
    checkpoint: str
    run_name: str  # also the output folder name (run_<run_name>), so it must differ per run
    train_ds: Dataset
    val_ds: Dataset

    def __post_init__(self) -> None:
        for name in ("checkpoint", "run_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")

    def build_trainer(self) -> WeightedTrainer:
        cfg = self.config
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        encode_kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "text_column": cfg.text_column,
            "id_column": cfg.id_column,
            "max_length": cfg.max_length,
        }
        encoded_training_dataset = encode_dataset(self.train_ds, **encode_kwargs)
        encoded_validation_dataset = encode_dataset(self.val_ds, **encode_kwargs)

        _, label2id, num_labels = label_maps(self.train_ds, cfg.id_column, cfg.label_column)
        check_label_coverage(label2id, self.val_ds, cfg.id_column, cfg.label_column)
        class_weights = (
            calculate_class_weights(self.train_ds, cfg.id_column, num_labels)[1]
            if cfg.use_class_weights
            else None
        )

        model = load_model(
            checkpoint=self.checkpoint,
            dataset=self.train_ds,
            id_column=cfg.id_column,
            label_column=cfg.label_column,
        )

        training_args = TrainingArguments(
            output_dir=f"run_{self.run_name}",
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_ratio,  # a float < 1 is treated as a ratio of total steps
            weight_decay=cfg.weight_decay,
            per_device_train_batch_size=cfg.train_batch,
            per_device_eval_batch_size=cfg.eval_batch,
            fp16=cfg.fp16,
            eval_strategy=cfg.eval_strategy,
            save_strategy=cfg.save_strategy,
            logging_strategy=cfg.logging_strategy,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            save_total_limit=cfg.save_total_limit,
            seed=cfg.seed,
            report_to=cfg.report_to,
            disable_tqdm=cfg.disable_tqdm,
        )

        return WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_training_dataset,
            eval_dataset=encoded_validation_dataset,
            processing_class=tokenizer,
            compute_metrics=make_compute_metrics(label2id),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
            class_weights=class_weights,
        )

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from ..train.data import tokenize_batch
from ..train.train import make_compute_metrics


@dataclass
class EvalResult:
    metrics: dict[str, float]
    report: dict[str, Any]
    confusion_matrix: np.ndarray  # rows = true, columns = predicted, ordered as label_names
    label_names: list[str]
    predictions: np.ndarray
    label_ids: np.ndarray

    def plot_confusion_matrix(self, title: str = "Confusion matrix") -> None:
        from ..view import plot_confusion_matrix

        plot_confusion_matrix(self, title)

    def plot_per_class_f1(self, title: str = "Per-class F1") -> None:
        from ..view import compare_per_class_f1

        compare_per_class_f1({title: self}, title)

    def misclassified(self, test_ds: Dataset, text_column: str, min_count: int = 4) -> pd.DataFrame:
        """Texts from every (true, predicted) mix-up that happened at least `min_count` times.

        `test_ds` must be the same dataset, in the same order, that was passed to evaluate_model.
        Rows are grouped by mix-up, most frequent first, with columns text / true / predicted.
        """
        if len(test_ds) != len(self.predictions):
            raise ValueError(
                f"test_ds has {len(test_ds)} rows but this result has {len(self.predictions)} predictions; "
                "pass the same dataset that was evaluated"
            )
        names = np.array(self.label_names)
        df = pd.DataFrame(
            {
                "text": test_ds[text_column],
                "true": names[self.label_ids],
                "predicted": names[self.predictions],
            }
        )
        wrong = df[df["true"] != df["predicted"]]
        pair_count = wrong.groupby(["true", "predicted"])["text"].transform("size")
        frequent = wrong.assign(_count=pair_count)[pair_count >= min_count]
        return (
            frequent.sort_values(
                ["_count", "true", "predicted"], ascending=[False, True, True], kind="stable"
            )
            .drop(columns="_count")
            .reset_index(drop=True)
        )


def evaluate_model(
    checkpoint: str,
    test_ds: Dataset,
    text_column: str,
    label_column: str,
    subfolder: str | None = None,
    batch_size: int = 64,
    max_length: int = 128,
    print_report: bool = True,
) -> EvalResult:
    """Evaluate a trained classifier on unseen data.

    `checkpoint` is a local path or Hub repo id (with optional `subfolder`). Gold labels are read
    from `label_column` as names and mapped through the model's own label2id, so ids in the test
    data do not need to match the ones used in training.

    Raises ValueError if the test data contains a label the model was not trained on.
    """
    subfolder_kwargs = {"subfolder": subfolder} if subfolder else {}
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, **subfolder_kwargs
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(checkpoint, **subfolder_kwargs)

    if not model.config.id2label:
        raise ValueError(f"{checkpoint} has no id2label in its config")
    id2label = {int(i): str(name) for i, name in model.config.id2label.items()}
    label2id = {name: i for i, name in id2label.items()}
    unknown = set(test_ds[label_column]) - label2id.keys()
    if unknown:
        raise ValueError(f"{label_column} values not known to the model: {sorted(unknown)}")

    def _encode(batch: dict[str, list[Any]]) -> BatchEncoding:
        encoded = tokenize_batch(batch, tokenizer, text_column, max_length)
        encoded["labels"] = [label2id[name] for name in batch[label_column]]
        return encoded

    encoded_ds = test_ds.map(_encode, batched=True, remove_columns=test_ds.column_names)

    evaluator = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="eval_tmp",
            per_device_eval_batch_size=batch_size,
            report_to="none",
            disable_tqdm=not print_report,
        ),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(label2id),
    )
    output = evaluator.predict(encoded_ds, metric_key_prefix="test")
    predictions = np.argmax(output.predictions, axis=-1)
    label_ids = output.label_ids
    assert isinstance(label_ids, np.ndarray)  # always set: _encode adds a single labels column

    all_ids = sorted(id2label)
    label_names = [id2label[i] for i in all_ids]
    metrics = {
        k.removeprefix("test_"): float(v)
        for k, v in (output.metrics or {}).items()
        if k in ("test_loss", "test_accuracy", "test_macro_f1")
    }
    report = classification_report(
        label_ids, predictions, labels=all_ids, target_names=label_names, zero_division=0, output_dict=True
    )
    matrix = confusion_matrix(label_ids, predictions, labels=all_ids)

    if print_report:
        print(f"=== {checkpoint}{'/' + subfolder if subfolder else ''} ===")
        print({k: round(v, 4) for k, v in metrics.items()})
        print(
            classification_report(
                label_ids, predictions, labels=all_ids, target_names=label_names, zero_division=0, digits=3
            )
        )
        print("Confusion matrix (rows = true, columns = predicted):")
        print(label_names)
        print(matrix)

    return EvalResult(
        metrics=metrics,
        report=report,
        confusion_matrix=matrix,
        label_names=label_names,
        predictions=predictions,
        label_ids=label_ids,
    )


# Usage
# from sage_ml.evaluate import evaluate_model
# result = evaluate_model("alexeino/sage_runs", test_ds, "text", "emotion", subfolder="run_2_epoch5")
# result.metrics["macro_f1"]

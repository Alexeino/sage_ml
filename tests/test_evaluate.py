from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from sage_ml.evaluate import EvalResult, evaluate_model


def test_text_and_label_columns_are_required(tiny_checkpoint: str, dataset: Dataset) -> None:
    with pytest.raises(TypeError):
        evaluate_model(tiny_checkpoint, dataset)  # type: ignore[call-arg]


def test_evaluate_model(
    tiny_checkpoint: str, dataset: Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    result = evaluate_model(tiny_checkpoint, dataset, "sentence", "label_name", print_report=False)

    assert isinstance(result, EvalResult)
    assert set(result.metrics) == {"loss", "accuracy", "macro_f1"}
    assert result.label_names == ["joy", "sadness", "anger"]
    assert result.confusion_matrix.shape == (3, 3)
    assert len(result.predictions) == len(dataset)


def test_unknown_label_is_rejected(
    tiny_checkpoint: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    unseen = Dataset.from_dict({"sentence": ["happy"], "label_name": ["fear"]})

    with pytest.raises(ValueError, match="fear"):
        evaluate_model(tiny_checkpoint, unseen, "sentence", "label_name", print_report=False)


def test_misclassified(tiny_checkpoint: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = Dataset.from_dict({"sentence": [f"text {i}" for i in range(6)], "label_name": ["joy"] * 6})
    result = evaluate_model(tiny_checkpoint, dataset, "sentence", "label_name", print_report=False)
    # Overwrite predictions so the mix-ups are known: 3x joy->anger, 2x joy->sadness, 1 correct.
    result.predictions = np.array([2, 2, 2, 1, 1, 0])

    mistakes = result.misclassified(dataset, "sentence", min_count=2)

    assert list(mistakes.columns) == ["text", "true", "predicted"]
    assert list(mistakes["predicted"]) == ["anger"] * 3 + ["sadness"] * 2
    assert result.misclassified(dataset, "sentence", min_count=3)["predicted"].unique().tolist() == ["anger"]
    with pytest.raises(ValueError, match="same dataset"):
        result.misclassified(dataset.select(range(3)), "sentence")

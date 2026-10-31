from pathlib import Path

import pytest
from datasets import Dataset
from transformers import PreTrainedModel

from sage_ml.train import Builder, TrainConfig, WeightedTrainer


@pytest.mark.parametrize("missing", ["checkpoint", "run_name", "train_ds", "val_ds"])
def test_builder_arguments_are_required(missing: str, config: TrainConfig, dataset: Dataset) -> None:
    kwargs = {"checkpoint": "some/model", "run_name": "run", "train_ds": dataset, "val_ds": dataset}
    del kwargs[missing]
    with pytest.raises(TypeError, match=missing):
        Builder(config, **kwargs)


@pytest.mark.parametrize("field", ["checkpoint", "run_name"])
def test_builder_rejects_empty_values(field: str, config: TrainConfig, dataset: Dataset) -> None:
    kwargs = {"checkpoint": "some/model", "run_name": "run", field: " "}
    with pytest.raises(ValueError, match=field):
        Builder(config, train_ds=dataset, val_ds=dataset, **kwargs)


def test_build_trainer(
    tiny_checkpoint: str,
    config: TrainConfig,
    dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    builder = Builder(config, checkpoint=tiny_checkpoint, run_name="muril", train_ds=dataset, val_ds=dataset)

    trainer = builder.build_trainer()

    assert isinstance(trainer, WeightedTrainer)
    assert str(trainer.args.output_dir).endswith("run_muril")
    assert isinstance(trainer.model, PreTrainedModel)
    assert trainer.model.config.id2label == {0: "joy", 1: "sadness", 2: "anger"}
    assert trainer.class_weights is not None


def test_class_weights_can_be_turned_off(
    tiny_checkpoint: str, dataset: Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    config = TrainConfig(
        text_column="sentence", id_column="label_id", label_column="label_name", use_class_weights=False
    )
    builder = Builder(config, checkpoint=tiny_checkpoint, run_name="r", train_ds=dataset, val_ds=dataset)

    assert builder.build_trainer().class_weights is None


def test_one_config_shared_by_two_builders(
    tiny_checkpoint: str,
    config: TrainConfig,
    dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    first = Builder(config, checkpoint=tiny_checkpoint, run_name="a", train_ds=dataset, val_ds=dataset)
    second = Builder(config, checkpoint=tiny_checkpoint, run_name="b", train_ds=dataset, val_ds=dataset)

    assert first.build_trainer().args.output_dir != second.build_trainer().args.output_dir
    assert first.config is second.config


def test_labels_come_from_id_column(
    tiny_checkpoint: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two label sets in one dataset. The "label" column belongs to the other task and must not be
    # picked up just because Trainer treats a column with that name as the target by default.
    monkeypatch.chdir(tmp_path)
    dataset = Dataset.from_dict(
        {
            "text": ["happy day", "sad day", "angry day", "happy"],
            "label": [2, 2, 2, 2],
            "emotion": ["anger"] * 4,
            "category_id": [0, 1, 2, 0],
            "category": ["personal", "work", "health", "personal"],
        }
    )
    config = TrainConfig(text_column="text", id_column="category_id", label_column="category")
    trainer = Builder(
        config, checkpoint=tiny_checkpoint, run_name="r", train_ds=dataset, val_ds=dataset
    ).build_trainer()

    batch = next(iter(trainer.get_train_dataloader()))
    assert sorted(batch["labels"].tolist()) == [0, 0, 1, 2]

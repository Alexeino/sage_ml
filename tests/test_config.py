from dataclasses import fields
from typing import Any

import pytest

from sage_ml.train import TrainConfig

COLUMNS: dict[str, Any] = {"text_column": "text", "id_column": "label", "label_column": "emotion"}


@pytest.mark.parametrize("missing", COLUMNS)
def test_column_names_are_required(missing: str) -> None:
    kwargs = {k: v for k, v in COLUMNS.items() if k != missing}
    with pytest.raises(TypeError, match=missing):
        TrainConfig(**kwargs)


@pytest.mark.parametrize("column", COLUMNS)
def test_empty_column_name_is_rejected(column: str) -> None:
    with pytest.raises(ValueError, match=column):
        TrainConfig(**{**COLUMNS, column: "  "})


def test_checkpoint_and_run_name_are_not_on_config() -> None:
    # These are per run and belong on Builder, so one config can be shared across models.
    names = {f.name for f in fields(TrainConfig)}
    assert "checkpoint" not in names
    assert "run_name" not in names


def test_only_columns_are_required() -> None:
    config = TrainConfig(**COLUMNS)
    assert config.use_class_weights is True

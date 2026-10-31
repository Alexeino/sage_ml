from pathlib import Path

import pytest
from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from sage_ml.train import TrainConfig

LABELS = ["joy", "sadness", "anger"]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A tiny randomly initialised BERT classifier saved locally, so tests need no downloads."""
    path: Path = tmp_path_factory.mktemp("tiny_model")
    vocab = path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "happy", "sad", "angry", "day"])
    )
    config = BertConfig(
        vocab_size=9,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)},
    )
    BertForSequenceClassification(config).save_pretrained(path)
    BertTokenizer(str(vocab)).save_pretrained(path)
    return str(path)


@pytest.fixture
def dataset() -> Dataset:
    # Non-default column names, so tests fail if a column name is ever hardcoded again.
    return Dataset.from_dict(
        {
            "sentence": ["happy day", "sad day", "angry day", "happy"],
            "label_id": [0, 1, 2, 0],
            "label_name": ["joy", "sadness", "anger", "joy"],
        }
    )


@pytest.fixture
def config() -> TrainConfig:
    return TrainConfig(text_column="sentence", id_column="label_id", label_column="label_name")

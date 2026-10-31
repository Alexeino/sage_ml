from typing import Any

from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase


def label_maps(
    dataset: Dataset, id_column: str, label_column: str
) -> tuple[dict[int, str], dict[str, int], int]:
    """Build id2label / label2id / num_labels from two columns of a Hugging Face Dataset.

    Raises ValueError if an id maps to more than one label, a label to more than one id, or the ids
    are not 0..N-1 with no gaps (e.g. a label has no examples in this split).
    """
    id2label: dict[int, str] = {}
    for i, name in zip(dataset[id_column], dataset[label_column], strict=True):
        i = int(i)
        prev = id2label.setdefault(i, name)
        if prev != name:
            raise ValueError(f"{id_column}={i} maps to both {prev!r} and {name!r}")

    id2label = dict(sorted(id2label.items()))
    label2id = {name: i for i, name in id2label.items()}
    if len(label2id) != len(id2label):
        names = list(id2label.values())
        dupes = [n for n in label2id if names.count(n) > 1]
        raise ValueError(f"{label_column} values mapped to multiple ids: {dupes}")

    missing_ids = sorted(set(range(max(id2label) + 1)) - id2label.keys()) if id2label else []
    if missing_ids or min(id2label, default=0) < 0:
        raise ValueError(
            f"{id_column} must be 0..N-1 with no gaps; ids with no examples: {missing_ids}. "
            "Every label needs at least one example in this split."
        )

    return id2label, label2id, len(id2label)


def check_label_coverage(
    label2id: dict[str, int], dataset: Dataset, id_column: str, label_column: str
) -> None:
    """Check every (id, label) pair in `dataset` matches `label2id` (usually built from the train split).

    Raises ValueError listing labels the model has not seen, or labels whose id differs.
    """
    pairs = set(zip(dataset[id_column], dataset[label_column], strict=True))
    unseen = sorted({name for _, name in pairs} - label2id.keys())
    if unseen:
        raise ValueError(f"{label_column} values not in the training split: {unseen}")
    mismatched = sorted((name, int(i)) for i, name in pairs if label2id[name] != int(i))
    if mismatched:
        raise ValueError(f"{label_column} values with a different {id_column} than in training: {mismatched}")
    print("Ok")


def tokenize_batch(
    dataset_chunk: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    max_length: int | None = None,
) -> BatchEncoding:
    # No padding here: Trainer pads each batch dynamically via DataCollatorWithPadding.
    return tokenizer(dataset_chunk[text_column], truncation=True, max_length=max_length)


def encode_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    id_column: str,
    max_length: int | None = None,
) -> Dataset:
    """Tokenize `text_column` and set `labels` from `id_column`, dropping every other column.

    Dropping the rest matters: Trainer treats any column named "label" as the target, so a stray
    one (e.g. another task's ids in the same dataset) would otherwise silently replace `id_column`.
    """

    def _encode(batch: dict[str, list[Any]]) -> BatchEncoding:
        encoded = tokenize_batch(batch, tokenizer, text_column, max_length)
        encoded["labels"] = [int(i) for i in batch[id_column]]
        return encoded

    return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)

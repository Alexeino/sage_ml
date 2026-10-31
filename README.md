# sage_ml

Reusable helpers for training and evaluating Hugging Face text classification models.
You describe your dataset once with `TrainConfig`, then build a ready-to-run `Trainer` for any
checkpoint with `Builder`, and score saved models on unseen data with `evaluate_model`.

Supports single-label classification (one label per text). Class weighting, early stopping, and
label-consistency checks are built in.

## Installation

Pin a tag (or commit) so every run uses the same code.

```bash
# From GitHub
pip install "git+https://github.com/Alexeino/sage_ml.git@v0.1.0"

# Local copy, editable (changes to the source take effect without reinstalling)
pip install -e path/to/sage_ml
```

**Kaggle:** turn on *Settings → Internet*, then run the install as the first cell:

```python
!pip install -q "git+https://github.com/Alexeino/sage_ml.git@v0.1.0"
```

`sage_ml` needs `transformers>=5`. If the install upgrades it after `transformers` was already
imported, restart the session.

## Preparing your dataset

Each split is a Hugging Face `Dataset` with three columns. Their names are up to you:

| Column | Holds | Example |
|---|---|---|
| text | the input to classify | `"I can't stop smiling today"` |
| id | the label as an integer | `0` |
| label | the label as a name | `"joy"` |

The ids must run `0..N-1` with no gaps, and every label must appear at least once in the
training split.

If your dataset only has an integer `ClassLabel` column, add the names first:

```python
ds = ds.map(lambda x: {"emotion": ds.features["label"].int2str(x["label"])})
```

### Several label columns in one dataset

To train separate models on different label columns of the same dataset (e.g. `emotion` and
`category`), give **each label column its own id column** and one config per task:

```python
emotion_config = TrainConfig(text_column="text", id_column="emotion_id", label_column="emotion")
category_config = TrainConfig(text_column="text", id_column="category_id", label_column="category")
```

The id column must hold the ids of *that* label column; a shared `label` column can't describe
both. Only the configured text and id columns are used for training. Every other column,
including another task's labels, is ignored.

To create an id column from a name column:

```python
names = sorted(set(ds["train"]["category"]))
ds = ds.map(lambda x: {"category_id": names.index(x["category"])})
```

## Training

```python
from sage_ml.train import Builder, TrainConfig

config = TrainConfig(text_column="text", id_column="label", label_column="emotion")

builder = Builder(
    config,
    checkpoint="google/muril-base-cased",
    run_name="muril",
    train_ds=train_ds,
    val_ds=val_ds,
)
trainer = builder.build_trainer()
trainer.train()
trainer.save_model("run_muril/best")
```

`checkpoint` and `run_name` have no defaults on purpose, so a run never silently uses the
wrong model. Checkpoints are written to `run_<run_name>/`.

### Comparing models

One config can be shared across several builders. Give each a different `run_name` so their
outputs don't overwrite each other:

```python
muril = Builder(
    config, checkpoint="google/muril-base-cased", run_name="muril", train_ds=train_ds, val_ds=val_ds
)
xlmr = Builder(config, checkpoint="xlm-roberta-base", run_name="xlmr", train_ds=train_ds, val_ds=val_ds)

muril.build_trainer().train()
xlmr.build_trainer().train()
```

To change one setting for a single run, copy the config with `dataclasses.replace`. The
original is left unchanged:

```python
from dataclasses import replace

no_weights = replace(config, use_class_weights=False)
```

### TrainConfig options

Only the three column names are required.

| Field | Default | Notes |
|---|---|---|
| `text_column`, `id_column`, `label_column` | required | see [Preparing your dataset](#preparing-your-dataset) |
| `use_class_weights` | `True` | weight the loss by inverse class frequency, for imbalanced data |
| `learning_rate` | `5e-5` | |
| `num_epochs` | `5` | |
| `train_batch` / `eval_batch` | `16` / `32` | per-device batch sizes |
| `warmup_ratio` | `0.1` | fraction of total steps used for learning-rate warmup |
| `weight_decay` | `0.01` | |
| `max_length` | `128` | longer texts are truncated |
| `patience` | `3` | early stopping: evaluations without improvement before stopping |
| `metric_for_best_model` | `"eval_loss"` | also `"eval_macro_f1"` or `"eval_accuracy"` |
| `greater_is_better` | `False` | set `True` when using F1 or accuracy above |
| `fp16` | `False` | set `True` on a CUDA GPU (e.g. Kaggle) for faster training |
| `seed` | `42` | |
| `report_to` | `"none"` | e.g. `"wandb"` or `"tensorboard"` |
| `save_total_limit` | `2` | number of checkpoints kept per run |

`eval_strategy`, `save_strategy`, `logging_strategy`, `load_best_model_at_end` and
`disable_tqdm` are passed straight to `TrainingArguments`.

During training, evaluation reports `eval_loss`, `eval_macro_f1` and `eval_accuracy`.

## Evaluating on unseen data

```python
from sage_ml.evaluate import evaluate_model

result = evaluate_model("run_muril/best", test_ds, text_column="text", label_column="emotion")
```

For a model on the Hub stored in a subfolder:

```python
result = evaluate_model(
    "alexeino/sage_runs", test_ds, text_column="text", label_column="emotion", subfolder="run_2_epoch5"
)
```

This prints the metrics, a per-class report and a confusion matrix, and returns an `EvalResult`:

| Attribute | Contents |
|---|---|
| `metrics` | `{"loss", "accuracy", "macro_f1"}` |
| `report` | per-class precision / recall / F1 as a dict |
| `confusion_matrix` | rows = true label, columns = predicted |
| `label_names` | class names, in the confusion matrix order |
| `predictions`, `label_ids` | predicted and true ids per example |

Labels are matched by **name** using the model's own mapping, so the test set's ids don't need
to match training. The test set doesn't need an id column at all. Pass `print_report=False` to
skip the printout.

Macro F1 averages over every class the model knows, including classes with no examples in the
test set (which score 0). If your test set doesn't cover every class, look at
`result.report["weighted avg"]` too.

### Plotting evaluation results

**Confusion matrix**: cells are shaded by the share of each true class, so rare classes are as
readable as common ones, and labelled with the raw counts:

```python
result.plot_confusion_matrix("MuRIL emotion")
```

**Per-class F1** for one model, as horizontal bars sorted from best to worst class. Each class
shows its test-set count (`n=`), which tells you how much to trust its score:

```python
result.plot_per_class_f1("MuRIL emotion")
```

**Per-class F1 for several models side by side**, up to 3. Evaluate each model on the same test
set, then:

```python
from sage_ml.view import compare_per_class_f1

muril_result = evaluate_model("run_muril/best", test_ds, text_column="text", label_column="emotion")
xlmr_result = evaluate_model("run_xlmr/best", test_ds, text_column="text", label_column="emotion")

compare_per_class_f1({"MuRIL": muril_result, "XLM-R": xlmr_result}, title="Emotion: per-class F1")
```

Classes are sorted by the first model's F1. The functions `plot_confusion_matrix(result)` and
`compare_per_class_f1({...})` are also available from `sage_ml.view`.

### Reading the mistakes

To see the texts behind the confusion matrix, `misclassified` collects every (true, predicted)
mix-up that happened at least `min_count` times:

```python
mistakes = result.misclassified(test_ds, text_column="text", min_count=4)
mistakes.head(20)   # columns: text, true, predicted
```

Rows are grouped by mix-up, most frequent first. Pass the same `test_ds` you gave to
`evaluate_model`, unshuffled and unfiltered, so each text lines up with its prediction.

## Plotting training runs

`sage_ml.view` plots per-epoch history. Each run can be a trainer (after `trainer.train()`) or
the path to a saved `trainer_state.json`, found in every checkpoint folder.

**One run**: training and validation loss next to validation macro-F1:

```python
from sage_ml.view import plot_run

plot_run(trainer, title="MuRIL emotion")
plot_run("run_muril/checkpoint-500/trainer_state.json", title="MuRIL emotion")
```

The dashed line marks the epoch the Trainer kept as best, which follows
`metric_for_best_model`. If that information is missing, it marks the epoch with the lowest
validation loss.

**Several runs**: validation loss and macro-F1 side by side, up to 3 runs:

```python
from sage_ml.view import compare_runs

compare_runs(
    {
        "MuRIL": muril_trainer,
        "XLM-R": xlmr_trainer,
        "MuRIL (earlier run)": "run_muril_v1/checkpoint-500/trainer_state.json",
    }
)
```

Runs can have different numbers of epochs. To get the history as DataFrames instead of a plot,
use `run_history(trainer)`, which returns `(evals, train)`.

## Label checks

Before loading the model, `build_trainer()` checks your labels and stops with a clear error if:

- an id has no examples in the training split (ids must be `0..N-1`)
- an id maps to two names, or a name to two ids
- the validation split has a label the training split doesn't
- a label has a different id in validation than in training

`evaluate_model` raises an error if the test set has a label the model wasn't trained on.

## Limitations

- One label per text: no multi-label, sentence-pair (e.g. NLI) or regression tasks.
- Metrics are fixed to macro F1 and accuracy.
- Decoder models such as GPT-2 or Llama often have no padding token. Set `tokenizer.pad_token`
  before training them.

## Development

```bash
uv pip install -e . --group dev   # or: pip install -e . mypy ruff pytest

make format     # format, auto-fix lint issues, then type-check
make check      # lint, type-check and run tests without changing files
make test       # tests only
```

The tests use a tiny local model, so they run offline in about a second.

## License

Apache-2.0. See [LICENSE](LICENSE).

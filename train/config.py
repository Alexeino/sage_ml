from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Required on purpose, so a run never silently uses the wrong columns.
    # The model checkpoint and run name are per run, so they live on Builder instead.
    text_column: str
    id_column: str
    label_column: str
    use_class_weights: bool = True
    learning_rate: float = 5e-5
    num_epochs: int = 5
    train_batch: int = 16
    eval_batch: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    patience: int = 3
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_length: int = 128
    fp16: bool = False
    seed: int = 42
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    load_best_model_at_end: bool = True
    save_total_limit: int = 2
    report_to: str = "none"
    disable_tqdm: bool = False

    def __post_init__(self) -> None:
        for name in ("text_column", "id_column", "label_column"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")

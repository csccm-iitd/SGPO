import yaml
from sgpo.config.schema import Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig


def _dict_to_dataclass(cls, d):
    """Convert a dict to a dataclass, ignoring extra keys."""
    if d is None:
        return cls()
    fieldnames = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in fieldnames}
    return cls(**filtered)


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    data_cfg = _dict_to_dataclass(DataConfig, raw.get("data"))
    model_cfg = _dict_to_dataclass(ModelConfig, raw.get("model"))
    training_cfg = _dict_to_dataclass(TrainingConfig, raw.get("training"))
    logging_cfg = _dict_to_dataclass(LoggingConfig, raw.get("logging"))
    device = raw.get("device", "auto")

    return Config(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        logging=logging_cfg,
        device=device,
    )

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import yaml

from sissr.configs.schema import ExperimentConfig


class ConfigYamlLoader(yaml.SafeLoader):
    pass


ConfigYamlLoader.yaml_implicit_resolvers = {
    key: [resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _yaml_load(raw_value: str) -> object:
    return yaml.load(raw_value, Loader=ConfigYamlLoader)


def _set_nested(d: dict[str, object], dotted_key: str, value: object) -> None:
    *parts, leaf = dotted_key.split(".")
    for part in parts:
        sub = d.get(part)
        if sub is None:
            sub = {}
            d[part] = sub
        if not isinstance(sub, dict):
            raise ValueError(f"Override path '{dotted_key}' traverses non-mapping field '{part}'")
        d = sub
    d[leaf] = value


def load_config(yaml_path: str, overrides: Sequence[str] = ()) -> ExperimentConfig:
    raw = _yaml_load(Path(yaml_path).read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected top-level mapping in {yaml_path}, got {type(raw).__name__}")

    config_dict = cast(dict[str, object], raw)
    for override in overrides:
        key, sep, val = override.partition("=")
        if not sep:
            raise ValueError(f"Invalid override '{override}'; expected key=value")
        override_value = "" if val == "" else _yaml_load(val)
        _set_nested(config_dict, key, override_value)
    return ExperimentConfig.model_validate(config_dict)

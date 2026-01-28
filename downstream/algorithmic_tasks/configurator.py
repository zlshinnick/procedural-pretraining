"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

import yaml

EXPERIMENT_CONFIG_KEYS = {
    "pretrained_model_path",
    "weights_to_be_transferred",
    "weights_to_be_trained",
    "embedding_init_strategy",
    "wandb_project",
}


def _types_compatible(existing_value, new_value) -> bool:
    if existing_value is None:
        return True
    if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
        return True
    return isinstance(new_value, type(existing_value))


def _override_from_mapping(mapping: dict, source_label: str) -> None:
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected a mapping in {source_label}, got {type(mapping)}")
    for key, val in mapping.items():
        if key in globals():
            if not _types_compatible(globals()[key], val):
                raise TypeError(
                    f"Type mismatch for '{key}' from {source_label}: "
                    f"expected {type(globals()[key])}, got {type(val)}"
                )
            print(f"Overriding: {key} = {val}")
            globals()[key] = val
        else:
            print(f"Setting new config key from {source_label}: {key} = {val}")
            globals()[key] = val


for arg in sys.argv[1:]:
    if "=" not in arg:
        # assume it's the name of a config file
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            config_contents = f.read()
            print(config_contents)
        if config_file.endswith((".yaml", ".yml")):
            config_data = yaml.safe_load(config_contents) or {}
            _override_from_mapping(config_data, config_file)
            if (
                "experiment_config_path" in globals()
                and "experiment_config_path" not in config_data
                and any(key in config_data for key in EXPERIMENT_CONFIG_KEYS)
            ):
                print(
                    f"Using {config_file} as experiment_config_path "
                    "because it contains experiment_config keys"
                )
                globals()["experiment_config_path"] = config_file
        else:
            exec(config_contents)
    else:
        # assume it's a --key=value argument
        assert arg.startswith("--")
        t = arg.split("=")
        key = t[0]
        val = "=".join(t[1:])
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

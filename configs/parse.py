import os
import yaml
from typing import Dict
from types import SimpleNamespace


def dict_to_namespace(d: dict) -> SimpleNamespace:
    """
    Recursively convert a dictionary to a SimpleNamespace object.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return dict_to_namespace(config_dict)


def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config (Dict): Configuration dictionary to save.
        config_path (str): Path to the output YAML configuration file.
    """
    directory = os.path.dirname(config_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    # Example: Load a config file
    config_path = "src/configs/guidecad.yaml"
    config = load_config(config_path)
    print("Loaded Configuration:")
    print(config)

    # Example: Modify and save the config
    config["new_param"] = "example_value"
    save_path = "src/configs/guidecad_updated.yaml"
    save_config(config, save_path)

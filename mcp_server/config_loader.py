import logging
import os

import yaml


def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_path}")

            # Basic validation (optional)
            if not config or 'components' not in config or 'ssh_defaults' not in config:
                logging.warning("Configuration appears incomplete. Missing 'components' or 'ssh_defaults'.")
            return config

    except yaml.YAMLError as e:
        logging.error(f"Failed to parse YAML file {config_path}: {e}")
        return None
    except Exception as e:
        logging.exception(f"Unexpected error while loading config from {config_path}: {e}")
        return None

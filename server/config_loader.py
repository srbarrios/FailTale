# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging
import os
import yaml

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    if not os.path.isfile(config_path):
        logging.error("Configuration file not found: %s", config_path)
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logging.info("Configuration loaded from: %s", config_path)

        if 'components' not in config or 'ssh_defaults' not in config:
            logging.warning("Incomplete configuration: missing 'components' or 'ssh_defaults'.")

        return config

    except yaml.YAMLError as e:
        logging.error("YAML parsing error in %s: %s", config_path, e)
    except Exception as e:
        logging.exception("Unexpected error while loading config from %s", config_path)

    return None

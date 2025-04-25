from abc import ABC, abstractmethod
import logging
# Import the function signature for type hinting if desired, or the function itself
# from ..ssh_executor import execute_remote_command_async

class BaseCollector(ABC):
    """Abstract Base Class for data collectors for different host roles."""

    def __init__(self, hostname, ssh_executor_func, ssh_config):
        """
        Initializes the base collector.
        Args:
            hostname (str): The target hostname or IP address.
            ssh_executor_func (callable): The function to execute SSH commands.
            ssh_config (dict): SSH configuration details (user, key_path, etc.).
        """
        if not hostname or not ssh_executor_func or not ssh_config:
            raise ValueError("Hostname, ssh_executor_func, and ssh_config are required for BaseCollector")

        self.hostname = hostname
        self.ssh_executor = ssh_executor_func # Store the function reference
        self.ssh_config = ssh_config
        logging.debug(f"BaseCollector initialized for host: {self.hostname}")

    @abstractmethod
    def collect(self, data_definitions):
        """
        Abstract method to collect data based on the provided definitions.
        Must be implemented by subclasses.

        Args:
            data_definitions (list): A list of dictionaries defining data to collect
                                     (e.g., {'description': '...', 'command': '...'}).

        Returns:
            list: A list of result dictionaries for each collected item.
        """
        pass

    def get_ssh_param(self, key, default=None):
        """Helper to safely get SSH parameters."""
        return self.ssh_config.get(key, default)

    # Add other common helper methods here if needed
    # def _clean_output(self, text):
    #     return text.strip() if text else ""

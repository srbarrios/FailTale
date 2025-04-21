import logging
from .base_collector import BaseCollector

class ProxyCollector(BaseCollector):
    """Collector implementation for hosts with the 'proxy' role."""

    def collect(self, data_definitions):
        """
        Collects data specifically for a proxy host using SSH.

        Args:
            data_definitions (list): List of data definitions [{'description': ..., 'command': ...}]
                                     specific to the 'proxy' role from config.yaml.

        Returns:
            list: List of result dictionaries: [{'description': ..., 'command': ...,
                   'status': ..., 'output': ..., 'error': ...}].
        """
        results = []
        role = "proxy" # Define role for logging clarity
        logging.info(f"Starting data collection for {role} host: {self.hostname}")

        if not data_definitions:
            logging.warning(f"No data definitions provided for role '{role}' on host {self.hostname}")
            return results

        # Extract SSH parameters once
        ssh_user = self.get_ssh_param('username')
        ssh_key = self.get_ssh_param('private_key_path')
        ssh_port = self.get_ssh_param('port', 22)
        conn_timeout = self.get_ssh_param('connection_timeout', 10)
        cmd_timeout = self.get_ssh_param('command_timeout', 30)

        if not ssh_user or not ssh_key:
             logging.error(f"SSH username or key path missing in config for host {self.hostname}")
             # Return results collected so far, or potentially raise an error
             return results # Or raise ConfigurationError("Missing SSH credentials")

        for item in data_definitions:
            description = item.get('description', 'No description')
            command = item.get('command')

            if not command:
                logging.warning(f"Skipping item '{description}' for {self.hostname} (role: {role}) due to missing command.")
                results.append({
                    "description": description, "command": None, "status": -100,
                    "output": None, "error": "Command missing in config",
                })
                continue

            logging.debug(f"Executing command for '{description}' on {self.hostname}: `{command}`")
            output, error, status = self.ssh_executor(
                hostname=self.hostname,
                username=ssh_user,
                private_key_path=ssh_key,
                command=command,
                port=ssh_port,
                conn_timeout=conn_timeout,
                cmd_timeout=cmd_timeout
            )

            results.append({
                "description": description,
                "command": command,
                "status": status,
                "output": output,
                "error": error,
            })

        logging.info(f"Finished collection for {role} host: {self.hostname}. Collected {len(results)} items.")
        return results

    # --- Optional proxy-specific helper methods ---
    # def _parse_specific_proxy_log(self, log_output):
    #     # Add custom parsing logic here if needed
    #     pass

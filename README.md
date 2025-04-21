# FailTale: Tool to collect context from failed tests

This project is a Proof of Concept for implementing a local server inspired by the Model Context Protocol (MCP). Its goal is to automate the collection of relevant debugging data from multiple components part of the product architecture when an automated test fails.

It optionally uses a local LLM (Ollama) to analyze the collected data and provide insights into the root cause.

**Key Features:**

* **Failure-Triggered Activation:** Integrates with test frameworks to activate only when a test fails.
* **Targeted Collection:** Uses a configuration file (`config.yaml`) to define which data is useful for each product component.
* **Secure SSH:** Connects to hosts via SSH using keys and minimally privileged (read-only) users.
* **On-Source Filtering:** Applies filters (`tail`, `grep`) directly in the remote commands to reduce data volume.
* **MCP Interface (Conceptual):** Exposes functionalities via a local API (to be defined according to the actual MCP specification).
* **Local LLM Integration:** Interacts with Ollama for optional analysis.
* **Local-First:** Designed to operate offline.

## Architecture (Conceptual)

1.  **Test Framework Hook (MCP Client):** When a test fails, a hook calls the MCP client.
2.  **MCP Client:** Constructs an MCP request and sends it to the local MCP server.
3.  **MCP Server (This Project):**
    * Receives the request.
    * Consults `config.yaml`.
    * Uses `ssh_executor` to connect to hosts and run `collectors` commands.
    * (Optional) Uses `llm_interaction` for analysis.
    * Returns a formatted response (according to MCP) to the client.
4.  **MCP Client:** Receives the response and injects it into the test report.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd your-repo
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Ollama:** Ensure Ollama is installed, running, and has a model downloaded (e.g., `ollama pull llama3`).
5.  **Configure SSH Access:**
    * Generate an SSH key pair if you don't have one (`ssh-keygen`).
    * Create a **minimally privileged** user on each target host (server, minion, proxy).
    * Add your public key (`~/.ssh/id_rsa.pub`) to the `~/.ssh/authorized_keys` file of the limited user on each target host.
    * Verify connectivity manually: `ssh -i /path/to/private_key limited_user@target_host 'echo hello'`
6.  **Create Configuration File:**
    * Copy the template: `cp environments/config.yaml.template environments/myenv.yaml`
    * Edit `environments/myenv.yaml` with your host details, commands, SSH user, and private key path. **Do not commit this file if it contains sensitive information!**

## Usage

1.  **Start the MCP Server:**
    ```bash
    python run_server.py --config environments/myenv.yaml
    ```
    The server will start listening on a local address (e.g., `http://localhost:5050`).

2.  **Integrate the Client:** Adapt the code in `examples/pytest_integration/` (or create one for your framework) so the failure hook calls your local MCP server endpoints.

## Development

* **Add Collectors:** Implement specific logic in `mcp_server_poc/collectors/`.
* **Define MCP Endpoints:** Develop endpoints in `mcp_server_poc/server.py` based on the **official MCP documentation**.
* **Write Tests:** Add unit and integration tests in the `tests/` directory. Use `pytest` to run them:
    ```bash
    pytest
    ```

## TODO / Next Steps

* [ ] Define the endpoints/methods in `server.py` according to the MCP spec.
* [ ] Implement detailed logic for the `collectors`.
* [ ] Implement a robust `ssh_executor` with error handling.
* [ ] Implement Ollama interaction (`llm_interaction`).
* [ ] Write thorough tests.
* [ ] Create a working example MCP client (`examples/`).

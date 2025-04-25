# FailTale: Tool to collect context from failed tests

This project is a Proof of Concept for implementing a Test Reviewer. Its goal is to automate the collection of relevant debugging data from multiple components part of the product when an automated test fails, then analyze the collected data and provide insights into the root cause.

**Key Features:**

* **Targeted Collection:** Uses a configuration file (`config.yaml`) to define which data is useful for each product component.
* **Secure SSH:** Connects to hosts via SSH using keys and minimally privileged (read-only) users.
* **On-Source Filtering:** Applies filters (`tail`, `grep`) directly in the remote commands to reduce data volume.
* **API Interface:** Exposes functionalities via a local API.
* **Local LLM Integration:** Interacts with Ollama in two phases: target components to debug and analysis of test failure.

## Architecture

2.  **Client:** Constructs a request and sends it to the local server.
3.  **Server:**
    * Receives the request.
    * Consults `config.yaml`.
    * Uses `llm_interaction` to get a list of hosts based on the test failure.
    * Uses `ssh_executor` to connect to hosts and run `collectors` commands.
    * Uses `llm_interaction` to analyse the test failure and return root cause hints.
    * Returns a formatted response to the client.
4.  **Client:** Receives the response and injects it into the test report.

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
4.  **Configure Ollama:** Ensure Ollama is installed, running, and has a model downloaded (e.g., `ollama pull granite3-dense:8b`).
5.  **Configure SSH Access:**
    * Generate an SSH key pair if you don't have one (`ssh-keygen`).
    * Create a **minimally privileged** user on each target host (server, minion, proxy).
    * Add your public key (`~/.ssh/id_rsa.pub`) to the `~/.ssh/authorized_keys` file of the limited user on each target host.
    * Verify connectivity manually: `ssh -i /path/to/private_key limited_user@target_host 'echo hello'`
6.  **Create Configuration File:**
    * Copy the template: `cp templates/config.yaml.template environments/myenv.yaml`
    * Edit `environments/myenv.yaml` with your host details, commands, SSH user, and private key path. **Do not commit this file if it contains sensitive information!**

## Usage

1.  **Start the Server:**
    ```bash
    python run_server.py --config examples/uyuni/uyuni.yaml
    ```
    The server will start listening on a local address (e.g., `http://localhost:5050`).

2.  **Integrate the Client:** Adapt the code in `run_client.py` (or create one for your framework) so the failure hook calls your local  server endpoints.
    ```bash
    python run_client.py --config examples/uyuni/uyuni.yaml --test-report examples/uyuni/test_report.txt --test-failure examples/uyuni/test_failure.txt
    ```

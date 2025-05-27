![fail-tale-image](https://github.com/user-attachments/assets/4d5f62f3-32bb-4c52-8258-7e802e190072)

# FailTale: Tool to collect context from failed tests

This project is a Proof of Concept for implementing a Test Reviewer. Its goal is to automate the collection of relevant debugging data from multiple components part of the product when an automated test fails, then analyze the collected data and provide insights into the root cause.

**Key Features:**

* **Targeted Collection:** Uses a configuration file (`config.yaml`) to define which data is useful for each product component.
* **Secure SSH:** Connects to hosts via SSH using keys and minimally privileged (read-only) users.
* **On-Source Filtering:** Applies filters (`tail`, `grep`) directly in the remote commands to reduce data volume.
* **API Interface:** Exposes functionalities via a local API.
* **Local LLM Integration:** Interacts with Ollama in two phases: target components to debug and analysis of test failure.

## Architecture

1. **Client:** Constructs a request and sends it to the local server.
2. **Server:**
    * Receives the request.
    * Consults `config.yaml`.
    * Uses `llm_interaction` to get a list of hosts based on the test failure.
    * Uses `ssh_executor` to connect to hosts and collect data.
    * Uses `llm_interaction` to analyse the test failure and return root cause hints.
    * Returns a formatted response to the client.
3. **Client:** Receives the response and injects it into the test report.

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
4.  **Configure Ollama:** Ensure Ollama is installed, running, and has a model downloaded (e.g., `ollama pull mistral`).
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

2.  **Integrate the Client:** Adapt the code in `run_client.py` (or create one for your framework) so the failure hook calls your local server endpoints.
    ```bash
    python run_client.py --config examples/uyuni/uyuni.yaml --test-report examples/uyuni/test_report.txt --test-failure examples/uyuni/test_failure.txt
    ```

### How to Use the Docker Setup (FailTale Server container)

This guide explains how to build and run the Docker container that includes the FailTale Flask server.

#### Prerequisites

1.  **Docker:** Ensure Docker Desktop or Docker Engine is installed and running on your system.
2.  **FailTale Project:** You have the `FailTale` project code locally, including:
    * `Dockerfile` (as provided previously)
    * `start.sh` (as provided previously)
    * `requirements.txt` (listing all Python dependencies like `Flask`, `asyncssh`, `PyYAML`, `requests`, `httpx`, etc.)
    * `config.yaml` (configured for the server, especially the `ollama.base_url` should be `http://localhost:11434`)
    * Your Python source code (`server/`, `run_server.py`, `run_client.py`, etc.)
3.  **SSH Key:** You have an SSH private key (e.g., `~/.ssh/id_rsa`) that has access to the external hosts your `ssh_executor.py` needs to connect to.
4.  **Python Environment (Host):** You need a Python environment on your *host* machine (outside the container) to run the `run_client.py` script. Make sure the necessary client dependencies (like `requests`, `PyYAML`) are installed there.

#### Steps

1.  **Place Files:**
    * Ensure the `Dockerfile` and `start.sh` files are located in the root directory of your local `FailTale` project clone.
    * Verify your `requirements.txt` is up-to-date in the project root.
    * Verify your `config.yaml` exists and is configured correctly (especially `ollama.base_url: "http://localhost:11434"` and `ssh_defaults`).

2.  **Optionally, build the Docker Image:**
    * Open your terminal or command prompt.
    * Navigate (`cd`) to the root directory of your `FailTale` project (where the `Dockerfile` is).
    * Run the build command:
        ```bash
        docker build -t failtale .
        ```
        * `-t failtale`: Tags the image with the name `failtale`. You can choose a different name.
        * `.`: Specifies the current directory as the build context.
        * *Note:* This build process will take some time, especially the first time, as it downloads the base image, installs dependencies, and pulls the Ollama models (`mistral`, `nomic-embed-text`).

3.  **Run the Docker Container:**
    * Execute the following command in your terminal:
        ```bash
        docker run -d --rm \
          --net=host \
          -v ~/.ssh/id_rsa:/root/.ssh/id_rsa:ro \
          -v ./examples/uyuni/uyuni.yaml:/app/config.yaml:ro \
          --name failtale_service \
          ghcr.io/srbarrios/failtale # Replace it with failtale if you built it locally
        ```
    * **Explanation:**
        * `-d`: Runs the container in detached mode (background).
        * `--rm`: Automatically removes the container when it stops.
        * `--net=host`: Uses the host network stack, allowing the container to access the host's network directly.
        * `-p 11434:11434`: Maps port 11434 (Ollama) inside the container to port 11434 on your host.
        * `-p 5050:5050`: Maps port 5050 (Flask server) inside the container to port 5050 on your host.
        * `-v ~/.ssh/id_rsa:/root/.ssh/id_rsa:ro`: **Crucial step.** Mounts your local private SSH key into the container at `/root/.ssh/id_rsa` in read-only mode (`:ro`).
            * **Important:** Replace `~/.ssh/id_rsa` with the actual path to *your* private key file if it's different.
        * `--name failtale_service`: Assigns a convenient name to the running container.
        * `failtale`: Specifies the image to run.
    * **Optional Mounts (Add more `-v` flags if needed):**
        * Known Hosts: `-v ~/.ssh/known_hosts:/root/.ssh/known_hosts:ro` (if your executor uses it).
        * Local Config: `-v ./config.yaml:/app/config.yaml:ro` (if you want to override the config cloned from git with a local version).

4.  **Verify Container Startup:**
    * You can check the container logs to see if both Ollama and the Flask server started correctly:
        ```bash
        docker logs failtale_service
        ```
    * Look for output similar to "\[Startup\] Ollama service appears to be running." and Flask's startup messages.

5.  **Option A) Run the Client (from your Host Machine):**
    * Open a *new* terminal window on your host machine.
    * Activate the Python virtual environment where you have the client dependencies installed.
    * Run the `run_client.py` script, pointing it to the server running inside the container (`http://localhost:5050`):
        ```bash
        python run_client.py \
          --config path/to/your/env_config.yaml \
          --server-url http://localhost:5050 \
          --test-report path/to/your/report.txt \
          --test-failure path/to/your/failure.txt
        ```
        * Replace `path/to/your/env_config.yaml`, `path/to/your/report.txt`, and `path/to/your/failure.txt` with the actual paths on your host machine.
        * The client will connect to `localhost:5050`, which Docker maps to the Flask server inside the container. The Flask server will then connect to Ollama at `http://localhost:11434` (within the container network). The SSH executor inside the container will use the mounted key to connect to external hosts.

   6. **Option B) Use the AI Test Reviewer Ruby Class from your Test Framework:**
       * If you're using a Ruby test framework, you can initialize the ai_test_reviewer class to interact with the FailTale server:
       ```ruby
           require 'ai_test_reviewer'

           reviewer = AiTestReviewer.new(
             config_path: 'path/to/your/env_config.yaml',
             server_url: 'http://localhost:5050'
           )

         After do |scenario|
           if scenario.failed?
              attach_ai_root_cause_hint(scenario)
         end

         def attach_ai_root_cause_hint(scenario)
           file_path = scenario.location.file
           feature_text = File.read(file_path)
           collected_data = $ai_test_reviewer.collect(feature_text)
           if scenario.exception
           error_message = scenario.exception.message
           stacktrace = scenario.exception.backtrace.join("\n")
           test_failure = "#{error_message}\n#{stacktrace}"
           else
           test_failure = 'No failure message available.'
           end
           result = $ai_test_reviewer.analyze(collected_data, page.html, feature_text, test_failure)
           root_cause_hint = result.fetch('root_cause_hint', nil)&.to_s
           return if root_cause_hint.nil? || root_cause_hint.empty?
           
           attach root_cause_hint, 'text/plain'
           $stdout.puts root_cause_hint
         end
      ```
    
7. **Stopping the Container:**
    * When you're finished, you can stop the container using its name:
        ```bash
        docker stop failtale_service
        ```
      (Since `--rm` was used, stopping it will also remove it).

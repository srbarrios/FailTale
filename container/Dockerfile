FROM python:3.11-slim-bookworm
LABEL maintainer="Oscar Barrios <srbarrios@gmail.com>"
LABEL description="Runs Ollama (llama3, nomic-embed-text) and the FailTale server."

# Set environment variables to prevent interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ollama serve & \
    OLLAMA_PID=$! && \
    sleep 10 && \
    ollama pull llama3 && \
    ollama pull nomic-embed-text && \
    kill $OLLAMA_PID && \
    sleep 2

RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    touch /root/.ssh/known_hosts && \
    chmod 600 /root/.ssh/known_hosts

WORKDIR /app

# Avoid docker cache issues when updating the repo
ADD https://api.github.com/repos/srbarrios/FailTale/git/refs/heads/main /tmp/version.json

RUN git clone https://github.com/srbarrios/FailTale.git . && \
    rm -rf ./examples

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the startup script into the container
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 5050

ENTRYPOINT ["/app/start.sh"]

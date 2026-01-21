#!/bin/bash
set -eo pipefail

echo "Starting OpenHands with Claude Code CLI transport..."

# Verify Claude CLI is available
if ! command -v claude &> /dev/null; then
    echo "ERROR: Claude Code CLI not found. Please ensure it's installed."
    exit 1
fi

echo "Claude Code CLI version: $(claude --version)"

# Check for OAuth token
if [ -z "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
    echo "WARNING: CLAUDE_CODE_OAUTH_TOKEN is not set."
    echo "The Claude CLI transport requires a valid OAuth token."
    echo "Get one with: claude setup-token"
fi

if [[ $NO_SETUP == "true" ]]; then
  echo "Skipping setup, running as $(whoami)"
  "$@"
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "The OpenHands entrypoint.sh must run as root"
  exit 1
fi

if [ -z "$SANDBOX_USER_ID" ]; then
  echo "SANDBOX_USER_ID is not set"
  exit 1
fi

if [ -z "$WORKSPACE_MOUNT_PATH" ]; then
  # This is set to /opt/workspace in the Dockerfile. But if the user isn't mounting, we want to unset it so that OpenHands doesn't mount at all
  unset WORKSPACE_BASE
fi

if [[ "$INSTALL_THIRD_PARTY_RUNTIMES" == "true" ]]; then
  echo "Downloading and installing third_party_runtimes..."
  echo "Warning: Third-party runtimes are provided as-is, not actively supported and may be removed in future releases."

  if pip install 'openhands-ai[third_party_runtimes]' -qqq 2> >(tee /dev/stderr); then
    echo "third_party_runtimes installed successfully."
  else
    echo "Failed to install third_party_runtimes." >&2
    exit 1
  fi
fi

if [[ "$SANDBOX_USER_ID" -eq 0 ]]; then
  echo "Running OpenHands as root"
  export RUN_AS_OPENHANDS=false
  "$@"
else
  echo "Setting up enduser with id $SANDBOX_USER_ID"
  if id "enduser" &>/dev/null; then
    echo "User enduser already exists. Skipping creation."
  else
    if ! useradd -l -m -u $SANDBOX_USER_ID -s /bin/bash enduser; then
      echo "Failed to create user enduser with id $SANDBOX_USER_ID. Moving openhands user."
      incremented_id=$(($SANDBOX_USER_ID + 1))
      usermod -u $incremented_id openhands
      if ! useradd -l -m -u $SANDBOX_USER_ID -s /bin/bash enduser; then
        echo "Failed to create user enduser with id $SANDBOX_USER_ID for a second time. Exiting."
        exit 1
      fi
    fi
  fi
  usermod -aG openhands enduser

  # Handle Docker socket if available
  if [ -e /var/run/docker.sock ]; then
    DOCKER_SOCKET_GID=$(stat -c '%g' /var/run/docker.sock)
    echo "Docker socket group id: $DOCKER_SOCKET_GID"
    if getent group $DOCKER_SOCKET_GID; then
      echo "Group with id $DOCKER_SOCKET_GID already exists"
    else
      echo "Creating group with id $DOCKER_SOCKET_GID"
      groupadd -g $DOCKER_SOCKET_GID docker
    fi
    usermod -aG $DOCKER_SOCKET_GID enduser
  fi

  mkdir -p /home/enduser/.cache/huggingface/hub/

  echo "Running as enduser"
  su enduser /bin/bash -c "${*@Q}" # This magically runs any arguments passed to the script as a command
fi

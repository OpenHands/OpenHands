#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

cd "$REPO_ROOT" || exit 1

# Install uv if not present (required for running agent-server via uvx)
if ! command -v uvx &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Add uv to PATH for the rest of this script
  export PATH="$HOME/.local/bin:$PATH"

  # Verify installation
  if ! command -v uvx &> /dev/null; then
    echo "Warning: uv installed but uvx not found in PATH."
    echo "You may need to add ~/.local/bin to your PATH:"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
  fi
fi

# Always run npm ci to ensure dependencies are installed and up-to-date.
# This is idempotent and ensures hooks (like on_stop.sh) have access to
# npm scripts like lint and test.
npm ci

if [ ! -f "$ENV_FILE" ]; then
  cp .env.sample "$ENV_FILE"
fi

if ! grep -Eq '^[[:space:]]*VITE_WORKING_DIR=' "$ENV_FILE"; then
  printf '\nVITE_WORKING_DIR="%s"\n' "$REPO_ROOT" >> "$ENV_FILE"
fi

if [ ! -f src/i18n/declaration.ts ]; then
  npm run make-i18n
fi

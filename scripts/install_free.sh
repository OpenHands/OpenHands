#!/usr/bin/env bash
set -euo pipefail

echo "Public free installer: installs backend and frontend for local development (non-Docker)."

# Check Node
if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: Node.js is not installed. Please install Node.js 22.x or later: https://nodejs.org/"
  exit 1
fi

# Check npm
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm is not installed. Please install Node.js which includes npm."
  exit 1
fi

# Check Python
PYTHON_CANDIDATES=(python3.13 python3.12 python3)
PYTHON=""
for cmd in "${PYTHON_CANDIDATES[@]}"; do
  if command -v "$cmd" >/dev/null 2>&1; then
    PYTHON="$cmd"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: Python 3.12 or 3.13 is required. Please install Python."
  exit 1
fi

# Check Poetry (optional fallback to pip)
USE_POETRY=0
if command -v poetry >/dev/null 2>&1; then
  USE_POETRY=1
  echo "Using Poetry for Python dependency management."
else
  echo "Poetry not found. The installer will try to use pip in a virtualenv as fallback."
fi

ROOT_DIR=$(pwd)

# Backend: install Python deps
if [ -f "pyproject.toml" ] || [ -d "dev_config" ]; then
  echo "Installing backend Python dependencies..."
  if [ "$USE_POETRY" -eq 1 ]; then
    poetry install --with dev,test,runtime
  else
    # Create venv
    if [ ! -d ".venv" ]; then
      echo "Creating virtualenv..."
      "$PYTHON" -m venv .venv
    fi
    . .venv/bin/activate
    if [ -f "requirements.txt" ]; then
      pip install -r requirements.txt
    elif [ -f "pyproject.toml" ]; then
      echo "pyproject.toml found but poetry is not available. Please install poetry or use make install-free with poetry available."
    fi
  fi
else
  echo "No backend Python project files detected; skipping backend install."
fi

# Frontend: install Node deps and build
if [ -d "frontend" ]; then
  echo "Installing frontend dependencies (frontend)..."
  cd frontend
  npm ci --silent
  echo "Building frontend..."
  npm run build --silent || { echo "Frontend build failed. Try 'cd frontend && npm run dev' for dev mode."; }
  cd "$ROOT_DIR"
else
  echo "No frontend directory found; skipping frontend install."
fi

cat <<'EOF'

Installation complete.
To run locally (non-Docker):
  make run
Or start frontend in dev mode:
  cd frontend && npm run dev

Please check README.md for additional notes.
EOF

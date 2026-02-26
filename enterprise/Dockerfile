ARG OPENHANDS_VERSION=latest
ARG BASE="ghcr.io/openhands/openhands"
FROM ${BASE}:${OPENHANDS_VERSION}

# Datadog labels
LABEL com.datadoghq.tags.service="deploy"
LABEL com.datadoghq.tags.env="${DD_ENV}"

# Install Node.js v20+ and npm (which includes npx)
# Apply security updates to fix CVEs
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get install -y jq gettext && \
    # Apply security updates for packages with available fixes
    apt-get upgrade -y \
        libc-bin \
        libc6 \
        libgnutls30 \
        libsqlite3-0 \
        perl-base && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install poetry and export before importing current code.
RUN /app/.venv/bin/pip install poetry poetry-plugin-export

# Install Python dependencies from poetry.lock for reproducible builds
# Copy lock files first for better Docker layer caching
COPY --chown=openhands:openhands enterprise/pyproject.toml enterprise/poetry.lock /tmp/enterprise/
RUN cd /tmp/enterprise && \
    # Export only main dependencies with hashes for supply chain security
    /app/.venv/bin/poetry export --only main -o requirements.txt && \
    # Remove the local path dependency (openhands-ai is already in base image)
    sed -i '/^-e /d; /openhands-ai/d' requirements.txt && \
    # Install pinned dependencies from lock file
    /app/.venv/bin/pip install -r requirements.txt && \
    # Cleanup - return to /app before removing /tmp/enterprise
    cd /app && \
    rm -rf /tmp/enterprise && \
    /app/.venv/bin/pip uninstall -y poetry poetry-plugin-export

WORKDIR /app
COPY --chown=openhands:openhands --chmod=770 enterprise .

USER openhands

# Command will be overridden by Kubernetes deployment template
CMD ["uvicorn", "saas_server:app", "--host", "0.0.0.0", "--port", "3000"]

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

# Install Python packages with security fixes
RUN pip install alembic psycopg2-binary cloud-sql-python-connector pg8000 gspread stripe python-keycloak asyncpg sqlalchemy[asyncio] resend tenacity slack-sdk ddtrace "posthog>=6.0.0" "limits==5.2.0" coredis prometheus-client shap scikit-learn pandas numpy && \
    # Update packages with known CVE fixes
    pip install --upgrade \
        "mcp>=1.10.0" \
        "pillow>=11.3.0"

WORKDIR /app
COPY enterprise .

RUN chown -R openhands:openhands /app && chmod -R 770 /app
USER openhands

# Command will be overridden by Kubernetes deployment template
CMD ["uvicorn", "saas_server:app", "--host", "0.0.0.0", "--port", "3000"]

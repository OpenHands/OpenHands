# Custom Agent Server Image with CORS support via NGINX Reverse Proxy
# Based on the official agent-server image with NGINX for CORS handling

FROM ghcr.io/openhands/agent-server:97652be-python

# Switch to root to modify files
USER root

# Upgrade SDK to latest version
RUN pip install --upgrade openhands-sdk || echo "⚠️  SDK upgrade failed, continuing..."

# Install NGINX and OpenSSL (for SSL certificate generation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx openssl && \
    rm -rf /var/lib/apt/lists/*

# Copy NGINX configuration and entrypoint script
COPY containers/agent-server/nginx.conf /etc/nginx/nginx.conf
COPY containers/agent-server/entrypoint.sh /tmp/custom-agent-server-entrypoint.sh
RUN chmod +x /tmp/custom-agent-server-entrypoint.sh && \
    mv /tmp/custom-agent-server-entrypoint.sh /usr/local/bin/custom-agent-server-entrypoint.sh

# Create directory for NGINX runtime files and SSL certificates
RUN mkdir -p /var/run/nginx && \
    mkdir -p /etc/nginx/ssl && \
    chown -R root:root /var/run/nginx && \
    chown -R root:root /etc/nginx

# Copy SDK patches to be applied at runtime
COPY sdk_patches/openhands/sdk/ /tmp/sdk_patches/openhands/sdk/
COPY containers/agent-server/apply_sdk_patches.sh /usr/local/bin/apply_sdk_patches.sh
RUN chmod +x /usr/local/bin/apply_sdk_patches.sh

# Set the entrypoint to our custom script
ENTRYPOINT ["/usr/local/bin/custom-agent-server-entrypoint.sh"]


# Always install a patched fn_call_converter.py
COPY sdk_patches/openhands/sdk/openhands/sdk/llm/mixins/fn_call_converter.py /usr/local/lib/python3.12/site-packages/openhands/sdk/llm/mixins/fn_call_converter.py

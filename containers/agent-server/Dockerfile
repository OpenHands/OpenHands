# Custom Agent Server Image with CORS support via NGINX Reverse Proxy
# Based on the official agent-server image with NGINX for CORS handling

FROM ghcr.io/openhands/agent-server:97652be-python

# Switch to root to modify files
USER root

# Install NGINX and OpenSSL (for SSL certificate generation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx openssl && \
    rm -rf /var/lib/apt/lists/*

# Copy NGINX configuration and entrypoint script
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /tmp/custom-agent-server-entrypoint.sh
RUN chmod +x /tmp/custom-agent-server-entrypoint.sh && \
    mv /tmp/custom-agent-server-entrypoint.sh /usr/local/bin/custom-agent-server-entrypoint.sh

# Create directory for NGINX runtime files and SSL certificates
RUN mkdir -p /var/run/nginx && \
    mkdir -p /etc/nginx/ssl && \
    chown -R root:root /var/run/nginx && \
    chown -R root:root /etc/nginx

# Set the entrypoint to our custom script
ENTRYPOINT ["/usr/local/bin/custom-agent-server-entrypoint.sh"]


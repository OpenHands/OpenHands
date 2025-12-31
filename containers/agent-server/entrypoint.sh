#!/bin/bash
# Custom entrypoint for agent-server with NGINX reverse proxy for CORS support

set -e

# Apply SDK patches BEFORE starting the server (to avoid module caching issues)
# This patches the files in site-packages before Python loads them
if [ -f "/usr/local/bin/apply_sdk_patches.sh" ]; then
    echo "🔧 Applying SDK patches (before agent-server starts)..."
    if /usr/local/bin/apply_sdk_patches.sh; then
        echo "✅ SDK patches applied successfully (before server start)"
        # Add a delay to ensure patches are fully written and any file system sync completes
        sleep 3
        echo "✅ Waited 3 seconds for patches to settle"
    else
        echo "⚠️  Failed to apply SDK patches, but continuing..."
    fi
else
    echo "⚠️  SDK patches script not found at /usr/local/bin/apply_sdk_patches.sh"
fi

# Agent-server will run on internal port 8002 (VSCode uses 8001, NGINX uses 8000)
# We ignore the command line arguments (--port 8000) and use our internal port
INTERNAL_PORT=8002

# Create SSL certificate directory
SSL_DIR=/etc/nginx/ssl
mkdir -p ${SSL_DIR}

# Generate self-signed SSL certificate if it doesn't exist
if [ ! -f ${SSL_DIR}/cert.pem ] || [ ! -f ${SSL_DIR}/key.pem ]; then
    echo "Generating self-signed SSL certificate..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ${SSL_DIR}/key.pem \
        -out ${SSL_DIR}/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
        2>/dev/null || {
        echo "WARNING: Failed to generate SSL certificate. HTTPS will not work."
        echo "Install openssl package if you need HTTPS support."
    }
    chmod 600 ${SSL_DIR}/key.pem
    chmod 644 ${SSL_DIR}/cert.pem
    echo "SSL certificate generated successfully."
fi

# Start agent-server on internal port 8002 in the background
# Ignore any command line arguments passed to the entrypoint
echo "Starting agent-server on internal port ${INTERNAL_PORT} (ignoring command line args)..."
/usr/local/bin/openhands-agent-server --port ${INTERNAL_PORT} &
AGENT_SERVER_PID=$!

# Wait for agent-server to be ready
echo "Waiting for agent-server to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:${INTERNAL_PORT}/health > /dev/null 2>&1; then
        echo "Agent-server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Agent-server failed to start"
        exit 1
    fi
    sleep 1
done

# Apply SDK patches AFTER agent-server is ready (SDK is now installed)
# This ensures patches are applied even if SDK wasn't available before
# CRITICAL: This is when _MEIPASS is created by PyInstaller, so we MUST patch it here
if [ -f "/usr/local/bin/apply_sdk_patches.sh" ]; then
    echo "🔧 Applying SDK patches (after agent-server started - _MEIPASS now exists)..."
    if /usr/local/bin/apply_sdk_patches.sh; then
        echo "✅ SDK patches applied successfully (after server start)"
        
        # Wait a moment for file system to sync
        sleep 1
        
        # Verify patches were applied by checking ALL possible locations
        echo "🔍 Verifying patches in all locations..."
        VERIFICATION_RESULT=$(python3 << 'PYEOF'
import sys
import importlib.util
import os
import glob

verified = []
not_found = []

# Method 1: Check loaded module
module_name = 'openhands.sdk.llm.mixins.fn_call_converter'
if module_name in sys.modules:
    module = sys.modules[module_name]
    if hasattr(module, '__file__') and module.__file__:
        file_path = module.__file__
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                if 'message.get("content")' in content:
                    verified.append(f"Loaded module: {file_path}")
                else:
                    not_found.append(f"Loaded module NOT patched: {file_path}")

# Method 2: Check _MEIPASS
if getattr(sys, 'frozen', False):
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        meipass_file = os.path.join(meipass, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
        if os.path.exists(meipass_file):
            with open(meipass_file, 'r') as f:
                content = f.read()
                if 'message.get("content")' in content:
                    verified.append(f"sys._MEIPASS: {meipass_file}")
                else:
                    not_found.append(f"sys._MEIPASS NOT patched: {meipass_file}")

# Method 3: Check /tmp/_MEI*
for meipass_dir in glob.glob('/tmp/_MEI*'):
    if os.path.isdir(meipass_dir):
        meipass_file = os.path.join(meipass_dir, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
        if os.path.exists(meipass_file):
            with open(meipass_file, 'r') as f:
                content = f.read()
                if 'message.get("content")' in content:
                    verified.append(f"/tmp/_MEI*: {meipass_file}")
                else:
                    not_found.append(f"/tmp/_MEI* NOT patched: {meipass_file}")

# Print results
if verified:
    print("✅ VERIFIED:")
    for v in verified:
        print(f"  {v}")
if not_found:
    print("⚠️  NOT PATCHED:")
    for nf in not_found:
        print(f"  {nf}")
PYEOF
)
        echo "$VERIFICATION_RESULT"
    else
        echo "⚠️  Failed to apply SDK patches, but continuing..."
    fi
else
    echo "⚠️  SDK patches script not found at /usr/local/bin/apply_sdk_patches.sh"
fi

# Generate NGINX config with CORS origins from environment variable
echo "Configuring NGINX with CORS origins: ${PERMITTED_CORS_ORIGINS:-localhost/127.0.0.1}"

# Create CORS origin map
cat > /tmp/nginx-cors-map.conf <<EOF
    # CORS origin map - dynamically generated from PERMITTED_CORS_ORIGINS
    map \$http_origin \$cors_origin {
        default "";
        # Always allow localhost and 127.0.0.1
        ~^http://localhost:[0-9]+\$ \$http_origin;
        ~^http://127\.0\.0\.1:[0-9]+\$ \$http_origin;
EOF

# Add each permitted origin to the map
if [ -n "$PERMITTED_CORS_ORIGINS" ]; then
    IFS=',' read -ra ORIGINS <<< "$PERMITTED_CORS_ORIGINS"
    for origin in "${ORIGINS[@]}"; do
        origin=$(echo "$origin" | xargs)  # trim whitespace
        if [ -n "$origin" ]; then
            # Escape special characters for regex (escape . but keep : and / as-is for URL matching)
            # Only escape dots, not colons or slashes
            escaped_origin=$(echo "$origin" | sed 's/\./\\./g')
            echo "        ~^${escaped_origin}\$ \$http_origin;" >> /tmp/nginx-cors-map.conf
        fi
    done
fi

echo "    }" >> /tmp/nginx-cors-map.conf

# Replace the existing CORS map in nginx.conf instead of adding a new one
# First, remove the old map block (from "map $http_origin $cors_origin {" to the closing "}")
sed -i '/map \$http_origin \$cors_origin {/,/^    }/d' /etc/nginx/nginx.conf

# Then insert the new map after the http { line
sed -i '/^http {/r /tmp/nginx-cors-map.conf' /etc/nginx/nginx.conf

# Check for VSCode static files
VSCODE_STATIC_DIR=/openhands/.openvscode-server/static
if [ -d "${VSCODE_STATIC_DIR}" ]; then
    echo "VSCode static directory found: ${VSCODE_STATIC_DIR}"
    # Check for common missing files
    if [ ! -f "${VSCODE_STATIC_DIR}/node_modules/vsda/rust/web/vsda_bg.wasm" ]; then
        echo "WARNING: vsda_bg.wasm not found in VSCode static directory"
    fi
    if [ ! -f "${VSCODE_STATIC_DIR}/node_modules/vsda/rust/web/vsda.js" ]; then
        echo "WARNING: vsda.js not found in VSCode static directory"
    fi
else
    echo "WARNING: VSCode static directory not found: ${VSCODE_STATIC_DIR}"
fi

# Test NGINX configuration
echo "Testing NGINX configuration..."
nginx -t

# Start NGINX in foreground
echo "Starting NGINX on port 8000 (HTTP) and 8443 (HTTPS)..."
exec nginx -g "daemon off;"

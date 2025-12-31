#!/bin/bash
# Script to mount SDK patches into agent-server containers
# This should be called when agent-server containers are created

CONTAINER_ID=$1
SDK_PATCHES_DIR="/home/noya/OpenHands/sdk_patches/openhands/sdk"
CONTAINER_SDK_DIR="/usr/local/lib/python3.12/site-packages/openhands/sdk"

if [ -z "$CONTAINER_ID" ]; then
    echo "Usage: $0 <container_id>"
    exit 1
fi

# Check if container exists
if ! docker ps -a --format "{{.ID}}" | grep -q "^${CONTAINER_ID}$"; then
    echo "Container $CONTAINER_ID not found"
    exit 1
fi

# Copy SDK patches into container
echo "Copying SDK patches to container $CONTAINER_ID..."
docker cp "$SDK_PATCHES_DIR" "$CONTAINER_ID:$(dirname $CONTAINER_SDK_DIR)/openhands/"
echo "✅ SDK patches copied"

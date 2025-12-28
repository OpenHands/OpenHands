#!/bin/bash
set -euo pipefail

# Check if DEPLOY_DIR argument was provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <DEPLOY_DIR>"
  echo "Example: $0 /path/to/deploy"
  exit 1
fi

# Normalize path (remove trailing slash)
DEPLOY_DIR="${DEPLOY_DIR%/}"

# Function to decrypt and rename
decrypt_and_move() {
  local secret_path="$1"
  local output_name="$2"

  ${DEPLOY_DIR}/scripts/decrypt.sh "${DEPLOY_DIR}/${secret_path}"
  mv decrypted.yaml "${output_name}"
  echo "Moved decrypted.yaml to ${output_name}"
}

# Decrypt each secret file
decrypt_and_move "openhands/envs/feature/secrets/github-app.yaml" "github_decrypted.yaml"
decrypt_and_move "openhands/envs/staging/secrets/keycloak-realm.yaml" "keycloak_realm_decrypted.yaml"
decrypt_and_move "openhands/envs/staging/secrets/keycloak-admin.yaml" "keycloak_admin_decrypted.yaml"

#!/bin/bash
# Installs the private host only. Does not fetch secrets or start an open app.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io docker-compose-v2
systemctl enable --now docker

install -d -m 0700 /srv/ruckus
device=/dev/disk/by-id/google-ruckus-state
for attempt in $(seq 1 30); do
  [ -b "$device" ] && break
  sleep 1
done
[ -b "$device" ] || { echo "Ruckus state disk is missing" >&2; exit 1; }
filesystem=$(blkid -s TYPE -o value "$device" || true)
if [ -z "$filesystem" ]; then
  # Only a completely blank new disk may be initialized.
  [ -z "$(wipefs --noheadings --output TYPE "$device")" ] || exit 1
  mkfs.ext4 -q "$device"
elif [ "$filesystem" != ext4 ]; then
  echo "Refusing to format a state disk with an existing filesystem" >&2
  exit 1
fi
install -d -m 0700 /srv/ruckus/state
mountpoint -q /srv/ruckus/state || mount "$device" /srv/ruckus/state
chmod 0700 /srv/ruckus/state

# Containers must not reach Google metadata credentials. The runtime service
# account has no OAuth scopes/project roles, as an independent protection.
iptables -C DOCKER-USER -d 169.254.0.0/16 -j DROP 2>/dev/null ||
  iptables -I DOCKER-USER -d 169.254.0.0/16 -j DROP
echo "Private host prepared. No application or employee access has been started."

#!/usr/bin/env bash
# Resolves the already-provisioned Nebius resources from §1 of MASTERCLASS.md
# (disk, subnet, instance) by name and persists their ids/ip as env vars, so
# VM_ID / VM_DISK_ID / SUBNET_ID / VM_IP survive a shell restart instead of
# only living in the export that created them.
#
# Prereqs: `nebius profile create` already run once (interactive, opens a
# browser) and `jq` installed.
#
# Usage: ./scripts/set_nebius_env.sh
set -euo pipefail

DISK_NAME="detox-hw-disk"
INSTANCE_NAME="dishant-ghai-detox-hw-vm-do-not-kill"
ENV_FILE="$HOME/.nebius-detox-hw.env"

command -v nebius >/dev/null 2>&1 || { echo "nebius CLI not found on PATH" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq not found — apt-get install -y jq" >&2; exit 1; }

if ! nebius profile list >/dev/null 2>&1; then
  echo "No Nebius profile configured. Run 'nebius profile create' (opens a browser) first." >&2
  exit 1
fi

VM_ID=$(nebius compute instance list --format json \
  | jq -r --arg n "$INSTANCE_NAME" '.items[] | select(.metadata.name == $n) | .metadata.id')

VM_DISK_ID=$(nebius compute disk list --format json \
  | jq -r --arg n "$DISK_NAME" '.items[] | select(.metadata.name == $n) | .metadata.id')

SUBNET_ID=$(nebius vpc subnet list --format json | jq -r '.items[0].metadata.id')

if [ -z "$VM_ID" ]; then
  echo "Could not find an instance named '$INSTANCE_NAME'. List instances with:" >&2
  echo "  nebius compute instance list --format json | jq -r '.items[].metadata.name'" >&2
  exit 1
fi

VM_IP=$(nebius compute instance get --id "$VM_ID" --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address | split("/")[0]')

for name in VM_ID VM_DISK_ID SUBNET_ID VM_IP; do
  if [ -z "${!name}" ]; then
    echo "Failed to resolve $name — got an empty value." >&2
    exit 1
  fi
done

{
  echo "export VM_ID=\"$VM_ID\""
  echo "export VM_DISK_ID=\"$VM_DISK_ID\""
  echo "export SUBNET_ID=\"$SUBNET_ID\""
  echo "export VM_IP=\"$VM_IP\""
} > "$ENV_FILE"

if ! grep -qxF "source $ENV_FILE" "$HOME/.bashrc" 2>/dev/null; then
  echo "source $ENV_FILE" >> "$HOME/.bashrc"
fi

echo "Wrote $ENV_FILE and sourced it from ~/.bashrc:"
cat "$ENV_FILE"

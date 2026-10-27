#!/usr/bin/env bash
# One-time on miciato: install the book-sync timer (checks Drive every 5 minutes).
#   git clone https://github.com/mvacaporale/book-companion /tmp/bc && bash /tmp/bc/deploy/miciato/install.sh
# Uses the same /root/.bookrc/mcp.env (pass:// refs) as the MCP server registration.
# Re-run only when the unit files change; code changes deploy on merge via uvx.
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
cp "$DIR/book-sync.service" "$DIR/book-sync.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now book-sync.timer
systemctl start --no-block book-sync.service  # first run now; logs: journalctl -u book-sync
systemctl list-timers book-sync.timer --no-pager

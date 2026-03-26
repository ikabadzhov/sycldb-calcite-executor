#!/bin/bash
set -euo pipefail

REMOTE_HOST="eugenio@devenv2"
REMOTE_ROOT="/tmp/tmp.5M8Fsnsuez/sycldb-calcite-executor"
REMOTE_CMD="${1:-make client}"

ssh "$REMOTE_HOST" "bash -lc 'source /home/eugenio/develop-env-vars.sh >/dev/null 2>&1 && cd \"$REMOTE_ROOT\" && $REMOTE_CMD'"

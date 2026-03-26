#!/bin/bash
set -euo pipefail

REMOTE_HOST="eugenio@devenv2"
REMOTE_ROOT="/tmp/tmp.5M8Fsnsuez/sycldb-calcite-executor"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <relative-path>..." >&2
    exit 1
fi

for REL_PATH in "$@"; do
    if [[ ! -e "$REL_PATH" ]]; then
        echo "missing path: $REL_PATH" >&2
        exit 1
    fi

    REMOTE_PATH="$REMOTE_ROOT/$REL_PATH"
    REMOTE_DIR="$(dirname "$REMOTE_PATH")"
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"

    if [[ -d "$REL_PATH" ]]; then
        echo "directory sync is not supported by this helper: $REL_PATH" >&2
        exit 1
    fi

    base64 < "$REL_PATH" | ssh "$REMOTE_HOST" "base64 -d > '$REMOTE_PATH'"
done

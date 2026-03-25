#!/bin/bash
set -euo pipefail

REMOTE_HOST="eugenio@devenv2"
REMOTE_EXECUTOR_DIR="/tmp/tmp.5M8Fsnsuez/sycldb-calcite-executor"
LOCAL_RESULTS_FILE="benchmark_results_final.devenv2.csv"

"$(dirname "$0")/remote_start_calcite.sh"

ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd \"$REMOTE_EXECUTOR_DIR\"
source /home/eugenio/develop-env-vars.sh >/dev/null 2>&1
pkill -f \"bash benchmark_all_devices.sh\" || true
pkill -f \"^\\./client queries/transformed/\" || true
rm -f benchmark_results_final.csv
make client
bash benchmark_all_devices.sh
'"

ssh "$REMOTE_HOST" "cat \"$REMOTE_EXECUTOR_DIR/benchmark_results_final.csv\"" > "$LOCAL_RESULTS_FILE"
echo "Copied remote benchmark CSV to $LOCAL_RESULTS_FILE"

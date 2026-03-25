#!/bin/bash
set -euo pipefail

REMOTE_HOST="eugenio@devenv2"
REMOTE_CALCITE_SRC="/media/ivan/SYCLDB/FULLENG/calcite-sycldb"
REMOTE_CALCITE_RUN="/tmp/calcite-sycldb-run"
REMOTE_GRADLE_HOME="/tmp/calcite-sycldb-gradle-home"
REMOTE_PROJECT_CACHE="/tmp/calcite-sycldb-project-cache"
REMOTE_LOG="/tmp/calcite-sycldb-run.log"

if ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail

if (echo > /dev/tcp/127.0.0.1/5555) >/dev/null 2>&1; then
    echo \"Calcite planner already listening on 127.0.0.1:5555\"
    exit 0
fi

if [ ! -d \"$REMOTE_CALCITE_RUN\" ]; then
    cp -a \"$REMOTE_CALCITE_SRC\" \"$REMOTE_CALCITE_RUN\"
    chmod -R u+w \"$REMOTE_CALCITE_RUN\"
fi

mkdir -p \"$REMOTE_GRADLE_HOME\" \"$REMOTE_PROJECT_CACHE\"
cd \"$REMOTE_CALCITE_RUN\"
nohup env GRADLE_USER_HOME=\"$REMOTE_GRADLE_HOME\" \
    ./gradlew --project-cache-dir \"$REMOTE_PROJECT_CACHE\" run \
    >\"$REMOTE_LOG\" 2>&1 </dev/null &

for _ in \$(seq 1 24); do
    if (echo > /dev/tcp/127.0.0.1/5555) >/dev/null 2>&1; then
        echo \"Calcite planner is listening on 127.0.0.1:5555\"
        exit 0
    fi
    sleep 5
done

echo \"Calcite planner failed to start\"
tail -n 80 \"$REMOTE_LOG\" || true
exit 1
'"; then
    exit 0
fi

if ssh "$REMOTE_HOST" 'if (echo > /dev/tcp/127.0.0.1/5555) >/dev/null 2>&1; then echo "Calcite planner already listening on 127.0.0.1:5555"; exit 0; fi; exit 1'; then
    exit 0
fi

exit 1

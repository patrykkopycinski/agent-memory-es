#!/bin/bash
# Machine-checked mutation proof for the tag filtering (filtering round).
#
# Replaces the old hand-rolled script, which had no verdict: a mutant that survived, or a pattern
# that no longer matched, still exited 0, and an interrupt left the tree mutated (review findings
# #1-#4, #6). All of that now lives in tests/tools/mutation_proof.py, which decides per mutant and
# exits non-zero unless every mutant is KILLED and every file is restored byte-identical.
#
# Usage: backend/tests/tools/rerank_mutation_proof.sh [--only M1,M5] [--timeout 600]
# The suite runs the backend tests inside the compose stack (VM override layered on the product
# compose). Override with MUT_SUITE_CMD for another runner.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
COMPOSE="$ROOT/backend/docker-compose.yml"
VM_OVERRIDE="${AMES_VM_OVERRIDE:-/opt/orca-base/work/ames-vm/docker-compose.vm.yml}"

if [ -z "${MUT_SUITE_CMD:-}" ]; then
  MUT_SUITE_CMD="sg docker -c \"docker compose -f $COMPOSE -f $VM_OVERRIDE run --rm --no-deps -v $ROOT:/srv -w /srv/backend ames-backend python -m pytest tests/test_tagfilter_pure.py tests/test_tag_filtering.py -q\""
fi

exec python3 "$HERE/mutation_proof.py" \
    --root "$ROOT" \
    --table "$HERE/filter_mutations.json" \
    --log-dir "${MUT_LOG_DIR:-/tmp/filter-mutation-logs}" \
    --command "$MUT_SUITE_CMD" "$@"

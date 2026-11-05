#!/usr/bin/env bash
# Host-side driver for scripts/test-remote.sh. Runs DETACHED on the docker host (bash 3.2-safe);
# do not call it by hand. State lives under RUN_DIR on the host, so a dropped ssh session or a
# killed local runner loses nothing: the local side re-attaches with `test-remote.sh --collect ID`.
#
#   RUN_DIR/env.sh      written by the local runner (ID, PORT, KEEP_ES, shas, ACTOR, PYARGS, ...)
#   RUN_DIR/head, base  git-archive trees of the two refs (removed when the driver finishes)
#   RUN_DIR/out/        junit-<label>.xml + <label>.log  (what the local side collects)
#   RUN_DIR/status      "running ..." while going, then "done head_rc=.. base_rc=.." or "infra <why>"
#
# On EXIT (any path, including SIGTERM) it removes the test containers, the ES container it
# created (unless --keep-es) and the shipped trees, journals, and releases the ops-claim.
set -uo pipefail
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH
RUN=${1:?usage: test-remote-host.sh RUN_DIR}
# shellcheck disable=SC1091
. "$RUN/env.sh"
OUT=$RUN/out
mkdir -p "$OUT"
ES="http://127.0.0.1:${PORT}"
ES_NAME="ames-review-es-${PORT}"
ES_IMAGE=${AMES_TEST_ES_IMAGE:-docker.elastic.co/elasticsearch/elasticsearch:9.6.0-SNAPSHOT}
MODEL_CACHE=${AMES_TEST_MODEL_CACHE:-$HOME/.cache/ames-test-models}
CONTAINER_MEM=${AMES_TEST_CONTAINER_MEM:-3g}
SUITE_TIMEOUT_S=${AMES_TEST_SUITE_TIMEOUT_S:-2400}
RESULT="infra driver exited early"
ES_CREATED=0

log() { echo "[$(TZ=Europe/Warsaw date +%H:%M:%S)] $*"; }
status() { echo "$*" > "$RUN/status"; }
claim_cmd() { OPS_ACTOR=$ACTOR "$HOME/bin/ops-claim" "$@" >/dev/null 2>&1 || true; }

finish() {
  local rc=$? idx
  trap - EXIT
  # xargs -r does not exist on macOS: test for emptiness first.
  docker ps -aq --filter "name=ames-rv-${ID}-" | while read -r ids; do docker rm -f "$ids" >/dev/null 2>&1; done
  if [ "$ES_CREATED" = 1 ] && [ "$KEEP_ES" != 1 ]; then
    docker rm -f "$ES_NAME" >/dev/null 2>&1
  elif curl -sf "$ES/_cluster/health" >/dev/null 2>&1; then
    # ES outlives this run (--keep-es / reused): drop only this run's indices.
    for idx in $(curl -s "$ES/_cat/indices/amtest_${ID}*?h=index" 2>/dev/null); do
      curl -s -o /dev/null -XDELETE "$ES/$idx"
    done
  fi
  rm -rf "$RUN/head" "$RUN/base"
  status "$RESULT"
  if [ "$KEEP_ES" = 1 ]; then
    claim_cmd journal "$RES" "run $ID finished; $ES_NAME kept on 127.0.0.1:$PORT (--keep-es), claim held to ttl"
  else
    claim_cmd journal "$RES" "run $ID finished and cleaned up: $RESULT"
    claim_cmd release "$RES"
  fi
  exit $rc
}
trap finish EXIT
trap 'RESULT="infra driver terminated"; exit 143' TERM INT

status "running starting-es"
# ---- 1. throwaway ES
if docker inspect "$ES_NAME" >/dev/null 2>&1; then
  log "reusing existing $ES_NAME (left by --keep-es)"
  [ "$(docker inspect -f '{{.State.Running}}' "$ES_NAME")" = true ] || docker start "$ES_NAME" >/dev/null
else
  log "starting $ES_NAME on 127.0.0.1:$PORT"
  docker run -d --name "$ES_NAME" -p "127.0.0.1:${PORT}:9200" \
    -e discovery.type=single-node -e xpack.security.enabled=false \
    -e ES_JAVA_OPTS="-Xms1g -Xmx1g" "$ES_IMAGE" >/dev/null || { RESULT="infra ES container failed to start"; exit 70; }
  ES_CREATED=1
fi
ok=0
for _ in $(seq 1 90); do
  curl -sf "$ES/_cluster/health?wait_for_status=yellow&timeout=2s" >/dev/null && { ok=1; break; }
  sleep 2
done
[ $ok = 1 ] || { RESULT="infra ES on $ES not healthy after 3 min"; exit 70; }
log "ES healthy: $ES"

# ---- 2. the prod backend image already carries every runtime dependency
BACKEND_IMAGE=$(docker inspect -f '{{.Config.Image}}' ames-backend 2>/dev/null)
[ -n "$BACKEND_IMAGE" ] || { RESULT="infra no ames-backend container to take the image from"; exit 70; }
log "image: $BACKEND_IMAGE"

# A missing parity model must FAIL the parity tests rather than skip them green.
REQUIRE_MODEL=0
MODEL_MOUNT=()
if [ -f "$MODEL_CACHE/multilingual-e5-small/REVISION" ] && [ -f "$MODEL_CACHE/bge-small-en-v1.5/REVISION" ]; then
  REQUIRE_MODEL=1
  MODEL_MOUNT=(-v "$MODEL_CACHE:/models:ro")
fi
log "AMES_REQUIRE_MODEL=$REQUIRE_MODEL"

# pytest path args that exist in this tree; bare options (-k expr, -x) pass through unchanged.
# Returns 1 (and sets NO_TARGETS) when every path arg is missing from the tree (e.g. a test file
# that only exists on head, run against base): the base run is then skipped, not mis-reported.
is_path_arg() { case "$1" in -*) return 1;; *.py|*.py::*|tests|tests/*) return 0;; *) return 1;; esac; }

run_suite() { # label sha
  local label=$1 sha=$2 name="ames-rv-${ID}-$1" pfx tree a p
  tree=$RUN/$label
  pfx="amtest_${ID}${label:0:1}_"
  ARGS=()
  local had_path=0 kept_path=0
  for a in ${PYARGS[@]+"${PYARGS[@]}"}; do
    if is_path_arg "$a"; then
      had_path=1; p=${a%%::*}
      if [ -e "$tree/backend/$p" ]; then ARGS+=("$a"); kept_path=1
      else log "  ($label tree has no $p; dropped from the $label run)"; fi
    else
      ARGS+=("$a")
    fi
  done
  if [ "$had_path" = 1 ] && [ "$kept_path" = 0 ]; then
    log "$label: none of the requested test paths exist at ${sha:0:7}; $label run skipped"
    echo "no requested test path exists at this ref" > "$OUT/$label.log"
    return 0
  fi
  [ "$had_path" = 1 ] || ARGS=(tests ${ARGS[@]+"${ARGS[@]}"})
  status "running $label"
  log "pytest $label @ ${sha:0:7} prefix=$pfx args: ${ARGS[*]}"
  # No --rm: the container is the durable record of the run until its log is copied out.
  docker run -d --name "$name" --memory "$CONTAINER_MEM" \
    --add-host host.docker.internal:host-gateway \
    -v "$tree:/repo" -v "$OUT:/out" ${MODEL_MOUNT[@]+"${MODEL_MOUNT[@]}"} -w /repo/backend \
    -e PYTHONDONTWRITEBYTECODE=1 -e AMES_INDEX_PREFIX="$pfx" -e AMES_EMBED_BACKEND=local \
    -e AMES_ES_URL="http://host.docker.internal:$PORT" -e FIX2_ES="http://host.docker.internal:$PORT" \
    -e AMES_HTTP_RETRY_BUDGET_S="${AMES_HTTP_RETRY_BUDGET_S:-2}" \
    -e AMES_REQUIRE_MODEL="$REQUIRE_MODEL" -e AMES_TEST_MODEL_CACHE=/models \
    "$BACKEND_IMAGE" sh -c 'pip install -q pytest >/dev/null 2>&1
      t=$1; shift
      exec timeout "$t" python -m pytest "$@"' sh \
    "$SUITE_TIMEOUT_S" "${ARGS[@]}" \
    -p no:cacheprovider -q -rfEs --continue-on-collection-errors "--junitxml=/out/junit-$label.xml" \
    >/dev/null || return 70
  local rc
  rc=$(docker wait "$name")
  docker logs "$name" > "$OUT/$label.log" 2>&1
  return "$rc"
}

rm -f "$OUT"/junit-*.xml
run_suite head "$HEAD_SHA"; HRC=$?
BRC=-
if [ "$BASE_SHA" != "-" ]; then
  run_suite base "$BASE_SHA"; BRC=$?
fi
RESULT="done head_rc=$HRC base_rc=$BRC"
log "$RESULT"

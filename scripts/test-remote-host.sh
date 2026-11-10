#!/usr/bin/env bash
# Host-side driver for scripts/test-remote.sh. Runs DETACHED on the docker host (bash 3.2-safe);
# do not call it by hand. State lives under RUN_DIR on the host, so a dropped ssh session or a
# killed local runner loses nothing: the local side re-attaches with `test-remote.sh --collect ID`.
#
#   RUN_DIR/env.sh      written by the local runner (ID, PORT, KEEP_ES, shas, ACTOR, PYARGS, ...)
#   RUN_DIR/head, base  git-archive trees of the two refs (removed when the driver finishes)
#   RUN_DIR/out/        junit-<label>.xml + <label>.log  (what the local side collects)
#   RUN_DIR/status      "running ..." while going, then "done head_rc=.. base_rc=.." or "infra <why>"
#   RUN_DIR/driver.pid  written by the launcher; the poll uses it to notice a SIGKILLed driver
#   ES container       labelled ames-test-run=<id> at creation: that label (not a marker file) says "this run owns it"
#
# On EXIT (any path, including SIGTERM) it removes the test containers, the ES container it
# created (unless --keep-es) and the shipped trees, and releases the host lock / external claim.
# `driver.sh --reap RUN_DIR` does the same cleanup (plus removing RUN_DIR) for a driver that was
# SIGKILLed / OOM-killed and so could not run its own trap; the local runner calls it on detection.
set -uo pipefail
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH
REAP=0
if [ "${1:-}" = "--reap" ]; then REAP=1; shift; fi
RUN=${1:?usage: test-remote-host.sh [--reap] RUN_DIR}
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
# This run owns the ES container iff docker run stamped it with our label (set at creation, so it survives a
# driver crash and can never match a container another run created under the same name).
es_owned() { [ -n "$(docker ps -aq --filter "name=^${ES_NAME}\$" --filter "label=ames-test-run=$ID" 2>/dev/null)" ]; }
ES_CREATED=0
es_owned && ES_CREATED=1

log() { echo "[$(date +%H:%M:%S)] $*"; }
status() { echo "$*" > "$RUN/status"; }
# optional external claim helper (subcommands journal|release); the lock dir is always ours to drop
claim_cmd() {
  local c=${CLAIM_CMD:-}
  case "$c" in \~/*) c=$HOME/${c#\~/};; esac
  [ -n "$c" ] && [ -x "$c" ] && OPS_ACTOR=$ACTOR "$c" "$@" >/dev/null 2>&1
  return 0
}
release_lock() {
  if [ -n "${LOCK_DIR:-}" ] && [ "$(cut -d' ' -f1 "$LOCK_DIR/owner" 2>/dev/null)" = "$ID" ]; then rm -rf "$LOCK_DIR"; fi
}

# Remove everything this run created on the host. Idempotent; safe on a half-built run.
cleanup() { # note
  local idx
  # xargs -r does not exist on macOS: test for emptiness first.
  docker ps -aq --filter "name=ames-rv-${ID}-" | while read -r ids; do docker rm -f "$ids" >/dev/null 2>&1; done
  if [ "$ES_CREATED" = 1 ] && [ "$KEEP_ES" != 1 ]; then
    docker rm -f "$ES_NAME" >/dev/null 2>&1
  elif curl -sf -m 5 "$ES/_cluster/health" >/dev/null 2>&1; then
    # ES outlives this run (--keep-es / reused / not ours): drop only this run's indices.
    # exactly this run's prefixes (amtest_<id>h_ / amtest_<id>b_): a bare amtest_<id>* would also match
    # another run whose id merely starts with ours
    for idx in $(curl -s -m 10 "$ES/_cat/indices/amtest_${ID}h_*,amtest_${ID}b_*?h=index" 2>/dev/null); do
      curl -s -m 10 -o /dev/null -XDELETE "$ES/$idx"
    done
  fi
  rm -rf "$RUN/head" "$RUN/base"
  if [ "$KEEP_ES" = 1 ] && [ "$ES_CREATED" = 1 ]; then
    claim_cmd journal "$RES" "run $ID finished; $ES_NAME kept on 127.0.0.1:$PORT (--keep-es), claim held to ttl"
  else
    claim_cmd journal "$RES" "run $ID: $1"
    claim_cmd release "$RES"
    release_lock
  fi
}

if [ "$REAP" = 1 ]; then
  # the driver is dead (SIGKILL / OOM / reboot): do its cleanup from the persisted state, then drop the run dir
  KEEP_ES=0
  echo "reaping run $ID: removing containers ames-rv-${ID}-*"
  cleanup "driver died; reaped (status was: $(cat "$RUN/status" 2>/dev/null))"
  [ "$ES_CREATED" = 1 ] && echo "removed ES container $ES_NAME (created by this run)" || echo "ES $ES_NAME not created by this run: dropped only its amtest_${ID}[hb]_* indices"
  echo "released lock/claim for $RES"
  rm -rf "$RUN"
  echo "removed $RUN"
  exit 0
fi

finish() {
  local rc=$?
  trap - EXIT
  cleanup "finished and cleaned up: $RESULT"
  status "$RESULT"
  exit $rc
}
trap finish EXIT
trap 'RESULT="infra driver terminated"; exit 143' TERM INT

# Reserved ports (a production AMES / shared ES may live there): refuse here too, not only in the launcher.
case "$PORT" in
  9200|9220|9268|9279) RESULT="infra refused: port $PORT is reserved"; exit 70;;
  ''|*[!0-9]*) RESULT="infra refused: bad port '$PORT'"; exit 70;;
esac

status "running starting-es"
# ---- 1. throwaway ES
if docker inspect "$ES_NAME" >/dev/null 2>&1; then
  log "reusing existing $ES_NAME (left by --keep-es)"
  if [ "$(docker inspect -f '{{.State.Running}}' "$ES_NAME")" != true ]; then
    docker start "$ES_NAME" >/dev/null || { RESULT="infra could not start existing $ES_NAME"; exit 70; }
  fi
else
  log "starting $ES_NAME on 127.0.0.1:$PORT"
  if ! docker run -d --name "$ES_NAME" --label "ames-test-run=$ID" -p "127.0.0.1:${PORT}:9200" \
    -e discovery.type=single-node -e xpack.security.enabled=false \
    -e ES_JAVA_OPTS="-Xms1g -Xmx1g" "$ES_IMAGE" >/dev/null; then
    # a failed `docker run` can leave a Created container behind: remove it only if it is OURS (label), never
    # a same-named container another run created
    es_owned && docker rm -f "$ES_NAME" >/dev/null 2>&1
    RESULT="infra ES container failed to start"; exit 70
  fi
  ES_CREATED=1
fi
ok=0
for _ in $(seq 1 90); do
  curl -sf "$ES/_cluster/health?wait_for_status=yellow&timeout=2s" >/dev/null && { ok=1; break; }
  sleep 2
done
[ $ok = 1 ] || { RESULT="infra ES on $ES not healthy after 3 min"; exit 70; }
log "ES healthy: $ES"
# RRF retrievers need a non-basic license (403 "license is non-compliant" otherwise). The ES is
# throwaway, so start its own 30-day trial; this touches nothing but this container.
curl -s -m 20 -XPOST "$ES/_license/start_trial?acknowledge=true" | tr -d '\n' | cut -c1-120 | sed 's/^/  license: /'
echo

# The built-in rerank endpoint downloads/deploys its model on first use (far slower than the backend's rerank
# timeout), which would hit whichever suite runs first (head). Warm it before any suite. A warm-up that
# never succeeds is only a WARNING: the rerank tests then fail honestly and the base comparison still applies.
# Each attempt is capped at 25 s (and by the budget left) so the TERM trap and `--abort` are never held up long.
status "running warming-rerank"
WARM_BUDGET_S=300
# The body is a mid-size rerank (10 passages of ~1.2 kB): a 2-passage call only deploys the model and the
# first realistic calls were still 35-45 s afterwards, while a 50-passage warm-up cost ~160 s per run.
WARM_DOC=$(printf 'quarterly warehouse relocation budget plan item %.0s' $(seq 1 25))
WARM_BODY='{"query":"warehouse relocation budget","input":['
for i in $(seq 1 10); do WARM_BODY="$WARM_BODY\"$WARM_DOC $i\","; done
WARM_BODY="${WARM_BODY%,}]}"
WARM_T0=$(date +%s); WARM_CODE=000; WARM_TIMES=; C1=0; WARM_OK=0; WARM_SETTLED=0
while :; do
  left=$(( WARM_BUDGET_S - ($(date +%s) - WARM_T0) ))
  [ "$left" -gt 0 ] || break
  C0=$(date +%s)
  WARM_CODE=$(curl -s -m $(( left < 25 ? left : 25 )) -o /dev/null -w '%{http_code}' -XPOST -H 'content-type: application/json' \
    "$ES/_inference/rerank/.rerank-v1-elasticsearch" -d "$WARM_BODY")
  C1=$(( $(date +%s) - C0 )); WARM_TIMES="$WARM_TIMES ${WARM_CODE}/${C1}s"
  if [ "$WARM_CODE" = 200 ]; then WARM_OK=$((WARM_OK + 1)); else WARM_OK=0; fi
  # settled = one fast 200, or two consecutive 200s however slow (a warm but slow host must not burn the budget)
  if [ "$WARM_OK" -ge 2 ] || { [ "$WARM_CODE" = 200 ] && [ "$C1" -le 10 ]; }; then WARM_SETTLED=1; break; fi
  # a permanent client error (e.g. 403 expired trial on a reused ES) will not heal: surface it now
  case "$WARM_CODE" in 400|401|403|404) break;; esac
  # after a client-side timeout the call may still be queued in ES: back off longer before adding load
  pause=5; [ "$WARM_CODE" = 000 ] && pause=15
  left=$(( WARM_BUDGET_S - ($(date +%s) - WARM_T0) ))
  [ "$left" -gt "$pause" ] || break
  sleep "$pause"
done
log "rerank warm-up calls (code/time):$WARM_TIMES"
if [ "$WARM_SETTLED" = 1 ] && [ "$C1" -le 10 ]; then
  log "rerank warm-up ok: HTTP $WARM_CODE after $(( $(date +%s) - WARM_T0 ))s"
elif [ "$WARM_SETTLED" = 1 ]; then
  log "rerank warm-up ok (slow: last call ${C1}s) after $(( $(date +%s) - WARM_T0 ))s"
else
  log "WARNING: rerank warm-up did not settle: last HTTP $WARM_CODE (${C1}s) after $(( $(date +%s) - WARM_T0 ))s; rerank tests may fail on a cold endpoint"
fi

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
  # Non-root: tests that chmod a dir unwritable (auth store, key store) only mean something when
  # the process is not root. HOME/PYTHONUSERBASE give `pip install --user` a writable place.
  docker run -d --name "$name" --memory "$CONTAINER_MEM" --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/home -e PYTHONUSERBASE=/tmp/pyuser -e PATH="/tmp/pyuser/bin:/usr/local/bin:/usr/bin:/bin" \
    --add-host host.docker.internal:host-gateway \
    -v "$tree:/repo" -v "$OUT:/out" ${MODEL_MOUNT[@]+"${MODEL_MOUNT[@]}"} -w /repo/backend \
    -e PYTHONDONTWRITEBYTECODE=1 -e AMES_INDEX_PREFIX="$pfx" -e AMES_EMBED_BACKEND=local \
    -e AMES_ES_URL="http://host.docker.internal:$PORT" -e FIX2_ES="http://host.docker.internal:$PORT" \
    -e AMES_HTTP_RETRY_BUDGET_S="${AMES_HTTP_RETRY_BUDGET_S:-2}" \
    -e AMES_REQUIRE_MODEL="$REQUIRE_MODEL" -e AMES_TEST_MODEL_CACHE=/models \
    "$BACKEND_IMAGE" sh -c 'mkdir -p /tmp/home; pip install -q --user pytest >/dev/null 2>&1
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

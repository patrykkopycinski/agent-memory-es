#!/usr/bin/env bash
# Run the full AMES backend pytest suite from this machine on a remote docker host (default m1max)
# against a THROWAWAY Elasticsearch, and gate on "no NEW failures vs a base ref".
#
#   scripts/test-remote.sh [--repo DIR] [--ref REF] [--base REF] [--host SSH] [--port N] [--keep-es] [-- pytest args]
#   scripts/test-remote.sh --collect RUN_ID      re-attach to a run (local runner died / ssh dropped)
#   scripts/test-remote.sh --abort RUN_ID        kill a run and clean the host
#
#   --repo  git checkout to archive refs from (default: the checkout this script lives in)
#   --ref   commit to test (default HEAD)             --base  also run BASE; exit 1 on NEW failures vs it
#   --host  ssh alias (default $AMES_TEST_HOST|m1max) --port  host-local ES port (default 9291)
#   --keep-es  leave the ES container (and the ops-claim) in place after the run
#   `-- ...` is passed to pytest; paths are relative to backend/ (e.g. `-- tests/test_tombstone.py -k reason`).
#
# How: the WHOLE repo at each ref is shipped with `git archive` (tests load ../../scripts/*). On the host
# a detached driver (scripts/test-remote-host.sh) starts a throwaway ES (docker, 127.0.0.1:PORT) and runs
# pytest in a container from the prod backend image (`docker inspect ames-backend`) with the tree mounted,
# AMES_EMBED_BACKEND=local and a per-run amtest_ index prefix. All state is on the host; this script only
# polls with short ssh calls, so a dropped ssh / killed runner loses nothing (see --collect).
# Host: takes ops-claim `<host>:ames-review-es`, refuses on low free memory, always cleans up on exit.
# The host is the PROD AMES host: ports 9200/9220/9268/9279 are refused; nothing here writes to prod ES.
#
# Exit: 0 green (no new failures with --base) | 1 failures / positive control failed | 64 refused (bad args,
#       reserved port) | 70 infra (ssh, docker, low memory, ES) | 75 host busy (claim held)
# Env:  AMES_TEST_MIN_FREE_MB (4096) / AMES_TEST_MIN_SWAP_FREE_MB (512) host floors; AMES_MIN_TESTS (50) positive control;
#       AMES_TEST_WAIT_S (6000) max time this script waits; AMES_TEST_SUITE_TIMEOUT_S (2400) per suite;
#       AMES_HTTP_RETRY_BUDGET_S (2); AMES_TEST_CONTAINER_MEM (3g); AMES_TEST_ES_IMAGE; AMES_TEST_MODEL_CACHE (host path)
# Logs: ~/.cache/ames-test-remote/<ref>-<time>/  (run.log, junit-*.xml, head.log, base.log)
set -uo pipefail

REPO="" REF=HEAD BASE="" HOST=${AMES_TEST_HOST:-m1max} PORT=9291 KEEP_ES=0 MODE=run RUN_ARG=""
PYARGS=()
usage() { sed -n 2,27p "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=${2:?--repo needs a value}; shift 2;;
    --ref) REF=${2:?--ref needs a value}; shift 2;;
    --base) BASE=${2:?--base needs a value}; shift 2;;
    --host) HOST=${2:?--host needs a value}; shift 2;;
    --port) PORT=${2:?--port needs a value}; shift 2;;
    --keep-es) KEEP_ES=1; shift;;
    --collect) MODE=collect; RUN_ARG=${2:?--collect needs a run id}; shift 2;;
    --abort) MODE=abort; RUN_ARG=${2:?--abort needs a run id}; shift 2;;
    -h|--help) usage; exit 0;;
    --) shift; PYARGS=("$@"); break;;
    *) echo "unknown argument: $1 (pytest args go after --)" >&2; exit 64;;
  esac
done
case "$PORT" in
  9200|9220|9268|9279) echo "REFUSED: port $PORT is reserved (prod 9268 / holdout / implementer ES)"; exit 64;;
  ''|*[!0-9]*) echo "REFUSED: bad port '$PORT'"; exit 64;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "${REPO:-$HERE}" rev-parse --show-toplevel) || { echo "not inside a git checkout"; exit 64; }
SSHOPT=(-o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3)
SSH=(ssh -n "${SSHOPT[@]}" "$HOST")     # no stdin: safe inside loops/pipes
SSHI=(ssh "${SSHOPT[@]}" "$HOST")       # reads stdin: for tar / env.sh uploads
RES="$HOST:ames-review-es"
MIN_TESTS=${AMES_MIN_TESTS:-50}
MIN_FREE_MB=${AMES_TEST_MIN_FREE_MB:-4096}
MIN_SWAP_FREE_MB=${AMES_TEST_MIN_SWAP_FREE_MB:-512}
WAIT_S=${AMES_TEST_WAIT_S:-6000}
CACHE=${AMES_TEST_LOG_DIR:-$HOME/.cache/ames-test-remote}
# shellcheck disable=SC2016  # expanded on the host, not here
PATHX='export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH'
hr() { "${SSH[@]}" "$PATHX; $1"; }   # one short remote command, docker/brew on PATH

ops_claim() { # args... ; ACTOR is recorded explicitly (cross-host claims have no pid anchor: TTL-only)
  "${SSH[@]}" "OPS_ACTOR='$ACTOR' \"\$HOME/bin/ops-claim\" $*"
}

indent() { printf '%s\n' "$1" | sed 's/^/  /'; }

# ---------- junit helpers (the XML is pytest's own output, fetched seconds ago) ----------
junit() { # mode xml ; modes: ran | skipped | colerrors | fails | skips
  python3 - "$1" "$2" <<'PY'
import sys, xml.etree.ElementTree as ET
mode, path = sys.argv[1:3]
try: root = ET.parse(path).getroot()
except Exception:
    print(0) if mode in ("ran", "skipped", "colerrors") else None
    sys.exit(0)
tcs = list(root.iter("testcase"))
tid = lambda tc: f'{tc.get("classname")}::{tc.get("name")}'
if mode == "ran":        print(sum(1 for tc in tcs if tc.get("classname")))   # collection errors have an empty classname
elif mode == "skipped":  print(sum(1 for tc in tcs if tc.find("skipped") is not None))
elif mode == "colerrors":print(sum(1 for tc in tcs if not tc.get("classname") and tc.find("error") is not None))
elif mode == "fails":
    for tc in tcs:
        if tc.find("failure") is not None or tc.find("error") is not None: print(tid(tc))
elif mode == "skips":
    for tc in tcs:
        if tc.find("skipped") is not None: print(tid(tc))
PY
}

# ---------- abort / collect modes: re-attach by run id ----------
if [ "$MODE" != run ]; then
  ID=$RUN_ARG
  case "$ID" in ''|*[!a-z0-9]*) echo "bad run id"; exit 64;; esac
  RUN=/tmp/ames-rv-$ID
  hr "test -d $RUN" 2>/dev/null || { echo "no run $ID on $HOST (already collected, aborted, or wrong --host)"; exit 70; }
fi
if [ "$MODE" = abort ]; then
  # bash defers its TERM trap until the foreground `docker wait` returns, so remove the containers too;
  # the driver's EXIT trap then removes the ES container and releases the claim. Wait for it to finish.
  hr "pkill -TERM -f 'ames-rv-$ID/driver.sh' || true; sleep 1; docker ps -aq --filter name=ames-rv-$ID- | while read -r c; do docker rm -f \$c >/dev/null; done; for i in \$(seq 1 60); do pgrep -f 'ames-rv-$ID/driver.sh' >/dev/null || break; sleep 1; done; rm -rf $RUN"
  echo "aborted $ID"
  exit 0
fi

mkdir -p "$CACHE"
START=$(date +%s)

if [ "$MODE" = collect ]; then
  ENVF=$(mktemp "${TMPDIR:-/tmp}/ames-rv-env.XXXXXX")
  hr "cat $RUN/env.sh" > "$ENVF" || { echo "cannot read $RUN/env.sh"; exit 70; }
  # shellcheck disable=SC1090
  . "$ENVF"; rm -f "$ENVF"
  SHORT=${HEAD_SHA:0:7}
  LOGDIR=$CACHE/$SHORT-$(date +%Y%m%d-%H%M%S)-collect
  mkdir -p "$LOGDIR"
  echo "re-attached to run $ID on $HOST (head=${HEAD_SHA:0:7} base=${BASE_SHA:0:7})"
else
  HEAD_SHA=$(git -C "$REPO" rev-parse --verify "$REF^{commit}" 2>/dev/null) || { echo "REFUSED: bad --ref $REF"; exit 64; }
  BASE_SHA=-
  if [ -n "$BASE" ]; then
    BASE_SHA=$(git -C "$REPO" rev-parse --verify "$BASE^{commit}" 2>/dev/null) || { echo "REFUSED: bad --base $BASE"; exit 64; }
  fi
  ID=$(date +%m%d%H%M%S)$RANDOM
  RUN=/tmp/ames-rv-$ID
  ACTOR="${AMES_TEST_ACTOR:-ames-test-remote-${USER:-user}-$ID}"
  SHORT=${HEAD_SHA:0:7}
  LOGDIR=$CACHE/$SHORT-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$LOGDIR"
fi
exec > >(tee -a "$LOGDIR/run.log") 2>&1
echo "AMES remote test: host=$HOST es=127.0.0.1:$PORT id=$ID head=${HEAD_SHA:0:7} base=${BASE_SHA:0:7} log=$LOGDIR/run.log"

DRIVER_STARTED=0 CLAIMED=0
[ "$MODE" = collect ] && DRIVER_STARTED=1

on_exit() {
  local rc=$?
  trap - EXIT
  if [ "$DRIVER_STARTED" = 0 ]; then
    # failed before the host driver took over: undo what this script did on the host
    "${SSH[@]}" "rm -rf $RUN" >/dev/null 2>&1
    [ "$CLAIMED" = 1 ] && ops_claim "release '$RES'" >/dev/null 2>&1
  fi
  exit $rc
}
on_signal() {
  echo
  if [ "$DRIVER_STARTED" = 1 ]; then
    echo "runner interrupted; the run keeps going on $HOST and cleans up after itself."
    echo "  re-attach: $0 --host $HOST --collect $ID     abort: $0 --host $HOST --abort $ID"
  fi
  exit 70
}
trap on_exit EXIT
trap on_signal INT TERM HUP

if [ "$MODE" = run ]; then
  "${SSH[@]}" true 2>/dev/null || { echo "INFRA: host $HOST not reachable over ssh"; exit 70; }

  # ---- 1. claim the shared resource (a refusal means somebody else is using the host's test ES)
  if CL=$(ops_claim "claim '$RES' 'AMES pytest run $ID head=${HEAD_SHA:0:7} (throwaway ES :$PORT)' 7200" 2>&1); then
    CLAIMED=1; echo "$CL"
  else
    echo "$CL"; echo "BUSY: $RES is claimed by another actor; not running"; exit 75
  fi

  # ---- 2. memory gate (the host is RAM-tight prod): free+inactive+speculative+purgeable pages
  AVAIL=$(hr "vm_stat | awk '/page size of/ {ps=\$8} /^Pages (free|inactive|speculative|purgeable):/ {gsub(/\\./,\"\",\$NF); n+=\$NF} END {print int(n*ps/1048576)}'" 2>/dev/null)
  SWAP=$(hr "sysctl -n vm.swapusage" 2>/dev/null | sed 's/  */ /g')
  case "$AVAIL" in ''|*[!0-9]*) echo "INFRA: could not read free memory on $HOST"; exit 70;; esac
  echo "host memory available: ${AVAIL} MB (floor ${MIN_FREE_MB} MB, AMES_TEST_MIN_FREE_MB); swap: $SWAP"
  SWAPFREE=$(hr "sysctl -n vm.swapusage" 2>/dev/null | sed -n 's/.*free = \([0-9]*\).*/\1/p')
  if [ -n "$SWAPFREE" ] && [ "$SWAPFREE" -lt "$MIN_SWAP_FREE_MB" ]; then
    echo "INFRA: swap nearly full on $HOST (${SWAPFREE} MB free < ${MIN_SWAP_FREE_MB} MB, AMES_TEST_MIN_SWAP_FREE_MB); refusing"
    exit 70
  fi
  if [ "$AVAIL" -lt "$MIN_FREE_MB" ]; then
    echo "INFRA: only ${AVAIL} MB free on $HOST (< ${MIN_FREE_MB} MB); refusing to start ES + test containers on the prod host"
    exit 70
  fi

  # ---- 3. ship the WHOLE tree at each ref + the driver
  hr "mkdir -p $RUN/head $RUN/out" || { echo "INFRA: cannot create $RUN"; exit 70; }
  ship() { git -C "$REPO" archive --format=tar "$1" | "${SSHI[@]}" "tar xf - -C $RUN/$2"; }
  ship "$HEAD_SHA" head || { echo "INFRA: shipping head failed"; exit 70; }
  if [ "$BASE_SHA" != - ]; then
    { hr "mkdir -p $RUN/base" && ship "$BASE_SHA" base; } || { echo "INFRA: shipping base failed"; exit 70; }
  fi
  scp -q "$HERE/test-remote-host.sh" "$HOST:$RUN/driver.sh" || { echo "INFRA: scp driver failed"; exit 70; }
  {
    printf 'ID=%q\nPORT=%q\nKEEP_ES=%q\nHEAD_SHA=%q\nBASE_SHA=%q\nACTOR=%q\nRES=%q\n' \
      "$ID" "$PORT" "$KEEP_ES" "$HEAD_SHA" "$BASE_SHA" "$ACTOR" "$RES"
    for n in AMES_TEST_ES_IMAGE AMES_TEST_CONTAINER_MEM AMES_TEST_SUITE_TIMEOUT_S AMES_HTTP_RETRY_BUDGET_S AMES_TEST_MODEL_CACHE; do
      [ -n "${!n:-}" ] && printf '%s=%q\n' "$n" "${!n}"
    done
    printf 'PYARGS=('; for a in ${PYARGS[@]+"${PYARGS[@]}"}; do printf '%q ' "$a"; done; printf ')\n'
  } | "${SSHI[@]}" "cat > $RUN/env.sh" || { echo "INFRA: writing env.sh failed"; exit 70; }

  # ---- 4. detach the driver: nothing below depends on this ssh session staying alive
  "${SSH[@]}" "$PATHX; nohup bash $RUN/driver.sh $RUN > $RUN/driver.log 2>&1 < /dev/null & echo started" >/dev/null \
    || { echo "INFRA: could not start the host driver"; exit 70; }
  DRIVER_STARTED=1
  echo "driver detached on $HOST (run $ID). Recovery if this runner dies: $0 --host $HOST --collect $ID"
fi

# ---------- poll with short ssh calls ----------
last="" fails=0 shown=0
while :; do
  if st=$(hr "cat $RUN/status 2>/dev/null"); then
    fails=0
  else
    fails=$((fails + 1))
    if [ "$fails" -ge 30 ]; then
      echo "INFRA: lost contact with $HOST for $fails polls; the run continues there. Re-attach: $0 --host $HOST --collect $ID"
      exit 70
    fi
    sleep 10; continue
  fi
  now=$(date +%s)
  if [ "$st" != "$last" ] || [ $((now - shown)) -ge 120 ]; then
    echo "[$(TZ=Europe/Warsaw date +%H:%M:%S)] +$((now - START))s status: ${st:-<none yet>}"
    last=$st; shown=$now
  fi
  case "$st" in done*|infra*) break;; esac
  if [ $((now - START)) -gt "$WAIT_S" ]; then
    echo "INFRA: gave up waiting after ${WAIT_S}s; the run continues on $HOST. Re-attach: $0 --host $HOST --collect $ID"
    exit 70
  fi
  sleep 10
done

# ---------- collect ----------
scp -q -r "$HOST:$RUN/out/." "$LOGDIR/" 2>/dev/null || echo "warning: could not copy results from $HOST:$RUN/out"
hr "cat $RUN/driver.log" 2>/dev/null | sed 's/^/  host: /'
# the driver trap has already removed containers, ES, trees and the claim; drop the run dir too
hr "rm -rf $RUN" || echo "warning: could not remove $RUN on $HOST"
DRIVER_STARTED=1

case "$st" in infra*) echo "===== RESULT INFRA: $st"; exit 70;; esac
HRC=$(echo "$st" | sed -n 's/.*head_rc=\([^ ]*\).*/\1/p'); BRC=$(echo "$st" | sed -n 's/.*base_rc=\([^ ]*\).*/\1/p')
HX=$LOGDIR/junit-head.xml BX=$LOGDIR/junit-base.xml
[ "$HRC" = 70 ] && { echo "===== RESULT INFRA: head container could not be started"; exit 70; }
[ "$BRC" = 70 ] && { echo "===== RESULT INFRA: base container could not be started"; exit 70; }

HN=$(junit ran "$HX"); HS=$(junit skipped "$HX"); HCE=$(junit colerrors "$HX")
echo "===== head ${HEAD_SHA:0:7}: executed $HN tests, $HS skipped, $HCE collection errors (pytest rc=$HRC)"
tail -n 25 "$LOGDIR/head.log" 2>/dev/null | sed 's/^/  | /'
echo "===== head skips (reason is in the pytest -rs lines):"
grep -E '^SKIPPED' "$LOGDIR/head.log" 2>/dev/null | sed 's/^/  /' || true

# positive control: a collection crash or a filter that selects nothing must not read as green
if [ "$HN" = 0 ]; then echo "===== RESULT FAIL: head executed 0 tests (suite did not really run)"; exit 1; fi
if [ "$HN" -lt "$MIN_TESTS" ] && [ ${#PYARGS[@]} = 0 ]; then
  echo "===== RESULT FAIL: head executed $HN < $MIN_TESTS tests (suite did not really run)"; exit 1
fi
[ "$HRC" = 124 ] && { echo "===== RESULT FAIL: head suite timed out"; exit 1; }

if [ "$BASE_SHA" = - ]; then
  echo "===== RESULT head_rc=$HRC executed=$HN skipped=$HS collection_errors=$HCE (no base comparison)"
  [ "$HRC" = 0 ] && exit 0 || exit 1
fi

BN=$(junit ran "$BX"); BS=$(junit skipped "$BX"); BCE=$(junit colerrors "$BX")
echo "===== base ${BASE_SHA:0:7}: executed $BN tests, $BS skipped, $BCE collection errors (pytest rc=$BRC)"
[ "$BRC" = 124 ] && { echo "===== RESULT FAIL: base suite timed out"; exit 1; }
junit fails "$HX" | sort > "$LOGDIR/fail-head.txt"
junit fails "$BX" | sort > "$LOGDIR/fail-base.txt"
junit skips "$HX" | sort > "$LOGDIR/skip-head.txt"
junit skips "$BX" | sort > "$LOGDIR/skip-base.txt"
NEW=$(comm -23 "$LOGDIR/fail-head.txt" "$LOGDIR/fail-base.txt")
FIXED=$(comm -13 "$LOGDIR/fail-head.txt" "$LOGDIR/fail-base.txt")
NEWSKIP=$(comm -23 "$LOGDIR/skip-head.txt" "$LOGDIR/skip-base.txt")
echo "===== NEW failures/errors vs base (gate):"; indent "${NEW:-<none>}"
echo "===== fixed vs base:"; indent "${FIXED:-<none>}"
echo "===== pre-existing (fail on both):"; indent "$(comm -12 "$LOGDIR/fail-head.txt" "$LOGDIR/fail-base.txt")"
echo "===== newly skipped vs base (informational; check none was passing on base):"; indent "${NEWSKIP:-<none>}"
NNEW=$(printf '%s' "$NEW" | grep -c . || true)
echo "===== RESULT head_rc=$HRC base_rc=$BRC head_tests=$HN base_tests=$BN head_skipped=$HS base_skipped=$BS head_collection_errors=$HCE new=$NNEW"
[ "$NNEW" = 0 ] && exit 0 || exit 1

#!/usr/bin/env bash
# Run the full AMES backend pytest suite from this machine on a remote docker host
# against a THROWAWAY Elasticsearch, and gate on "no NEW failures vs a base ref".
#
#   scripts/test-remote.sh --host SSH [--repo DIR] [--ref REF] [--base REF] [--port N] [--keep-es] [-- pytest args]
#   scripts/test-remote.sh --collect RUN_ID      re-attach to a run (local runner died / ssh dropped)
#   scripts/test-remote.sh --abort RUN_ID        kill a run and clean the host
#
#   --repo  git checkout to archive refs from (default: the checkout this script lives in)
#   --ref   commit to test (default HEAD)             --base  also run BASE; exit 1 on NEW failures vs it
#   --host  ssh alias of the docker host (default $AMES_TEST_HOST; required, no built-in default)
#   --port  host-local ES port (default 9291)
#   --keep-es  leave the ES container (and any external claim) in place after the run
#   `-- ...` is passed to pytest; paths are relative to backend/ (e.g. `-- tests/test_tombstone.py -k reason`).
#
# How: the WHOLE repo at each ref is shipped with `git archive` (tests load ../../scripts/*). On the host
# a detached driver (scripts/test-remote-host.sh) starts a throwaway ES (docker, 127.0.0.1:PORT) and runs
# pytest in a container from the prod backend image (`docker inspect ames-backend`) with the tree mounted,
# AMES_EMBED_BACKEND=local and a per-run amtest_ index prefix. All state is on the host; this script only
# polls with short ssh calls, so a dropped ssh / killed runner loses nothing (see --collect).
# Host: takes a host-side lock dir (/tmp/ames-test-lock-<name>; one run at a time) and, when
# $AMES_TEST_CLAIM_CMD names a claim helper on the host, also its claim; refuses on low free memory;
# always cleans up on exit. A driver that is SIGKILLed / OOM-killed is detected by the poll (pid check)
# and reaped host-side (containers, throwaway ES, indices, lock, claim, run dir): exit 70.
# Ports 9200/9220/9268/9279 are refused (they may be a production AMES / shared ES); nothing here writes there.
#
# Exit: 0 green (no new failures with --base) | 1 failures / positive control failed / a test that passed
#       on base is skipped on head | 64 refused (bad args, reserved port, no host) | 70 infra (ssh, docker,
#       low memory, ES, dead driver, base did not run) | 75 host busy (lock or claim held)
# Env:  AMES_TEST_MIN_FREE_MB (4096) / AMES_TEST_MIN_SWAP_FREE_MB (512) host floors; AMES_MIN_TESTS (50) positive control;
#       AMES_TEST_WAIT_S (6000) max time this script waits; AMES_TEST_SUITE_TIMEOUT_S (2400) per suite;
#       AMES_HTTP_RETRY_BUDGET_S (2); AMES_TEST_CONTAINER_MEM (3g); AMES_TEST_ES_IMAGE; AMES_TEST_MODEL_CACHE (host path)
#       AMES_TEST_CLAIM_CMD  host-side claim helper (subcommands claim|release|journal, e.g. `~/bin/<helper>`); unset = lock dir only
#       AMES_TEST_CLAIM_RESOURCE (default <host>:ames-review-es)  AMES_TEST_ACTOR  claim identity
# Logs: ${AMES_TEST_LOG_DIR:-${XDG_CACHE_HOME:-~/.cache}/ames-test-remote}/<ref>-<time>/  (run.log, junit-*.xml, head.log, base.log)
set -uo pipefail

REPO="" REF=HEAD BASE="" HOST=${AMES_TEST_HOST:-} PORT=9291 KEEP_ES=0 MODE=run RUN_ARG=""
PYARGS=()
usage() { sed -n 2,34p "$0" | sed 's/^# \{0,1\}//'; }
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
[ -n "$HOST" ] || { echo "REFUSED: no docker host: pass --host SSH_ALIAS or set AMES_TEST_HOST" >&2; usage >&2; exit 64; }
case "$PORT" in
  9200|9220|9268|9279) echo "REFUSED: port $PORT is reserved (prod 9268 / holdout / implementer ES)"; exit 64;;
  ''|*[!0-9]*) echo "REFUSED: bad port '$PORT'"; exit 64;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "${REPO:-$HERE}" rev-parse --show-toplevel) || { echo "not inside a git checkout"; exit 64; }
SSHOPT=(-o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3)
SSH=(ssh -n "${SSHOPT[@]}" "$HOST")     # no stdin: safe inside loops/pipes
SSHI=(ssh "${SSHOPT[@]}" "$HOST")       # reads stdin: for tar / env.sh uploads
RES=${AMES_TEST_CLAIM_RESOURCE:-$HOST:ames-review-es}
CLAIM_CMD=${AMES_TEST_CLAIM_CMD:-}
LOCK_DIR=/tmp/ames-test-lock-$(printf %s "$RES" | tr -c 'A-Za-z0-9._-' '_')
MIN_TESTS=${AMES_MIN_TESTS:-50}
MIN_FREE_MB=${AMES_TEST_MIN_FREE_MB:-4096}
MIN_SWAP_FREE_MB=${AMES_TEST_MIN_SWAP_FREE_MB:-512}
WAIT_S=${AMES_TEST_WAIT_S:-6000}
CACHE=${AMES_TEST_LOG_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/ames-test-remote}
# shellcheck disable=SC2016  # expanded on the host, not here
PATHX='export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH'
hr() { "${SSH[@]}" "$PATHX; $1"; }   # one short remote command, docker/brew on PATH

# Mutual exclusion on the host. The lock dir (mkdir is atomic) always applies; an optional external claim
# helper ($AMES_TEST_CLAIM_CMD, run ON the host) is layered on top so other tools can see the claim.
# Exit: 0 taken | 75 busy (lock held, or helper exit 1) | 70 anything else.
host_claim() {
  "${SSHI[@]}" "bash -s -- $(printf '%q ' "$ID" "$LOCK_DIR" "$RES" "$ACTOR" "$CLAIM_CMD" "AMES pytest run $ID head=${HEAD_SHA:0:7} (throwaway ES :$PORT)")" <<'HOSTSH'
ID=$1 L=$2 RES=$3 ACTOR=$4 CMD=$5 INTENT=$6 TTL=7200
now=$(date +%s)
take() { mkdir "$L" 2>/dev/null && echo "$ID $now" > "$L/owner"; }
mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo "$now"; }   # GNU first: BSD `stat -f` means something else on Linux
# Is the lock stale? Decided by OWNER LIVENESS, never by age alone:
#   - owner's driver is gone (driver.pid names no live driver.sh process), or
#   - owner's run dir has been missing > 120 s (claim older than 120 s and no /tmp/ames-rv-<owner>), or
#   - no owner file: the lock dir itself is older than 120 s.
# A launcher that was killed before it started a driver leaves a run dir without driver.pid: that is
# only stale once the run dir has been idle for the whole TTL.
is_stale() {
  oid="" ots=0
  read -r oid ots < "$L/owner" 2>/dev/null
  if [ -z "$oid" ]; then [ $((now - $(mtime "$L"))) -gt 120 ]; return; fi
  rd=/tmp/ames-rv-$oid
  if [ ! -d "$rd" ]; then [ $((now - ots)) -gt 120 ]; return; fi
  if [ -f "$rd/driver.pid" ]; then
    p=$(cat "$rd/driver.pid" 2>/dev/null)
    ps -p "$p" -o command= 2>/dev/null | grep -q 'driver.sh' && return 1
    return 0
  fi
  [ $((now - $(mtime "$rd"))) -gt "$TTL" ]
}
if ! take; then
  if is_stale; then
    stale_oid=$oid
    # atomic retire: only the winner of the mv proceeds; the loser re-checks (and finds a fresh lock)
    if mv "$L" "$L.stale.$ID" 2>/dev/null; then
      if [ "$(cut -d' ' -f1 "$L.stale.$ID/owner" 2>/dev/null)" = "$stale_oid" ]; then
        rm -rf "$L.stale.$ID"
        if take; then echo "took over stale lock of run ${stale_oid:-?}"; else echo "lock $L taken by another run"; exit 75; fi
      else
        # we retired a lock that was replaced between the check and the mv: put it back
        mv "$L.stale.$ID" "$L" 2>/dev/null || rm -rf "$L.stale.$ID"
        echo "lock $L is held by a live run"; exit 75
      fi
    else
      echo "lock $L held by run ${oid:-?} (another runner is retiring it)"; exit 75
    fi
  else
    echo "lock $L held by run ${oid:-?}"; exit 75
  fi
fi
case "$CMD" in \~/*) CMD=$HOME/${CMD#\~/};; esac
if [ -n "$CMD" ]; then
  if [ -x "$CMD" ]; then
    OPS_ACTOR=$ACTOR "$CMD" claim "$RES" "$INTENT" "$TTL"; rc=$?
    if [ "$rc" != 0 ]; then rm -rf "$L"; [ "$rc" = 1 ] && exit 75; exit 70; fi
  else
    echo "note: claim helper $CMD not executable on this host; using the lock dir only"
  fi
fi
echo "locked $L (run $ID)"
HOSTSH
}
host_release() { # drop our lock + external claim on the host (the lock only if it is ours)
  "${SSHI[@]}" "bash -s -- $(printf '%q ' "$ID" "$LOCK_DIR" "$RES" "$ACTOR" "$CLAIM_CMD")" <<'HOSTSH'
ID=$1 L=$2 RES=$3 ACTOR=$4 CMD=$5
[ "$(cut -d' ' -f1 "$L/owner" 2>/dev/null)" = "$ID" ] && rm -rf "$L"
case "$CMD" in \~/*) CMD=$HOME/${CMD#\~/};; esac
[ -n "$CMD" ] && [ -x "$CMD" ] && OPS_ACTOR=$ACTOR "$CMD" release "$RES" >/dev/null 2>&1
true
HOSTSH
}

indent() { printf '%s\n' "$1" | sed 's/^/  /'; }

# ---------- junit helpers (the XML is pytest's own output, fetched seconds ago) ----------
junit() { # mode xml ; modes: ran | skipped | colerrors | fails | skips | passes | skipinfo
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
elif mode == "passes":
    for tc in tcs:
        if tc.get("classname") and tc.find("skipped") is None and tc.find("failure") is None and tc.find("error") is None:
            print(tid(tc))
elif mode == "skipinfo":   # id<TAB>reason (message + body, one line)
    for tc in tcs:
        sk = tc.find("skipped")
        if sk is not None:
            print(tid(tc) + "\t" + " ".join(((sk.get("message") or "") + " " + (sk.text or "")).split()))
PY
}

# Skip gate (junit in, one line per lost test out): "OK<TAB>id<TAB>reason" or "BAD<TAB>id<TAB>reason".
# A test is LOST when it passed on base and is skipped on head. A whole-file skip (module-level pytest.skip,
# importorskip, require_llm) is ONE junit entry with an empty classname and name="tests.test_x"; it is
# expanded to every base-passing test of that module. A lost test is excused only when the skip is the
# shared LLM-gateway probe ("... (AMES_LLM_BASE) is unreachable (Exc); this test needs a live LLM") AND base
# shows the same gateway-down skip/failure for that test or its module; anything else is a head regression.
skipgate() { # head.xml base.xml
  python3 - "$1" "$2" <<'PY'
import re, sys, xml.etree.ElementTree as ET
def load(path):
    try: return list(ET.parse(path).getroot().iter("testcase"))
    except Exception: return []
head, base = load(sys.argv[1]), load(sys.argv[2])
GW = re.compile(r"LLM gateway \S+ \(AMES_LLM_BASE\) is unreachable \(\w+\); this test needs a live LLM")
def text(el): return " ".join(((el.get("message") or "") + " " + (el.text or "")).split())
def module(tc):
    cn = tc.get("classname")
    if not cn: return tc.get("name")                       # whole-file entry: name is the dotted module
    return re.sub(r"\.[A-Z]\w*$", "", cn)                  # drop a Test class segment
tid = lambda tc: f'{tc.get("classname")}::{tc.get("name")}'
def kind(tc):
    for k in ("skipped", "failure", "error"):
        if tc.find(k) is not None: return k
    return "pass"
base_pass = [tc for tc in base if tc.get("classname") and kind(tc) == "pass"]
# base evidence of "gateway down" per test id and per module (a gateway skip, or a failure/collection error saying so)
gw_ids, gw_mods = set(), set()
for tc in base:
    k = kind(tc)
    if k == "pass": continue
    el = tc.find(k)
    if GW.search(text(el)) or "URLError" in text(el):
        gw_ids.add(tid(tc)); gw_mods.add(module(tc))
for tc in head:
    sk = tc.find("skipped")
    if sk is None: continue
    reason = text(sk)
    if tc.get("classname"):
        lost = [tid(tc)] if tid(tc) in {tid(b) for b in base_pass} else []
    else:
        m = tc.get("name")
        lost = [tid(b) for b in base_pass if b.get("classname") == m or b.get("classname").startswith(m + ".")]
        if not lost: continue
    for t in lost:
        mod = t.split("::")[0]
        mod = re.sub(r"\.[A-Z]\w*$", "", mod)
        excused = GW.search(reason) and (t in gw_ids or mod in gw_mods)
        print(("OK" if excused else "BAD") + "\t" + t + "\t" + reason[:160])
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
  # the host-side reap is idempotent: it also cleans up when the driver was already dead
  hr "pkill -TERM -f 'ames-rv-$ID/driver.sh' || true; sleep 1; docker ps -aq --filter name=ames-rv-$ID- | while read -r c; do docker rm -f \$c >/dev/null; done; for i in \$(seq 1 60); do pgrep -f 'ames-rv-$ID/driver.sh' >/dev/null || break; sleep 1; done; bash $RUN/driver.sh --reap $RUN"
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
    [ "$CLAIMED" = 1 ] && host_release >/dev/null 2>&1
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
  CL=$(host_claim 2>&1); crc=$?
  echo "$CL"
  case "$crc" in
    0) CLAIMED=1;;
    75) echo "BUSY: $RES is held by another run/actor; not running"; exit 75;;
    *) echo "INFRA: claiming $RES on $HOST failed (rc=$crc)"; exit 70;;
  esac

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
    printf 'ID=%q\nPORT=%q\nKEEP_ES=%q\nHEAD_SHA=%q\nBASE_SHA=%q\nACTOR=%q\nRES=%q\nCLAIM_CMD=%q\nLOCK_DIR=%q\n' \
      "$ID" "$PORT" "$KEEP_ES" "$HEAD_SHA" "$BASE_SHA" "$ACTOR" "$RES" "$CLAIM_CMD" "$LOCK_DIR"
    for n in AMES_TEST_ES_IMAGE AMES_TEST_CONTAINER_MEM AMES_TEST_SUITE_TIMEOUT_S AMES_HTTP_RETRY_BUDGET_S AMES_TEST_MODEL_CACHE; do
      [ -n "${!n:-}" ] && printf '%s=%q\n' "$n" "${!n}"
    done
    printf 'PYARGS=('; for a in ${PYARGS[@]+"${PYARGS[@]}"}; do printf '%q ' "$a"; done; printf ')\n'
  } | "${SSHI[@]}" "cat > $RUN/env.sh" || { echo "INFRA: writing env.sh failed"; exit 70; }

  # ---- 4. detach the driver: nothing below depends on this ssh session staying alive
  "${SSH[@]}" "$PATHX; nohup bash $RUN/driver.sh $RUN > $RUN/driver.log 2>&1 < /dev/null & echo \$! > $RUN/driver.pid; echo started" >/dev/null \
    || { echo "INFRA: could not start the host driver"; exit 70; }
  DRIVER_STARTED=1
  echo "driver detached on $HOST (run $ID). Recovery if this runner dies: $0 --host $HOST --collect $ID"
fi

# ---------- poll with short ssh calls ----------
# One call reads driver liveness FIRST, then the status (a driver that exits writes its final status
# before it dies, so "dead + non-terminal status" can only mean it was killed: SIGKILL / OOM / host reboot).
POLLQ="a=1; if [ -f $RUN/driver.pid ]; then p=\$(cat $RUN/driver.pid); ps -p \$p -o command= 2>/dev/null | grep -q 'driver.sh' || a=0; fi; echo alive=\$a; echo status=\$(cat $RUN/status 2>/dev/null)"
last="" fails=0 shown=0
while :; do
  if out=$(hr "$POLLQ"); then
    fails=0
  else
    fails=$((fails + 1))
    if [ "$fails" -ge 30 ]; then
      echo "INFRA: lost contact with $HOST for $fails polls; the run continues there. Re-attach: $0 --host $HOST --collect $ID"
      exit 70
    fi
    sleep 10; continue
  fi
  st=$(printf '%s\n' "$out" | sed -n 's/^status=//p')
  alive=$(printf '%s\n' "$out" | sed -n 's/^alive=//p')
  now=$(date +%s)
  if [ "$st" != "$last" ] || [ $((now - shown)) -ge 120 ]; then
    echo "[$(date +%H:%M:%S)] +$((now - START))s status: ${st:-<none yet>}"
    last=$st; shown=$now
  fi
  case "$st" in done*|infra*) break;; esac
  if [ "$alive" = 0 ]; then
    echo "INFRA: the host driver for run $ID died (status stuck at '${st:-<none>}'; SIGKILL/OOM?). Reaping the host."
    scp -q -r "$HOST:$RUN/out/." "$LOGDIR/" 2>/dev/null || true
    hr "cat $RUN/driver.log 2>/dev/null" | sed 's/^/  host: /'
    hr "bash $RUN/driver.sh --reap $RUN" | sed 's/^/  reap: /'
    DRIVER_STARTED=1
    echo "===== RESULT INFRA: driver died mid-run; host cleaned (partial logs in $LOGDIR)"
    exit 70
  fi
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
[ "$BRC" = 124 ] && { echo "===== RESULT INFRA: base suite timed out; base did not run"; exit 70; }
# A base that did not really run would turn every head failure into "NEW" or hide them as "pre-existing".
# Base collection errors are legitimate pre-existing failures (they show up as 'fixed'/'pre-existing'); what
# is fatal is a base that executed (almost) nothing.
base_ran=1
if [ ${#PYARGS[@]} = 0 ]; then
  [ "$BN" -ge "$MIN_TESTS" ] || base_ran=0
else
  [ "$BN" -ge 1 ] || grep -q 'no requested test path exists' "$LOGDIR/base.log" 2>/dev/null || base_ran=0
fi
if [ "$base_ran" = 0 ]; then
  echo "===== RESULT INFRA: base did not run (executed $BN tests, $BCE collection errors; need >= $MIN_TESTS unscoped)"; exit 70
fi
[ "$BCE" = 0 ] || echo "note: base has $BCE collection error(s); they count as pre-existing / fixed below"
junit fails "$HX" | sort > "$LOGDIR/fail-head.txt"
junit fails "$BX" | sort > "$LOGDIR/fail-base.txt"
junit skips "$HX" | sort > "$LOGDIR/skip-head.txt"
junit skips "$BX" | sort > "$LOGDIR/skip-base.txt"
junit passes "$BX" | sort > "$LOGDIR/pass-base.txt"
NEW=$(comm -23 "$LOGDIR/fail-head.txt" "$LOGDIR/fail-base.txt")
FIXED=$(comm -13 "$LOGDIR/fail-head.txt" "$LOGDIR/fail-base.txt")
NEWSKIP=$(comm -23 "$LOGDIR/skip-head.txt" "$LOGDIR/skip-base.txt")
# Skip gate (see skipgate above): tests that passed on base but are skipped on head.
BADSKIP="" OKSKIP=""
while IFS=$'\t' read -r verdict t reason; do
  [ -n "$t" ] || continue
  if [ "$verdict" = OK ]; then OKSKIP="$OKSKIP$t  [$reason]"$'\n'; else BADSKIP="$BADSKIP$t  [$reason]"$'\n'; fi
done < <(skipgate "$HX" "$BX")
echo "===== NEW failures/errors vs base (gate):"; indent "${NEW:-<none>}"
echo "===== fixed vs base:"; indent "${FIXED:-<none>}"
echo "===== pre-existing (fail on both):"; indent "$(comm -12 "$LOGDIR/fail-head.txt" "$LOGDIR/fail-base.txt" | grep . || echo "<none>")"
echo "===== newly skipped vs base (informational):"; indent "${NEWSKIP:-<none>}"
echo "===== passed on base, SKIPPED on head, allow-listed (gateway-down skip that base also shows):"; indent "$(printf '%s' "$OKSKIP" | grep . || echo "<none>")"
echo "===== passed on base, SKIPPED on head, NOT allowed (gate):"; indent "$(printf '%s' "$BADSKIP" | grep . || echo "<none>")"
NNEW=$(printf '%s' "$NEW" | grep -c . || true)
NBAD=$(printf '%s' "$BADSKIP" | grep -c . || true)
echo "===== RESULT head_rc=$HRC base_rc=$BRC head_tests=$HN base_tests=$BN head_skipped=$HS base_skipped=$BS head_collection_errors=$HCE new=$NNEW lost_to_skip=$NBAD"
[ "$NNEW" = 0 ] && [ "$NBAD" = 0 ] && exit 0 || exit 1

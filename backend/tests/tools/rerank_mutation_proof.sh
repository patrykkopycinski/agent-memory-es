#!/bin/bash
# round-5 mutation proof (v2: score untouched, rerank_rank ordering)
cd "$(git rev-parse --show-toplevel)"; . /opt/orca-base/work/ames-vm/env.sh
T() { sg docker -c "docker compose -f /opt/orca-base/Projects/agent-memory-es/backend/docker-compose.yml -f /opt/orca-base/work/ames-vm/docker-compose.vm.yml run --rm --no-deps -v "$PWD":/srv -w /srv/backend ames-backend python -m pytest tests/test_rerank_recall.py -q 2>&1" | grep -E '^FAILED|passed|failed' | sed 's/ - .*//'; }
M=backend/app/memory.py; R=backend/app/reranker.py; A=backend/app/main.py
cp $M /tmp/r5.mem.keep; cp $R /tmp/r5.rr.keep; cp $A /tmp/r5.main.keep
mut() { echo "== $1"; if ! python3 - "$2" "$3" "$4" <<'E'
import sys
f,a,b=sys.argv[1:4]; s=open(f).read()
if a not in s: sys.exit("MUTANT NOT APPLIED (pattern missing): %r" % a[:80])
open(f,'w').write(s.replace(a,b,1))
E
then echo "!! INVALID MUTANT (not applied)"; return; fi
T; cp /tmp/r5.mem.keep $M; cp /tmp/r5.rr.keep $R; cp /tmp/r5.main.keep $A; }
mut "M1 rerank runs AFTER the budget" $M '    reranked = False
    if (RERANK_DEFAULT if rerank is None else rerank):
        ordered, reranked = _rerank_fused(query, ordered)
    window = _apply_budgets(ordered, size, per_doc_budget=per_doc or PER_DOC_BUDGET)' '    reranked = False
    window = _apply_budgets(ordered, size, per_doc_budget=per_doc or PER_DOC_BUDGET)
    if (RERANK_DEFAULT if rerank is None else rerank):
        window, reranked = _rerank_fused(query, window)'
mut "M2 rerank never applied" $M 'if (RERANK_DEFAULT if rerank is None else rerank):' 'if False:'
mut "M3 rerank always applied (rerank=false ignored)" $M 'if (RERANK_DEFAULT if rerank is None else rerank):' 'if True:'
mut "M4 default flipped ON (must stay opt-in)" $M 'RERANK_DEFAULT = os.environ.get("AMES_RERANK", "0").strip().lower() in ("1", "true", "yes", "on")' 'RERANK_DEFAULT = os.environ.get("AMES_RERANK", "1").strip().lower() in ("1", "true", "yes", "on")'
mut "M13 stale rerank_score pop made index-based again (tail only)" $M '        if h["id"] not in scored:' '        if i >= len(ranked):'
mut "M5 depth ignored (whole list sent)" $M '    head, tail = ordered[:depth], ordered[depth:]' '    head, tail = ordered, []'
mut "M6 candidates the endpoint omitted are dropped" $M '    ranked += [h for h in head if h["id"] not in scored]    # never lose a candidate' '    pass'
mut "M7 _apply_budgets ignores rerank_rank (sorts by score)" $M 'picked.sort(key=lambda x: (x["rerank_rank"],) if "rerank_rank" in x else (-x.get("score", 0.0),))' 'picked.sort(key=lambda x: -x.get("score", 0.0))'
mut "M8 fallback fakes a rerank" $M '    if not out.get("reranked"):
        return ordered, False' '    if not out.get("reranked"):
        return ordered, True'
mut "M9 OSError escapes the reranker" $R 'urllib.error.URLError, RuntimeError, OSError, ValueError):' 'urllib.error.URLError, RuntimeError):'
mut "M10 HTTP rerank not plumbed" $A 'body.per_doc, body.rerank)' 'body.per_doc)'
mut "M11 score overwritten by rerank score (breaks score=rrf+recency)" $M '        h["rerank_rank"] = i' '        h["rerank_rank"] = i; h["score"] = float(-i)'
mut "M12 stale rerank_score never removed" $M '        if h["id"] not in scored:
            # an item the endpoint did not return must not carry a score it never received
            # (covers both an omitted head item and every item beyond `depth`)
            h.pop("rerank_score", None)' '        pass'
cmp $M /tmp/r5.mem.keep && cmp $R /tmp/r5.rr.keep && cmp $A /tmp/r5.main.keep && echo "restored byte-identical"

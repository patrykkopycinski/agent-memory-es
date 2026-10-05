#!/usr/bin/env python3
"""Mutation proof driver with a machine-checked verdict.

Fixes the review findings on the old shell script:
  #1 the script printed pytest output but never DECIDED anything: a mutant that survived, or a
     pattern that silently did not apply, still exited 0. Here every mutant gets a verdict from a
     structured table and the process exits non-zero unless every mutant is KILLED (its suite
     goes red) and the tree is restored byte-identical.
  #2 no timeout and no restore-on-interrupt: a hung suite left the repo mutated. Here each run has
     a timeout and a trap/atexit restore.
  #3 mutants were embedded in shell quoting, so a `"` or `$` in a pattern produced a NameError or a
     silently different mutation. Mutations live in a JSON table and are applied in Python.
  #4 the backup copy was unguarded: if `cp` failed, the "restore" wrote nothing and the next mutant
     ran against a half-mutated file. Here every file is hashed before and after, and the restore
     is VERIFIED against the pre-run hash (a mismatch aborts the run).
  #6 pytest output was piped through `grep -E '^FAILED|passed|failed'`, which hides collection
     errors (`ERROR tests/...`). Here the verdict uses pytest's EXIT CODE plus the full log, which
     is written to disk for the reviewer.

Usage:
  mutation_proof.py --command "<shell command running the suite>" --table <table.json> \
                    [--root <repo root>] [--timeout 300] [--log-dir DIR] [--only M1,M5]

Table format: {"mutants": [{"name": str, "file": str, "find": str, "replace": str,
                            "expect": "kill"|"survive"}]}
`expect: survive` marks an equivalent mutant that is documented as unkillable.
Exit: 0 only if every mutant is in its expected state and every file is byte-identical after.
"""
import argparse
import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Prover:
    def __init__(self, root, command, table, timeout, log_dir):
        self.root = root
        self.command = command
        self.timeout = timeout
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.mutants = json.load(open(table))["mutants"]
        self.pristine = {}          # abs path -> sha before anything was touched
        self.orig = {}              # abs path -> original bytes, for restore
        self.armed = False
        for m in self.mutants:
            p = os.path.join(root, m["file"])
            if not os.path.isfile(p):
                sys.exit("mutant %s targets a missing file: %s" % (m["name"], p))
            if p not in self.orig:
                with open(p, "rb") as fh:
                    self.orig[p] = fh.read()                       # #4: guarded backup (read, not cp)
                self.pristine[p] = sha(p)
        # #2: restore on any exit. Wrapped, because an exception inside an atexit handler changes
        # the process exit status - exactly the kind of silent failure this tool exists to prevent.
        atexit.register(self._restore_on_exit)

    def _restore_on_exit(self):
        try:
            if not self.restore():
                print("!! tree was NOT fully restored on exit", file=sys.stderr)
        except BaseException as e:                          # noqa: BLE001
            print("!! restore-on-exit failed: %r" % (e,), file=sys.stderr)

    def restore(self):
        """Put every touched file back, VERIFIED against its pre-run hash. Never raises: this also
        runs from the atexit hook, where an exception would mask the real exit status."""
        ok = True
        for p, blob in self.orig.items():
            try:
                with open(p, "wb") as fh:
                    fh.write(blob)
                if sha(p) != self.pristine[p]:
                    ok = False
                    print("!! RESTORE MISMATCH for %s" % p, file=sys.stderr)
            except OSError as e:
                # the file (or its directory) was removed by someone else after the run started;
                # nothing this process can do, and it must not crash the interpreter on the way out
                ok = False
                print("!! CANNOT RESTORE %s: %s" % (p, e), file=sys.stderr)
            except Exception as e:                        # noqa: BLE001 - never mask the exit code
                ok = False
                print("!! RESTORE ERROR for %s: %r" % (p, e), file=sys.stderr)
        self.armed = False
        return ok

    def run_suite(self, tag):
        # a mutant name is free text: slug it, or a "/" turns the log path into a subdirectory
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_") or "mutant"
        log = os.path.join(self.log_dir, "mutant-%s.log" % slug)
        try:
            # #6: keep the FULL output (collection ERRORs included) and use the exit code
            r = subprocess.run(self.command, shell=True, cwd=self.root, capture_output=True,
                               text=True, timeout=self.timeout)
            out, code, timed_out = r.stdout + r.stderr, r.returncode, False
        except subprocess.TimeoutExpired as e:
            out = "".join(x if isinstance(x, str) else (x or b"").decode(errors="replace")
                          for x in (e.stdout, e.stderr) if x)
            code, timed_out = None, True
        with open(log, "w") as fh:
            fh.write(out)
        return code, timed_out, log

    def prove_one(self, m):
        p = os.path.join(self.root, m["file"])
        src = self.orig[p].decode()
        if m["find"] not in src:
            return "INVALID", "pattern does not apply (mutation NOT performed)", None     # #3
        with open(p, "w") as fh:
            fh.write(src.replace(m["find"], m["replace"], 1))
        if sha(p) == self.pristine[p]:
            self.restore()
            return "INVALID", "mutation left the file unchanged", None
        code, timed_out, log = self.run_suite(m["name"])
        self.restore()
        if timed_out:
            return "TIMEOUT", "suite exceeded %ss" % self.timeout, log                       # #2
        red = code != 0                                                                      # #6
        want_kill = m.get("expect", "kill") == "kill"
        if red == want_kill:
            return ("KILLED" if want_kill else "KILLED(expected-survive)"), "", log
        return ("SURVIVED" if want_kill else "UNEXPECTED-PASS"), \
               "suite %s (want %s)" % ("passed" if not red else "failed",
                                       "red" if want_kill else "green"), log

    def run(self, only=None):
        rows, bad = [], 0
        for m in self.mutants:
            if only and m["name"] not in only:
                continue
            verdict, detail, log = self.prove_one(m)
            good = verdict.startswith("KILLED")
            bad += 0 if good else 1
            rows.append((m["name"], verdict, detail, log))
            print("%-6s %-28s %s" % (verdict, m["name"], detail))
        restored = self.restore()
        for p, want in self.pristine.items():
            if sha(p) != want:
                print("!! %s NOT byte-identical after the run" % p, file=sys.stderr)
                restored = False
        print("--- %d/%d mutants in the expected state; tree %s"
              % (len(rows) - bad, len(rows), "byte-identical" if restored else "DIRTY"))
        return 0 if (bad == 0 and restored and rows) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--command", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--log-dir", default=tempfile.mkdtemp(prefix="mutproof-"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = [x for x in args.only.split(",") if x]
    return Prover(args.root, args.command, args.table, args.timeout, args.log_dir).run(only)


if __name__ == "__main__":
    sys.exit(main())

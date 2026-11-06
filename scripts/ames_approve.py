"""Human-only approval CLI for governed team decisions.

  ames_approve.py list
  ames_approve.py show    <doc_id>
  ames_approve.py approve <doc_id> [--note TEXT]
  ames_approve.py reject  <doc_id> --reason TEXT

The approver key is read ONLY from an interactive terminal prompt (getpass). It is never taken
from an argument, an environment variable or a file, and the tool refuses to run when stdin is
not a TTY, so a script or an agent cannot drive it. The service URL may come from --url or
AMES_SERVICE_URL (it is not a secret). Before the prompt the tool prints the target host and asks
for confirmation, and it refuses plain http unless the host is loopback, so the key is never sent
in the clear to a remote machine or to a URL you did not mean.
"""
import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

LOOPBACK = ("localhost", "127.0.0.1", "::1")
DEFAULT_URL = "http://localhost:8123"


def request(url: str, key: str, method: str, path: str, body: dict = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data, method=method,
        headers={"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e).get("detail", "")
        except Exception:
            detail = ""
        return e.code, {"error": detail or e.reason}


def check_url(url: str) -> bool:
    """https anywhere, or http to a loopback host only. SystemExit otherwise."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError:
        raise SystemExit(f"ames_approve: not a valid service URL: {url!r}") from None
    if parts.scheme not in ("http", "https") or not host:
        raise SystemExit(f"ames_approve: the service URL needs an http(s) scheme and a host: {url!r}")
    if parts.scheme == "http" and host not in LOOPBACK:
        raise SystemExit(f"ames_approve: refusing to send the key over plain http to {host}; "
                         "use an https URL (plain http is accepted for loopback only)")
    return True


def confirm_target(url: str, stdin=None, ask=input) -> None:
    """Show where the key will go and require an explicit yes. Needs a TTY; returns None on yes."""
    stdin = stdin if stdin is not None else sys.stdin
    if not stdin.isatty():
        raise SystemExit("ames_approve: refusing to run without an interactive terminal")
    host = urllib.parse.urlsplit(url).hostname
    print(f"ames_approve: the approver key will be sent to {host} ({url})")
    if ask("Continue? [y/N] ").strip().lower() != "y":
        raise SystemExit("ames_approve: aborted")


def read_key(stdin=None, prompt=getpass.getpass) -> str:
    """The key comes from a terminal prompt or nowhere."""
    stdin = stdin if stdin is not None else sys.stdin
    if not stdin.isatty():
        raise SystemExit("ames_approve: refusing to run without an interactive terminal "
                         "(the approver key is only read from a prompt)")
    key = prompt("Approver key: ").strip()
    if not key:
        raise SystemExit("ames_approve: no key entered")
    return key


def render(p: dict) -> str:
    lines = [f"{p['id']}  by {p['proposed_by']}  expires {p.get('expires_at') or 'never'}"]
    if p.get("proposal_for"):
        lines.append(f"  OVERRIDES active decision {p['proposal_for']}")
    lines.append(f"  approvals: {len(p['approved_by'])}/{p['approvals_required']} "
                 f"({', '.join(p['approved_by']) or 'none'}); needed: {p['approvals_needed']}")
    for o in p.get("objections") or []:
        lines.append(f"  objection by {o['by']}: {o['reason']}")
    lines.append("  " + (p.get("text") or "").replace("\n", "\n  "))
    return "\n".join(lines)


def run(args, key: str, out=print) -> int:
    if args.cmd in ("list", "show"):
        status, res = request(args.url, key, "GET", "/governance/proposals?limit=100")
        if status != 200:
            out(f"error {status}: {res.get('error')}")
            return 1
        items = res["proposals"]
        if args.cmd == "show":
            items = [p for p in items if p["id"] == args.doc_id or p.get("doc_group") == args.doc_id]
            if not items:
                out("no such open proposal")
                return 1
        for p in items:
            out(render(p))
        if args.cmd == "list":
            out(f"{res['total']} open proposal(s)")
            for c in res.get("override_conflicts") or []:
                out(f"WARNING: override {c['id']} is active but could not supersede "
                    f"{c.get('proposal_for')}; both are in force until an operator resolves it")
                out("  " + (c.get("text") or "").replace("\n", "\n  "))
        return 0
    if args.cmd == "approve":
        status, res = request(args.url, key, "POST", "/governance/approve",
                              {"doc_id": args.doc_id, **({"note": args.note} if args.note else {})})
    else:
        status, res = request(args.url, key, "POST", "/governance/reject",
                              {"doc_id": args.doc_id, "reason": args.reason})
    if status != 200:
        out(f"error {status}: {res.get('error')}")
        return 1
    out(json.dumps(res))
    if res.get("conflict"):
        out(f"WARNING: {res.get('warning') or 'override activated but its target is held by another decision'}")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ames_approve", description=__doc__.split("\n")[0])
    p.add_argument("--url", default=os.environ.get("AMES_SERVICE_URL", DEFAULT_URL))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("show").add_argument("doc_id")
    a = sub.add_parser("approve")
    a.add_argument("doc_id")
    a.add_argument("--note")
    r = sub.add_parser("reject")
    r.add_argument("doc_id")
    r.add_argument("--reason", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    check_url(args.url)
    confirm_target(args.url)
    return run(args, read_key())


if __name__ == "__main__":
    sys.exit(main())

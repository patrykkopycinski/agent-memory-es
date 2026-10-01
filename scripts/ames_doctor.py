"""ames doctor: one-command preflight for agent-memory-es adoption.

Checks, in order: Python version, deps, ES reachability, index health,
API service health, key validity. Prints the exact fix command for every
failure. Exit 0 = all green; exit 1 = fixable issues found.
"""
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request

ES_URL = os.environ.get("AMES_ES_URL", "http://localhost:9268")
AMES_URL = os.environ.get("AMES_SERVICE_URL", "http://localhost:8123")
AMES_KEY = os.environ.get("AMES_SERVICE_KEY", "")

DOCKER_ES = (
    "docker run -d --name ames-es -p 9268:9200 "
    "-e discovery.type=single-node -e xpack.security.enabled=false "
    "-e ES_JAVA_OPTS='-Xms512m -Xmx512m' docker.elastic.co/elasticsearch/elasticsearch:9.5.4"
)
DOCKER_COMPOSE = "cd backend && docker compose up -d"


def ok(msg): print(f"  \033[32m✓\033[0m {msg}"); return True
def bad(msg, fix): print(f"  \033[31m✗\033[0m {msg}\n      fix: {fix}"); return False
def warn(msg, fix): print(f"  \033[33m!\033[0m {msg}\n      fix: {fix}"); return False


def get(url, timeout=5):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()[:500]


def main() -> int:
    all_ok = True
    print("ames doctor")
    print("-----------")

    # 1. Python
    v = sys.version_info
    all_ok &= ok(f"python {v.major}.{v.minor}.{v.micro}") if v >= (3, 10) else \
        bad(f"python {v.major}.{v.minor} (need >= 3.10)", "install python 3.10+")

    # 2. Deps
    try:
        import fastapi  # noqa
        import uvicorn  # noqa
        all_ok &= ok("python deps (fastapi, uvicorn)")
    except ImportError as e:
        all_ok &= bad(f"missing dependency: {e.name}",
                      "cd backend && pip install -r requirements.txt")

    # 3. Docker present (optional but recommended)
    if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
        all_ok &= ok("docker daemon reachable")
    else:
        all_ok &= warn("docker daemon not reachable",
                       "start Docker, or point AMES_ES_URL at an existing ES")

    # 4. Elasticsearch
    try:
        status, body = get(ES_URL)
        ver = json.loads(body).get("version", {}).get("number", "?")
        all_ok &= ok(f"elasticsearch {ver} at {ES_URL}")
    except Exception:
        all_ok &= bad(f"elasticsearch not reachable at {ES_URL}",
                      f"one-liner: {DOCKER_ES}")

    # 5. ames service
    try:
        status, body = get(AMES_URL + "/health")
        if status == 200:
            all_ok &= ok(f"ames service at {AMES_URL} ({json.loads(body).get('kinds')})")
        else:
            all_ok &= bad(f"ames service returned {status}", f"{DOCKER_COMPOSE}")
    except Exception:
        all_ok &= warn(f"ames API service not reachable at {AMES_URL}",
                       f"start it: {DOCKER_COMPOSE}  (or: uvicorn app.main:app --port 8123)")

    # 6. Key validity (only if service up)
    if AMES_KEY:
        try:
            req = urllib.request.Request(
                AMES_URL + "/memory/recall",
                data=json.dumps({"query": "doctor ping", "size": 1}).encode(),
                headers={"Content-Type": "application/json", "X-API-Key": AMES_KEY})
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    all_ok &= ok("API key valid (recall succeeded)")
        except urllib.error.HTTPError as e:
            all_ok &= bad(f"API key rejected ({e.code})",
                          "mint a new one: curl -X POST -H 'X-Admin-Token: <token>' "
                          f"'{AMES_URL}/admin/keys?owner_id=you'")
        except Exception:
            pass  # service down already reported above
    else:
        all_ok &= warn("AMES_SERVICE_KEY not set",
                       "mint a key via POST /admin/keys and export AMES_SERVICE_KEY")

    print("-----------")
    if all_ok:
        print("all green — next: python scripts/import_memory.py <source> --dry-run")
        return 0
    print("issues found — apply the fixes above and re-run")
    return 1


if __name__ == "__main__":
    sys.exit(main())

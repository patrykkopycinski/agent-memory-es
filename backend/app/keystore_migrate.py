"""Copy a file-mode key store into the Elasticsearch key store.

    python -m app.keystore_migrate --from-file /data/ames_keys.json

Legacy plaintext name-keyed entries are normalized to digests first; only digests are written.
Idempotent: digests already present are left alone. Prints counts only, never tokens."""
import argparse
import json
import sys

from . import auth, keystore_es


def migrate(path: str) -> dict:
    with open(path) as f:
        keys = json.load(f)
    auth._normalize_legacy(keys)
    created = existing = skipped = 0
    for digest, entry in keys.items():
        if not isinstance(entry, dict) or not entry.get("owner_id"):
            skipped += 1    # name-keyed plaintext copies: their digest entry was just added
            continue
        if keystore_es.put_digest(digest, entry):
            created += 1
        else:
            existing += 1
    return {"created": created, "existing": existing, "skipped": skipped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-file", required=True, help="path to the file-mode key store")
    args = parser.parse_args(argv)
    try:
        counts = migrate(args.from_file)
    except (OSError, ValueError, auth.KeyStoreUnavailable) as exc:
        print(f"migration failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

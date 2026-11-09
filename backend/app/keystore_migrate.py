"""Copy a file-mode key store into the Elasticsearch key store.

    python -m app.keystore_migrate --from-file /data/ames_keys.json

Legacy plaintext name-keyed entries are normalized to digests first; only digests are written.
Digests already present are left alone, and entries with a malformed digest or unknown role are
skipped; an entry with no role is imported with the default role, as file mode treated it. A
successful run writes a sentinel in the key store and a second run is refused unless --force: a
rerun would resurrect keys revoked in Elasticsearch since. Retire the file after migration.
Prints counts only, never tokens."""
import argparse
import json
import re
import sys

from . import auth, keystore_es


class AlreadyMigrated(RuntimeError):
    pass


_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def migrate(path: str, force: bool = False) -> dict:
    if not force and keystore_es.migration_done():
        raise AlreadyMigrated("already migrated; retire the file or pass --force")
    with open(path) as f:
        keys = json.load(f)
    auth._normalize_legacy(keys)
    created = existing = skipped = 0
    for digest, entry in keys.items():
        if not isinstance(entry, dict) or not entry.get("owner_id"):
            skipped += 1    # name-keyed plaintext copies: their digest entry was just added
            continue
        if entry.get("role") is None:
            entry = dict(entry, role=auth.DEFAULT_ROLE)    # file mode accepted it: keep that key working
        if not _DIGEST_RE.fullmatch(str(digest)) or entry["role"] not in auth.ROLES:
            skipped += 1
            continue
        if keystore_es.put_digest(digest, entry):
            created += 1
        else:
            existing += 1
    keystore_es.mark_migrated()
    return {"created": created, "existing": existing, "skipped": skipped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-file", required=True, help="path to the file-mode key store")
    parser.add_argument("--force", action="store_true",
                        help="run again after a completed migration (resurrects keys revoked since)")
    args = parser.parse_args(argv)
    try:
        counts = migrate(args.from_file, args.force)
    except AlreadyMigrated as exc:
        print(f"migration refused: {exc}", file=sys.stderr)
        return 1
    except auth.KeyStoreUnavailable as exc:
        print(f"migration failed: {exc}", file=sys.stderr)    # operation + exception class only
        return 1
    except (OSError, ValueError) as exc:
        print(f"migration failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

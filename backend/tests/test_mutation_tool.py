"""Tests for the mutation-proof driver (tests/tools/mutation_proof.py).

The old shell script had no verdict: a surviving mutant, or a mutation that never applied, still
exited 0, and an interrupt left the tree mutated. These tests pin the fixes:
  #1 a surviving mutant is reported and the process does NOT exit 0
  #2 a hung suite times out and the tree is still restored
  #3 a pattern that no longer matches is INVALID, never a silent no-op
  #4 the restore is verified byte-identical against the pre-run hash
  #6 the verdict comes from the exit code, so a suite that ERRORs (not FAILs) counts as red
"""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))     # backend/
sys.path.insert(0, os.path.join(HERE, "tools"))

import mutation_proof  # noqa: E402

TARGET = "target.py"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _repo(tmp_path, body="VALUE = 1\n", check="import sys; sys.exit(0 if VALUE == 1 else 1)\n"):
    (tmp_path / TARGET).write_text(body)
    (tmp_path / "check.py").write_text(check)
    return str(tmp_path)


def _table(tmp_path, mutants):
    p = tmp_path / "table.json"
    p.write_text(json.dumps({"mutants": mutants}))
    return str(p)


def _mut(name="M", find="VALUE = 1", replace="VALUE = 2", expect="kill", f=TARGET):
    m = {"name": name, "file": f, "find": find, "replace": replace}
    if expect:
        m["expect"] = expect
    return m


def _prover(root, table, command="python3 check.py", timeout=30):
    return mutation_proof.Prover(root, command, table, timeout, root + "/logs")


def test_killed_mutant_gives_exit_zero_and_restores(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut()])
    p = _prover(root, table)
    assert p.run() == 0
    assert open(os.path.join(root, TARGET)).read() == "VALUE = 1\n"      # #4 byte-identical


def test_surviving_mutant_is_reported_and_fails_the_run(tmp_path):
    root = _repo(tmp_path)
    # the check does not read target.py, so mutating it cannot turn the suite red
    table = _table(tmp_path, [_mut()])
    p = _prover(root, table, command="python3 -c \"import sys; sys.exit(0)\"")
    assert p.run() == 1                                                  # #1 no silent pass
    assert open(os.path.join(root, TARGET)).read() == "VALUE = 1\n"


def test_pattern_that_does_not_apply_is_invalid_not_a_no_op(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut(find="VALUE = 999")])
    p = _prover(root, table)
    assert p.run() == 1                                                  # #3
    assert open(os.path.join(root, TARGET)).read() == "VALUE = 1\n"


def test_a_no_op_mutation_is_invalid(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut(replace="VALUE = 1")])                # identical text
    assert _prover(root, table).run() == 1


def test_suite_that_errors_during_collection_counts_as_red(tmp_path):
    root = _repo(tmp_path, check="raise ImportError('boom')\n")
    table = _table(tmp_path, [_mut()])
    assert _prover(root, table).run() == 0                               # #6 exit code, not grep


def test_timeout_is_reported_and_tree_restored(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut()])
    p = _prover(root, table, command="sleep 5", timeout=1)
    assert p.run() == 1                                                  # #2
    assert open(os.path.join(root, TARGET)).read() == "VALUE = 1\n"


def test_restore_is_verified_against_the_pre_run_hash(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut()])
    p = _prover(root, table)
    p.run()
    assert _sha(os.path.join(root, TARGET)) == p.pristine[os.path.join(root, TARGET)]


def test_restore_repairs_a_file_left_in_any_state(tmp_path):
    root = _repo(tmp_path)
    p = _prover(root, _table(tmp_path, [_mut()]))
    with open(os.path.join(root, TARGET), "w") as fh:      # as if a mutant was interrupted
        fh.write("VALUE = 99\\n# half-applied\\n")
    assert p.restore() is True
    assert open(os.path.join(root, TARGET)).read() == "VALUE = 1\n"


def test_expect_survive_marks_a_documented_equivalent_mutant(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut(expect="survive")])
    p = _prover(root, table, command="python3 -c \"import sys; sys.exit(0)\"")
    assert p.run() == 0                                                  # documented, not hidden


def test_table_targeting_a_missing_file_is_refused(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut(f="nope.py")])
    with pytest.raises(SystemExit):
        _prover(root, table)


def test_full_log_is_written_next_to_the_run(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut(name="M-log")])
    p = _prover(root, table)
    p.run()
    assert os.path.isfile(os.path.join(root, "logs", "mutant-M-log.log"))


def test_mutant_name_with_path_characters_still_logs(tmp_path):
    root = _repo(tmp_path)
    table = _table(tmp_path, [_mut(name="M15 duplicate/odd indices")])
    p = _prover(root, table)
    assert p.run() == 0
    logs = os.listdir(os.path.join(root, "logs"))
    assert logs == ["mutant-M15_duplicate_odd_indices.log"], logs


def test_restore_recreates_a_file_removed_by_someone_else(tmp_path):
    root = _repo(tmp_path)
    p = _prover(root, _table(tmp_path, [_mut()]))
    os.remove(os.path.join(root, TARGET))          # removed after the run started, dir still there
    assert p.restore() is True                     # recreated from the in-memory copy
    assert open(os.path.join(root, TARGET)).read() == "VALUE = 1\n"


def test_restore_reports_false_without_raising_when_the_directory_is_gone(tmp_path):
    import shutil
    root = _repo(tmp_path)
    p = _prover(root, _table(tmp_path, [_mut()]))
    shutil.rmtree(root)                            # e.g. an outer owner cleaned its temp dir
    assert p.restore() is False                    # reported, NOT raised (an atexit raise masks it)

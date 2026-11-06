#!/usr/bin/env python3
"""Skip gate for scripts/test-remote.sh.

    skipgate.py HEAD.xml BASE.xml     one line per lost test: "OK<TAB>id<TAB>reason" or "BAD<TAB>id<TAB>reason"
    skipgate.py --self-test           offline fixture tests (pure Python, no ES, no host)
    exit: 0 ok | 2 a junit file is missing or unparseable (infra: the caller must not read it as "no skips")

A test is LOST when it passed on base and is skipped on head. A whole-file skip (module-level pytest.skip,
importorskip, require_llm) is ONE junit entry with an empty classname and name="tests.test_x"; it is expanded
to every base-passing test of that module.

A lost test is excused ("OK") only when the head skip is the shared LLM-gateway probe
("LLM gateway <url> (AMES_LLM_BASE) is unreachable (<Exc>); this test needs a live LLM", tests/_llm_gate.py)
AND base shows the same gateway-down evidence for that test (gw_ids) or for its whole module (gw_mods):
  - gw_mods: module-level base evidence only, i.e. a whole-file entry (empty classname) that is skipped or
    errored with the gateway reason (a collection error counts);
  - gw_ids: per-test base evidence (a skipped/failed/errored test with the gateway reason).
One gateway-skipped test never excuses its siblings, and a generic connection error is not gateway evidence.
Anything else is a head regression ("BAD").
"""
import re
import sys
import xml.etree.ElementTree as ET

GW = re.compile(r"LLM gateway \S+ \(AMES_LLM_BASE\) is unreachable \(\w+\); this test needs a live LLM")


def _text(el):
    return " ".join(((el.get("message") or "") + " " + (el.text or "")).split())


def _strip_class(dotted):
    return re.sub(r"\.[A-Z]\w*$", "", dotted)  # drop a Test class segment


def _tid(tc):
    return f'{tc.get("classname")}::{tc.get("name")}'


def _kind(tc):
    for k in ("skipped", "failure", "error"):
        if tc.find(k) is not None:
            return k
    return "pass"


def analyze(head, base):
    """head/base: lists of <testcase> elements. Returns [(verdict, test_id, reason)]."""
    base_pass = [tc for tc in base if tc.get("classname") and _kind(tc) == "pass"]
    base_pass_ids = {_tid(tc) for tc in base_pass}
    gw_ids, gw_mods = set(), set()
    for tc in base:
        k = _kind(tc)
        if k == "pass" or not GW.search(_text(tc.find(k))):
            continue
        if tc.get("classname"):
            gw_ids.add(_tid(tc))
        else:
            gw_mods.add(tc.get("name"))
    out = []
    for tc in head:
        sk = tc.find("skipped")
        if sk is None:
            continue
        reason = _text(sk)
        if tc.get("classname"):
            lost = [_tid(tc)] if _tid(tc) in base_pass_ids else []
        else:
            m = tc.get("name")
            lost = [_tid(b) for b in base_pass if b.get("classname") == m or b.get("classname").startswith(m + ".")]
        for t in lost:
            excused = bool(GW.search(reason)) and (t in gw_ids or _strip_class(t.split("::")[0]) in gw_mods)
            out.append(("OK" if excused else "BAD", t, reason[:160]))
    return out


def _load(path):
    """A missing or unparseable junit file raises: reading it as 'no tests' would hide every lost test."""
    return list(ET.parse(path).getroot().iter("testcase"))


# ------------------------------------------------------------------ self test
_R = "LLM gateway http://x/v1 (AMES_LLM_BASE) is unreachable (URLError); this test needs a live LLM"


def _tcs(xml):
    return list(ET.fromstring("<testsuites><testsuite>" + xml + "</testsuite></testsuites>").iter("testcase"))


def _case(name, head, base, expect):
    """expect: {test_id: 'OK'|'BAD'}; any lost test not listed (or listed but not reported) fails the case."""
    got = {t: v for v, t, _ in analyze(_tcs(head), _tcs(base))}
    ok = got == expect
    print(("PASS" if ok else "FAIL") + "  " + name + ("" if ok else f"\n      expected {expect}\n      got      {got}"))
    return ok


def self_test():
    ok = True
    # (a) reproducer: base module has 2 passing tests + 1 gateway-skipped test; head skips the whole file
    ok &= _case(
        "(a) one gateway-skipped sibling does not excuse passing tests (whole-file head skip)",
        f'<testcase classname="" name="tests.test_memory_api"><skipped message="s">{_R}</skipped></testcase>',
        '<testcase classname="tests.test_memory_api" name="test_retain"/>'
        '<testcase classname="tests.test_memory_api" name="test_recall"/>'
        f'<testcase classname="tests.test_memory_api" name="test_reflect_no_llm"><skipped message="s">{_R}</skipped></testcase>',
        {"tests.test_memory_api::test_retain": "BAD", "tests.test_memory_api::test_recall": "BAD"},
    )
    # (b) base whole-module gateway skip + head whole-module gateway skip: nothing passed on base, nothing is lost
    ok &= _case(
        "(b) base whole-module gateway skip + head whole-module gateway skip: no BAD",
        f'<testcase classname="" name="tests.test_llm"><skipped message="s">{_R}</skipped></testcase>',
        f'<testcase classname="" name="tests.test_llm"><skipped message="s">{_R}</skipped></testcase>',
        {},
    )
    # (b') same, with base module evidence AND base-passing tests of that module (e.g. a rerun): module evidence excuses
    ok &= _case(
        "(b') base module-level gateway evidence excuses that module's lost tests (and not another module's)",
        f'<testcase classname="" name="tests.test_llm"><skipped message="s">{_R}</skipped></testcase>'
        f'<testcase classname="" name="tests.test_other"><skipped message="s">{_R}</skipped></testcase>',
        f'<testcase classname="" name="tests.test_llm"><error message="collection">{_R}</error></testcase>'
        '<testcase classname="tests.test_llm" name="test_x"/>'
        '<testcase classname="tests.test_other" name="test_y"/>',
        {"tests.test_llm::test_x": "OK", "tests.test_other::test_y": "BAD"},
    )
    # (c) per-test gateway skip on base, same test gateway-skipped on head: it did not pass on base, so it is not lost
    ok &= _case(
        "(c) test gateway-skipped on base and on head: not lost, no BAD",
        f'<testcase classname="tests.test_a" name="test_1"><skipped message="s">{_R}</skipped></testcase>',
        f'<testcase classname="tests.test_a" name="test_1"><skipped message="s">{_R}</skipped></testcase>',
        {},
    )
    # (c') per-test evidence (gw_ids) excuses that exact test only: id seen both gateway-skipped and passing on base
    ok &= _case(
        "(c') per-test base evidence excuses that test only, not its sibling",
        f'<testcase classname="tests.test_a" name="test_1"><skipped message="s">{_R}</skipped></testcase>'
        f'<testcase classname="tests.test_a" name="test_2"><skipped message="s">{_R}</skipped></testcase>',
        f'<testcase classname="tests.test_a" name="test_1"><skipped message="s">{_R}</skipped></testcase>'
        '<testcase classname="tests.test_a" name="test_1"/>'
        '<testcase classname="tests.test_a" name="test_2"/>',
        {"tests.test_a::test_1": "OK", "tests.test_a::test_2": "BAD"},
    )
    # (d) base passed, head whole-file skip for a non-gateway reason
    ok &= _case(
        "(d) base passed, head whole-file non-gateway skip",
        '<testcase classname="" name="tests.test_a"><skipped message="s">module skipped</skipped></testcase>',
        '<testcase classname="tests.test_a" name="test_1"/><testcase classname="tests.test_a.TestK" name="test_2"/>',
        {"tests.test_a::test_1": "BAD", "tests.test_a.TestK::test_2": "BAD"},
    )
    # (e) an ES connection error on base is not gateway evidence
    ok &= _case(
        "(e) base URLError failure is not gateway evidence",
        f'<testcase classname="tests.test_j" name="t1"><skipped message="s">{_R}</skipped></testcase>',
        '<testcase classname="tests.test_j" name="t1"/>'
        '<testcase classname="tests.test_j" name="t2"><failure message="boom">URLError: ES refused connection</failure></testcase>',
        {"tests.test_j::t1": "BAD"},
    )
    # (f) gateway-down head skip on a test that passed on base with no gateway evidence anywhere
    ok &= _case(
        "(f) head gateway skip, base passed, no base gateway evidence",
        f'<testcase classname="tests.test_g" name="t"><skipped message="s">{_R}</skipped></testcase>',
        '<testcase classname="tests.test_g" name="t"/>',
        {"tests.test_g::t": "BAD"},
    )
    # (g) missing / unparseable junit is an error (exit 2), never an empty result
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.xml")
        bad = os.path.join(d, "bad.xml")
        with open(good, "w") as fh:
            fh.write("<testsuites><testsuite><testcase classname='tests.a' name='t'/></testsuite></testsuites>")
        with open(bad, "w") as fh:
            fh.write("<testsuites><testsuite>")
        for label, args in (("missing head", [os.path.join(d, "nope.xml"), good]),
                            ("missing base", [good, os.path.join(d, "nope.xml")]),
                            ("unparseable head", [bad, good]),
                            ("unparseable base", [good, bad])):
            rc = subprocess.run([sys.executable, os.path.abspath(__file__)] + args,
                                capture_output=True, text=True).returncode
            good_rc = rc == 2
            print(("PASS" if good_rc else "FAIL") + f"  (g) {label} junit -> exit 2 (got {rc})")
            ok &= good_rc
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), good, good],
                            capture_output=True, text=True).returncode
        print(("PASS" if rc == 0 else "FAIL") + f"  (g) two valid junit files -> exit 0 (got {rc})")
        ok &= rc == 0
    print("SELF-TEST " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) != 3:
        sys.exit("usage: skipgate.py HEAD.xml BASE.xml | --self-test")
    try:
        head_tcs, base_tcs = _load(sys.argv[1]), _load(sys.argv[2])
    except (OSError, ET.ParseError) as e:
        print(f"skipgate: cannot read junit: {e}", file=sys.stderr)
        sys.exit(2)
    for verdict, t, reason in analyze(head_tcs, base_tcs):
        print(f"{verdict}\t{t}\t{reason}")

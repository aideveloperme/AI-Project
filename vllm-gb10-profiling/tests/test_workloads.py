import pytest

from servebench.workloads import Workload, WorkloadSpec, for_level


def test_deterministic():
    spec = WorkloadSpec(name="x", input_tokens=50)
    a = [Workload(spec).next().messages for _ in range(1)]
    b = [Workload(spec).next().messages for _ in range(1)]
    assert a == b


def test_unique_prompts_have_no_common_prefix():
    w = Workload(WorkloadSpec(name="x", input_tokens=50))
    p1, p2 = w.next().messages[0]["content"], w.next().messages[0]["content"]
    assert p1.split()[0] != p2.split()[0]


def test_levels_get_fresh_prompts_but_same_shared_prefixes():
    spec = WorkloadSpec(name="rag", kind="shared_prefix", shared_prefix_tokens=64, num_prefixes=1, input_tokens=16)
    a = Workload(for_level(spec, 1)).next().messages
    b = Workload(for_level(spec, 8)).next().messages
    assert a[0] == b[0]  # system prefix identical -> cacheable
    assert a[1] != b[1]  # unique part differs -> no cross-level leakage


def test_bad_kind():
    with pytest.raises(ValueError):
        Workload(WorkloadSpec(name="x", kind="nope"))

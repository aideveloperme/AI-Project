import pytest

from sentinel.rca.engine import node_severity
from sentinel.simulator.faults import GPU_SCOPED, Fault, FaultType
from tests.helpers import Pipeline

EXPECTED = {
    FaultType.THERMAL: "thermal_throttling",
    FaultType.POWER_CAP: "power_throttling",
    FaultType.MEMORY_BW: "hbm_degradation",
    FaultType.CLOCK_REDUCTION: "clock_misconfiguration",
    FaultType.ECC_ERRORS: "ecc_hardware_fault",
    FaultType.NVLINK_DEGRADATION: "nvlink_degradation",
    FaultType.PCIE_DEGRADATION: "pcie_degradation",
    FaultType.CPU_BOTTLENECK: "cpu_bottleneck",
    FaultType.HOST_MEMORY_PRESSURE: "host_memory_pressure",
    FaultType.NETWORK_DEGRADATION: "network_degradation",
    FaultType.COMMUNICATION_BOTTLENECK: "communication_bottleneck",
    FaultType.APP_REGRESSION: "software_regression",
}


@pytest.fixture(scope="module")
def pipeline():
    p = Pipeline(seed=3)
    p.step(40)
    return p


@pytest.mark.parametrize("fault", list(FaultType))
def test_top_hypothesis_matches_injected_fault(pipeline, fault):
    p = pipeline
    p.sim.inject(Fault(fault, "gpu-04", gpu=3 if fault in GPU_SCOPED else None, severity=0.8))
    p.step(12)
    ev, hyps = p.diagnose("gpu-04")
    p.sim.clear()
    p.step(30)
    assert hyps, f"no hypothesis for {fault}"
    assert hyps[0].rule_id == EXPECTED[fault], [(h.rule_id, h.confidence) for h in hyps]
    top = hyps[0]
    assert top.evidence_for, "hypotheses must preserve evidence"
    assert 0.3 <= top.confidence <= 0.9
    assert top.recommended_actions
    # healthy peers produce no hypotheses
    _, other = p.diagnose("gpu-01")
    assert other == []


def test_software_regression_confidence_capped(pipeline):
    p = pipeline
    p.sim.inject(Fault(FaultType.APP_REGRESSION, "gpu-05", severity=0.9))
    p.step(10)
    ev, hyps = p.diagnose("gpu-05")
    p.sim.clear()
    p.step(30)
    assert hyps[0].rule_id == "software_regression"
    assert hyps[0].confidence_label != "high"  # never certain without hardware evidence
    assert node_severity(ev, hyps) == "critical"  # ≥20% throughput loss


def test_confidence_high_requires_hard_evidence(pipeline):
    p = pipeline
    p.sim.inject(Fault(FaultType.THERMAL, "gpu-06", gpu=1, severity=1.0))
    p.step(12)
    ev, hyps = p.diagnose("gpu-06")
    p.sim.clear()
    p.step(30)
    assert hyps[0].confidence_label == "high"
    assert any("thermal" in e.lower() for e in hyps[0].evidence_for)

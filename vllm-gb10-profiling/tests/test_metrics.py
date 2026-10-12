from servebench.client import RequestResult
from servebench.metrics import summarize


def _r(i, ttft, e2e, out=11, ok=True):
    return RequestResult(
        request_id=str(i),
        ok=ok,
        ttft_s=ttft,
        e2e_s=e2e,
        prompt_tokens=100,
        completion_tokens=out,
        itl_s=[0.01] * (out - 1),
    )


def test_summary_basic():
    rs = [_r(i, 0.1 * (i + 1), 1.0 + 0.1 * i) for i in range(10)]
    s = summarize(rs, wall_s=2.0)
    assert s["num_ok"] == 10 and s["error_rate"] == 0
    assert s["output_tok_per_s"] == 55.0  # 110 tokens / 2s
    assert s["ttft_ms"]["p50"] == 550.0
    assert s["itl_ms"]["p50"] == 10.0


def test_tpot_definition():
    r = _r(0, 0.2, 1.2, out=11)
    assert abs(r.tpot_s - 0.1) < 1e-9  # (1.2 - 0.2) / (11 - 1)


def test_goodput_respects_slo_and_errors():
    rs = [_r(0, 0.1, 1.1), _r(1, 3.0, 4.0), _r(2, None, None, ok=False)]
    rs[2].error, rs[2].status_code = "boom", 500
    s = summarize(rs, wall_s=1.0, slo_ttft_ms=1000)
    assert s["goodput_rps"] == 1.0
    assert s["slo_attainment"] == round(1 / 3, 4)
    assert s["errors"] == {"500: boom": 1}


def test_empty():
    s = summarize([], wall_s=1.0)
    assert s["num_requests"] == 0 and s["ttft_ms"]["p50"] is None

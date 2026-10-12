from servebench.prom import counter_delta, parse_prometheus

TEXT = """# HELP vllm:num_requests_running x
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 2.0
vllm:prefix_cache_queries_total{engine="0"} 1000.0
vllm:prefix_cache_hits_total{engine="0"} 250.0
vllm:kv_cache_usage_perc{engine="0"} NaN
"""


def test_parse_sums_label_sets_and_skips_nan():
    m = parse_prometheus(TEXT)
    assert m["vllm:num_requests_running"] == 5.0
    assert "vllm:kv_cache_usage_perc" not in m


def test_counter_delta_hit_rate():
    before = parse_prometheus(TEXT)
    after = dict(before)
    after["vllm:prefix_cache_queries_total"] += 400
    after["vllm:prefix_cache_hits_total"] += 300
    d = counter_delta(before, after)
    assert d["prefix_cache_queries"] == 400 and d["prefix_cache_hit_rate"] == 0.75
    assert d["preemptions"] is None  # metric absent -> unknown, not zero

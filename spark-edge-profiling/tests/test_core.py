"""CPU-only correctness tests (no GPU, no downloads): run with `pytest -q tests`."""

import torch
import torch.nn as nn
import torchvision

from sparkprof.fusion import count_ops, fold_all_bn, fold_bn_into_conv, fuse_resnet_conv_bn_relu, fused_units
from sparkprof.models import GatedMHA, convert_vit_attention
from sparkprof.roofline import annotate_costs, classify, summary
from sparkprof.surgery import head_importance, prune_heads, prune_vit_heads, replace_activation
from sparkprof.timing import LayerTimer, summarize


def _randomize_bn(model):
    g = torch.Generator().manual_seed(0)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.1)
    return model.eval()


def test_fold_bn_single():
    conv, bn = nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16)
    _randomize_bn(nn.Sequential(conv, bn))
    x = torch.randn(2, 8, 12, 12)
    torch.testing.assert_close(fold_bn_into_conv(conv, bn)(x), bn(conv(x)), atol=1e-5, rtol=1e-4)


def test_resnet_fusion_levels_match():
    m = _randomize_bn(torchvision.models.resnet18(num_classes=10))
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref = m(x)
        torch.testing.assert_close(fold_all_bn(m)(x), ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(fuse_resnet_conv_bn_relu(m)(x), ref, atol=1e-4, rtol=1e-4)
    m50 = _randomize_bn(torchvision.models.resnet50(num_classes=10))
    with torch.no_grad():
        torch.testing.assert_close(fuse_resnet_conv_bn_relu(m50)(x), m50(x), atol=1e-3, rtol=1e-3)
    ops = count_ops(fuse_resnet_conv_bn_relu(m50))
    assert ops["bn"] == 0 and ops["fused"] == 53 and ops["conv"] == 0


def test_fused_units_keys_align():
    m = _randomize_bn(torchvision.models.resnet50(num_classes=10))
    x = torch.randn(1, 3, 64, 64)
    with LayerTimer(m) as lt:
        u0 = fused_units(lt.run(x, warmup=0, iters=2))
    f = fuse_resnet_conv_bn_relu(m)
    with LayerTimer(f) as lt:
        u2 = fused_units(lt.run(x, warmup=0, iters=2))
    common = set(u0) & set(u2)
    assert "conv1" in common and "layer1.0.conv1" in common and "layer1.0.downsample" in common
    assert "layer1.0.relu" in common  # post-residual ReLU exists before and after


def test_gated_mha_matches_torch():
    mha = nn.MultiheadAttention(64, 8, batch_first=True).eval()
    g = GatedMHA.from_torch_mha(mha).eval()
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        torch.testing.assert_close(g(x, x, x)[0], mha(x, x, x, need_weights=False)[0], atol=1e-5, rtol=1e-4)


def _tiny_vit():
    m = torchvision.models.VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=4,
                                             hidden_dim=32, mlp_dim=64, num_classes=10)
    return convert_vit_attention(m).eval()


def test_head_pruning():
    m = _tiny_vit()
    attn = m.encoder.layers[0].self_attention
    x = torch.randn(2, 5, 32)
    # pruning a head == zeroing it through the gate
    attn.use_gate, attn.head_gate = True, torch.tensor([1.0, 0.0, 1.0, 1.0])
    with torch.no_grad():
        gated = attn(x)[0]
    attn.use_gate = False
    with torch.no_grad():
        torch.testing.assert_close(prune_heads(attn, [0, 2, 3])(x)[0], gated, atol=1e-5, rtol=1e-4)

    loader = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(2)]
    scores = head_importance(m, loader, torch.device("cpu"), batches=2)
    assert all(s.shape == (4,) for s in scores.values())
    pruned, kept = prune_vit_heads(m, scores, 0.5)
    assert all(len(k) == 2 for k in kept.values())
    assert pruned(torch.randn(1, 3, 32, 32)).shape == (1, 10)
    assert sum(p.numel() for p in pruned.parameters()) < sum(p.numel() for p in m.parameters())


def test_replace_activation():
    m = torchvision.models.efficientnet_b0(num_classes=10)
    r, n = replace_activation(m)
    assert n > 0 and not any(isinstance(x, nn.SiLU) for x in r.modules())
    assert r(torch.randn(1, 3, 64, 64)).shape == (1, 10)


def test_cost_model_resnet50():
    m = torchvision.models.resnet50(num_classes=1000).eval()
    with LayerTimer(m) as lt:
        rows = annotate_costs(m, lt.run(torch.randn(8, 3, 224, 224), warmup=0, iters=1))
    conv_fc = sum(r["flops"] for r in rows if r["type"] in ("Conv2d", "Linear")) / 8
    assert 8.0e9 < conv_fc < 8.4e9  # ResNet-50 ≈ 4.1 GMAC per image
    # FP32 CUDA-core roof on GB10: ridge = 31e12 / 273e9 ≈ 114 FLOP/B
    classify(rows, peak_tflops=31, peak_gbs=273)
    s = summary(rows)
    assert s["memory_bound_layers"] > 0 and s["compute_bound_layers"] > 0
    # elementwise ops are always memory-bound; 3x3 convs in layer1 are compute-bound
    bound = {r["layer"]: r["bound"] for r in rows}
    assert bound["layer1.0.bn1"] == "memory" and bound["layer1.0.conv2"] == "compute"


def test_cost_model_vit():
    m = _tiny_vit()
    with LayerTimer(m) as lt:
        rows = annotate_costs(m, lt.run(torch.randn(1, 3, 32, 32), warmup=0, iters=1))
    attn = [r for r in rows if r["type"] == "GatedMHA"]
    assert len(attn) == 2 and all(r["flops"] > 0 for r in attn)


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0], batch=4)
    assert s["mean_ms"] == 2.5 and s["throughput_ips"] == 1600.0 and s["p50_ms"] in (2.0, 3.0)

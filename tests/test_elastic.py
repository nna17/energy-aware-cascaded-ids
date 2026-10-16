import numpy as np
import torch

from eids.models.elastic import ElasticMLP, SearchSpace, SubnetConfig, extract_subnet, param_count
from eids.training.bn_recalib import recalibrate_bn


def _space(n_features=10):
    return SearchSpace(n_features, [16, 16, 8], [0.25, 0.5, 1.0], [1, 2, 3], [0.5, 1.0], [32])


def test_subnet_forward_shapes():
    space = _space()
    net = ElasticMLP(10, 4, space.max_widths)
    x = torch.randn(32, 10)
    for cfg in (space.largest(), space.smallest(), space.sample(np.random.default_rng(0))):
        net.set_active(cfg)
        assert net(x).shape == (32, 4)


def test_extracted_subnet_matches_supernet():
    torch.manual_seed(0)
    space = _space()
    net = ElasticMLP(10, 4, space.max_widths)
    cfg = SubnetConfig(depth=2, widths=(8, 4), n_features=5)
    net.set_active(cfg)
    x = torch.randn(256, 10)
    recalibrate_bn(net, [(x, None)] * 3, n_batches=3)
    static = extract_subnet(net, cfg)
    with torch.no_grad():
        assert torch.allclose(net(x), static(x), atol=1e-5)
    linears = [m for m in static.net if isinstance(m, torch.nn.Linear)]
    assert sum(m.weight.numel() + m.bias.numel() for m in linears) == param_count(cfg, 4)


def test_sub_models_share_weights():
    space = _space()
    net = ElasticMLP(10, 4, space.max_widths)
    n_params = sum(p.numel() for p in net.parameters())
    net.set_active(space.smallest())
    assert sum(p.numel() for p in net.parameters()) == n_params  # nothing duplicated

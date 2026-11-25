import json
from pathlib import Path

import torch

from scripts.audit_dataset import audit
from scripts.sdr002_diagnostics import gradients
from sdr_mamba.data_v2 import generate, pixel_bfs, window_graph_distances
from sdr_mamba.model_v2 import WindowSSM, raw_readout_gradient, zero_neighbor
from sdr_mamba.scan import tile
from sdr_mamba.state_transport import initial_states


def test_window_edges_follow_target_pixel_connectivity_and_pulse_order():
    # The target must leave the top window row and reenter it to reach the right.
    occupancy = {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (1, 3), (0, 3)}
    path = pixel_bfs(occupancy | {(0, 2)}, (0, 0), 4)
    assert path[(0, 3)] == 7 and path[(0, 2)] == 8
    graph = window_graph_distances(path, (0, 0), 2, 4)
    assert graph == {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}
    bank = torch.zeros(2, 2, 2, 4, 1, 1, 1)
    bank[0, 0, 0, 2] = 8
    for y, x, direction, next_direction, amplitude in ((1, 0, 2, 0, 4), (1, 1, 0, 3, 2), (0, 1, 3, 1, 1)):
        received = initial_states(bank, "sst", .5, 1)
        expected = torch.zeros_like(bank)
        expected[0, y, x, direction] = amplitude
        torch.testing.assert_close(received, expected)
        # Rotate the impulse after receipt, modeling the direction change at a junction.
        bank = torch.zeros_like(bank)
        bank[0, y, x, next_direction] = received[0, y, x, direction]


def test_data_hash_balance_same_map_counterfactual_and_independent_dfs():
    config = json.loads((Path(__file__).parents[1] / "configs/sdr002.json").read_text())
    for entry in config["data"].values():
        entry["maps"] = 3
    datasets, manifest = generate(config)
    assert generate(config)[1] == manifest
    assert audit(datasets, manifest, config)["status"] == "PASS"
    for data in datasets.values():
        for image, label, pixel in zip(data["image"], data["target"], data["pixel"]):
            occupancy, seed = image.bool().unbind(-1)
            root = tuple(seed.nonzero()[0].tolist())
            seen, pending = set(), [root]
            while pending:
                y, x = pending.pop()
                if (y, x) in seen:
                    continue
                seen.add((y, x))
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < 32 and 0 <= xx < 32 and occupancy[yy, xx]:
                        pending.append((yy, xx))
            assert seen == {tuple(point) for point in label.nonzero().tolist()}
            assert torch.equal(pixel >= 0, label)


def small_model():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    return WindowSSM(k=2, d_model=8, d_state=2, expand=1, headdim=4, groups=1, gate=.5, mixing=.6)


def test_matched_feature_transport_and_no_cross_sample_leakage():
    model = small_model()
    image = torch.randn(2, 4, 6, 2)
    result, trace = model(image, 3, "feature_neighbor", return_trace=True)
    assert trace[0].received_features is None
    assert trace[1].received_features.shape == trace[1].state.shape
    assert trace[1].initial is None
    expected = initial_states(trace[0].sent_features, "sst", .5, .6)
    torch.testing.assert_close(trace[1].received_features, expected)
    torch.testing.assert_close(model(image[:1], 3, "feature_neighbor"), result[:1])
    torch.testing.assert_close(model(image, 1, "feature_neighbor"), model(image, 1, "reset"))
    model.gate = 0
    torch.testing.assert_close(model(image, 3, "feature_neighbor"), model(image, 3, "reset"))


def test_current_final_readout_and_future_transport_are_distinct():
    model = small_model()
    image = torch.randn(2, 4, 6, 2)
    logits, trace = model(image, 3, "sst", return_trace=True)
    loss = logits.square().mean()
    direct = raw_readout_gradient(loss, trace[-1])
    assert direct.isfinite().all() and direct.norm() > 0
    assert torch.autograd.grad(loss, trace[-1].state, allow_unused=True, retain_graph=True)[0] is None
    for tensor in (trace[0].state, trace[1].initial):
        grad = torch.autograd.grad(loss, tensor, retain_graph=True)[0]
        assert grad.isfinite().all() and grad.norm() > 0
    rows = gradients(model, trace, [f.logits.square().mean() for f in trace], loss, {})
    assert all(r["current_raw_readout_grad_rms"] > 0 for r in rows)
    assert all(r["future_state_transport_grad_rms"] is None for r in rows if r["round"] == 3)
    assert all(r["directional_shared_projection_weight_grad_l2"] > 0 for r in rows)
    neighbor_zero = zero_neighbor(trace[0].state, .5, .6)
    d = trace[0].state.unbind(3)
    torch.testing.assert_close(neighbor_zero[:, :, :, 0], .2 * (d[2] + d[3]) / 2)


def test_oracle_state_impulse_only_changes_the_correct_next_window():
    model = small_model()
    model.mixing = 1
    features = model.encoder(torch.randn(2, 6, 8, 2))
    empty = model.step(features)
    pulse = torch.zeros_like(empty.state)
    pulse[0, 1, 1, 0] = 4
    changed = model.step(features, pulse)
    assert (changed.state[0, 1, 1, 0] - empty.state[0, 1, 1, 0]).abs().max() > 0

    def next_logits(frame):
        return model.step(features, initial_states(frame.state, "sst", .5, 1)).logits

    effect = tile((next_logits(changed) - next_logits(empty)).unsqueeze(-1), 2).abs().amax((3, 4, 5))
    expected = torch.zeros_like(effect, dtype=torch.bool)
    expected[0, 1, 2] = True
    torch.testing.assert_close(effect > 0, expected)

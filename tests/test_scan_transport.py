import pytest
import torch

from sdr_mamba.model import LoopedSSM
from sdr_mamba.scan import inverse, scan, tile, untile
from sdr_mamba.state_transport import initial_states, shuffle_sources


def test_scans_non_square_and_order():
    image = torch.arange(24).reshape(1, 4, 6, 1)
    windows = tile(image, 2)
    assert windows.shape == (1, 2, 3, 2, 2, 1)
    directional = scan(windows)
    assert directional[0, 0, 0, :, :, 0].tolist() == [
        [0, 1, 6, 7], [7, 6, 1, 0], [0, 6, 1, 7], [7, 1, 6, 0],
    ]
    for result in inverse(directional, 2).unbind(3):
        torch.testing.assert_close(untile(result), image)
    with pytest.raises(ValueError, match="multiples"):
        tile(image[:, :, :-1], 2)


@pytest.mark.parametrize("direction, displacement", list(enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)])))
def test_single_window_impulse_no_wrap(direction, displacement):
    for y, x in [(1, 2), (0, 0), (2, 4)]:
        bank = torch.zeros(2, 3, 5, 4, 1, 1, 1)
        bank[0, y, x, direction] = 10
        transported = initial_states(bank, "sst", gate=.5, mixing=1)
        dy, dx = displacement
        expected = torch.zeros_like(bank)
        if 0 <= y + dy < 3 and 0 <= x + dx < 5:
            expected[0, y + dy, x + dx, direction] = 5
        torch.testing.assert_close(transported, expected)


def test_shuffled_payload_count_and_perpendicular_component():
    own = torch.arange(30).reshape(2, 3, 5, 1).float()
    shuffled = shuffle_sources(own)
    for before, after in zip(own, shuffled):
        assert (before != after).all()
        torch.testing.assert_close(before.flatten().sort().values, after.flatten().sort().values)
    bank = torch.stack((own, own + 1, own + 10, own + 20), dim=3)
    sst = initial_states(bank, "sst", .5, .6)
    local = initial_states(bank, "local", .5, .6)
    torch.testing.assert_close(sst[:, :, 0, 0], .2 * (own[:, :, 0] + 15))
    torch.testing.assert_close(local[:, :, 0, 0], .3 * own[:, :, 0] + .2 * (own[:, :, 0] + 15))


@pytest.fixture
def example():
    torch.manual_seed(74)
    torch.set_num_threads(1)
    return LoopedSSM(k=2, d_model=8, d_state=3, headdim=4), torch.randn(2, 4, 6, 2)


def test_reset_equivalence_and_no_cross_image_leaks(example):
    model, x = example
    expected = model(x, 1, "reset")
    for mode in ("local", "sst", "shuffled", "feature_neighbor"):
        torch.testing.assert_close(model(x, 1, mode), expected, rtol=0, atol=0)
    result = model(x, 3, "sst")
    model(x.flip(0), 2, "shuffled")
    torch.testing.assert_close(model(x, 3, "sst"), result, rtol=0, atol=0)
    torch.testing.assert_close(model(x[:1], 3, "sst"), result[:1])
    model.gate = 0
    for mode in ("local", "sst", "shuffled", "feature_neighbor"):
        torch.testing.assert_close(model(x, 3, mode), model(x, 3, "reset"), rtol=0, atol=0)


def test_cross_round_initial_and_final_state_gradient(example):
    model, x = example
    _, rounds = model(x, 3, "sst", return_trace=True)
    assert rounds[0].initial is None
    assert rounds[0].final.shape == (2, 2, 3, 4, 4, 4, 3)
    # Recompute with fixed features to isolate the state path from feature recurrence.
    logits, _, _ = model.step(rounds[1].feature_in.detach(), rounds[1].initial)
    gradients = torch.autograd.grad(logits.square().mean(), (rounds[1].initial, rounds[0].final))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm().item() > 0

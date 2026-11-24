"""Small CPU smoke test for cascade gradients and checkpoint reconstruction."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from look2hear.models.apollo import Apollo, ApolloCascade, model_from_state


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    base = Apollo(sr=44100, win=20, feature_dim=256, layer=6)
    model = ApolloCascade(base)
    audio = torch.randn(1, 1, 2048) * 0.1
    for expand in (False, True):
        model.configure([3000, 10000], 1000, expand)
        model.zero_grad(set_to_none=True)
        output, stages = model(audio, return_specialists=True)
        assert output.shape == audio.shape and torch.isfinite(output).all()
        # High-only loss verifies gradients reach both earlier trainable tails.
        stages[-1].square().mean().backward()
        for tail in (model.net[4:], *model.specialist_tails):
            grads = [p.grad for p in tail.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(g.abs().max() > 0 for g in grads)
        assert all(p.grad is None for p in model.net[:4].parameters())
        assert all(p.grad is None for p in model.BN.parameters())
        state = model.state_dict()
        restored = model_from_state(state, 44100, 20, 256, 6)
        restored.load_state_dict(state, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(audio), output.detach())
        print(f"cascade expansion={expand}: OK")


if __name__ == "__main__":
    main()

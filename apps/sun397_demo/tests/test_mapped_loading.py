from types import SimpleNamespace

import pytest


def test_mapped_cpu_weights_preserve_outputs_and_nonpersistent_buffers(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from sun397_demo.worker import mapped_factory

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 3)
            # RoPE periods in DINOv3 are initialized buffers, not checkpoint parameters.
            self.register_buffer("periods", torch.tensor([2.0, 4.0, 8.0]), persistent=False)

        def forward(self, values):
            return self.projection(values) * self.periods

    original = Tiny().eval()
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": original.state_dict()}, path)
    state = torch.load(path, mmap=True, weights_only=True)
    factory = mapped_factory(SimpleNamespace(build_model=lambda repo, config: Tiny()), state)
    restored = factory(tmp_path, {}).to("cpu").eval()
    restored.load_state_dict(state["model_state_dict"], strict=True)
    assert torch.equal(restored.periods, original.periods)
    assert restored.projection.weight.data_ptr() == state["model_state_dict"]["projection.weight"].data_ptr()
    assert not any(p.is_meta for p in restored.parameters())
    inputs = torch.tensor([[0.2, 0.7]])
    torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        restored.load_state_dict({}, strict=True)


def test_readonly_checkpoint_preserves_shared_storage_and_original_file(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from sun397_demo.worker import readonly_checkpoint, mapped_factory
    import hashlib
    model = torch.nn.Linear(3, 2).eval()
    weights = model.state_dict()
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": weights, "alias": weights["weight"].T,
                "config": {"dtype": "float32"}}, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    state = readonly_checkpoint(path)
    assert state["alias"].untyped_storage().data_ptr() == state["model_state_dict"]["weight"].untyped_storage().data_ptr()
    assert state["alias"].stride() == weights["weight"].T.stride()
    factory = mapped_factory(SimpleNamespace(build_model=lambda repo, config: torch.nn.Linear(3,2)), state)
    restored = factory(tmp_path, {}).eval().requires_grad_(False)
    restored.load_state_dict(state["model_state_dict"], strict=True)
    with torch.inference_mode():
        inputs = torch.tensor([[0.3, 0.5, -0.4]])
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

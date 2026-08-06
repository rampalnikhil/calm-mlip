"""Tests for accelerator device resolution.

These tests must pass on CPU-only CI machines, so the PyTorch probe is always
driven through a fake ``torch`` module rather than the real one.
"""

from __future__ import annotations

import sys
import types

import pytest

from calm.calculators.exceptions import CalculatorSpecError
from calm.calculators.providers import GRACEProvider, resolve_device
from calm.calculators.spec import CalculatorSpec


def _install_fake_torch(monkeypatch, *, available, raises=False):
    """Install a fake ``torch`` whose accelerator probe is fully controlled."""

    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")

    def is_available():
        if raises:
            raise RuntimeError("driver probe exploded")
        return available

    cuda.is_available = is_available
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda)
    return torch


class TestResolveDevice:
    def test_none_resolves_to_cpu(self):
        assert resolve_device(None) == "cpu"

    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps", "xpu"])
    def test_explicit_devices_pass_through_untouched(self, device, monkeypatch):
        # An explicit device must never be rewritten, even when no accelerator
        # is visible; the backend owns that validation.
        _install_fake_torch(monkeypatch, available=False)
        assert resolve_device(device) == device

    def test_auto_selects_cuda_when_accelerator_visible(self, monkeypatch):
        _install_fake_torch(monkeypatch, available=True)
        assert resolve_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu_without_accelerator(self, monkeypatch):
        _install_fake_torch(monkeypatch, available=False)
        assert resolve_device("auto") == "cpu"

    def test_auto_falls_back_to_cpu_when_torch_missing(self, monkeypatch):
        # Non-PyTorch backends may be installed without torch present.
        monkeypatch.setitem(sys.modules, "torch", None)
        assert resolve_device("auto") == "cpu"

    def test_auto_falls_back_to_cpu_when_probe_raises(self, monkeypatch):
        # A broken or partially installed driver must degrade, not crash.
        _install_fake_torch(monkeypatch, available=False, raises=True)
        assert resolve_device("auto") == "cpu"


@pytest.fixture
def fake_tensorpotential(monkeypatch):
    """Minimal tensorpotential stand-in so GRACEProvider.create can be reached."""

    root = types.ModuleType("tensorpotential")
    calculator = types.ModuleType("tensorpotential.calculator")

    def grace_fm(model, **kwargs):
        return {"model": model, "kwargs": dict(kwargs)}

    calculator.grace_fm = grace_fm
    root.calculator = calculator
    monkeypatch.setitem(sys.modules, "tensorpotential", root)
    monkeypatch.setitem(sys.modules, "tensorpotential.calculator", calculator)
    monkeypatch.setattr(GRACEProvider, "is_available", lambda self: True)
    return root


class TestGraceDeviceHandling:
    """GRACE has no device argument to forward, so it must not accept one."""

    def test_explicit_accelerator_is_rejected(self, fake_tensorpotential, monkeypatch):
        _install_fake_torch(monkeypatch, available=True)
        spec = CalculatorSpec(family="grace", model="GRACE-1L-OMAT", device="cuda")
        with pytest.raises(CalculatorSpecError, match="cannot honor device"):
            GRACEProvider().create(spec)

    def test_cpu_is_accepted(self, fake_tensorpotential, monkeypatch):
        _install_fake_torch(monkeypatch, available=True)
        spec = CalculatorSpec(family="grace", model="GRACE-1L-OMAT", device="cpu")
        assert GRACEProvider().create(spec)["model"] == "GRACE-1L-OMAT"

    def test_auto_is_accepted_and_falls_back(self, fake_tensorpotential, monkeypatch):
        # "auto" is advisory: it must degrade to CPU rather than raise, so a
        # portable script can name it without special-casing GRACE.
        _install_fake_torch(monkeypatch, available=True)
        spec = CalculatorSpec(family="grace", model="GRACE-1L-OMAT", device="auto")
        assert GRACEProvider().create(spec)["model"] == "GRACE-1L-OMAT"

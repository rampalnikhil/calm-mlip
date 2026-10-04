"""Public projections and live-calculator workflows preserve exact provenance."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from calm.calculators.spec import CalculatorSpec
from calm.project.application.bulks import BulksService
from calm.public.inputs.materials import Material
from calm.public.inputs import project_config
from calm.public.persistence.adapter import WorkspaceAdapter
from calm.public.workflows.material_optimization import ProjectMaterialOptimizationService
from calm.public.workflows.materials import persist_material
from calm.public.workflows.saving import PublicProjectSaver
from test_helpers import CaptureReporter


@pytest.mark.parametrize("projection", ["object", "mapping", "direct_mapping"])
def test_material_roundtrip_keeps_exact_calculator(sqlite_uow_factory, projection):
    spec = CalculatorSpec(family="ase", model="LJ", version="test", source="local",
                          device="cpu", dtype="float64", options={"epsilon": 0.2})
    bulks = BulksService(uow_factory=sqlite_uow_factory)
    original = bulks.add_bulk(label="original", payload={"material": "Ar"}, optimized_with=spec)
    if projection == "direct_mapping":
        material = asdict(original)
    else:
        material = Material.from_workspace(asdict(original) if projection == "mapping" else original)
        assert material.optimized_with == original.optimized_with_calculator_uid_full
    saved = persist_material(workspace=SimpleNamespace(add_bulk=bulks.add_bulk),
                             obj=material, reporter=CaptureReporter())
    assert saved.uid_full == original.uid_full
    with sqlite_uow_factory() as uow:
        reopened = uow.bulks.get_by_uid_full(saved.uid_full)
        calculator = uow.calculators.get_by_uid_full(reopened.optimized_with_calculator_uid_full)
        assert calculator.spec == spec.to_dict()
        assert len(uow.calculators.list()) == 1


def test_live_calculator_spec_reaches_authoritative_saver(sqlite_uow_factory, monkeypatch):
    spec = CalculatorSpec(family="ase", model="LJ", options={"epsilon": 0.2})
    live = SimpleNamespace(calm_spec=spec)
    bulks = BulksService(uow_factory=sqlite_uow_factory)
    saver = PublicProjectSaver(workspace=WorkspaceAdapter(SimpleNamespace(add_bulk=bulks.add_bulk)))
    service = ProjectMaterialOptimizationService(project=SimpleNamespace(), saver=saver)
    optimized = Material(name="Ar", label="optimized", atoms=None, state="optimized_bulk")
    calls = []

    def optimize(material, **kwargs):
        calls.append(kwargs["calculator"])
        return optimized

    configured = []
    monkeypatch.setattr(service, "_optimize_structure", optimize)
    monkeypatch.setattr(project_config, "configure", lambda project, **kw: configured.append(kw))
    saved = service.optimize_material(Material(name="Ar", label=None, atoms=None),
                                      calculator=live, reporter=CaptureReporter())
    assert calls == [live]
    assert configured[0]["calculator"] == spec.to_dict()
    with sqlite_uow_factory() as uow:
        bulk = uow.bulks.get_by_uid_full(saved.uid_full)
        calculator = uow.calculators.get_by_uid_full(bulk.optimized_with_calculator_uid_full)
        assert calculator.spec == spec.to_dict()


def test_unknown_calculator_uid_does_not_create_default_spec(sqlite_uow_factory):
    service = BulksService(uow_factory=sqlite_uow_factory)
    with pytest.raises(ValueError, match="not registered"):
        service.add_bulk(payload={}, optimized_with="calc:v2:" + "0" * 64)
    with sqlite_uow_factory() as uow:
        assert uow.calculators.list() == []

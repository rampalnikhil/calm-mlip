from types import SimpleNamespace

import pytest

from calm.calculators.exceptions import CalculatorFingerprintError
from calm.calculators.identity import get_calculator_spec
from calm.calculators.spec import CalculatorSpec
from calm.public.inputs import project_config
from calm.public.inputs.materials import Material
from calm.public.workflows.material_optimization import ProjectMaterialOptimizationService
from test_helpers import CaptureReporter


@pytest.mark.parametrize("model", ["EMT", "LJ", "Morse"])
def test_builtin_live_calculator_parameters_are_preserved(model):
    pytest.importorskip("ase")
    from calm.calculators.api import make_calculator

    options = {"epsilon": 0.2} if model == "LJ" else {}
    live = make_calculator(CalculatorSpec(family="ase", model=model, options=options))
    inferred = get_calculator_spec(live)
    assert inferred.family == "ase" and inferred.model == model
    assert inferred.options_dict() == dict(live.parameters)
    reconstructed = make_calculator(inferred)
    assert dict(reconstructed.parameters) == dict(live.parameters)


@pytest.mark.parametrize("failure", ["unknown", "configured_conflict", "mlip_conflict", "both", "missing"])
def test_provenance_errors_precede_optimization(monkeypatch, failure):
    service = ProjectMaterialOptimizationService(project=SimpleNamespace(), saver=None)
    calls = []
    monkeypatch.setattr(service, "_optimize_structure", lambda *a, **kw: calls.append(kw))
    inputs = {"calculator": object()}
    exception = CalculatorFingerprintError
    if failure in {"configured_conflict", "mlip_conflict"}:
        inputs = {"calculator": SimpleNamespace(calm_spec=CalculatorSpec(family="ase", model="EMT"))}
        configuration = ({"default_calculator_uid_full": "calc:other"}
                         if failure == "configured_conflict" else {"default_mlip": "mace"})
        monkeypatch.setattr(project_config, "load_configuration", lambda project: configuration)
        exception = RuntimeError
    elif failure == "both":
        from calm.public.inputs.potentials import Potential

        inputs["potential"] = Potential(family="ase", model="EMT")
        exception = TypeError
    elif failure == "missing":
        inputs = {}
        exception = ValueError
    with pytest.raises(exception):
        service.optimize_material(Material(name="test", label=None, atoms=None),
                                  reporter=CaptureReporter(), **inputs)
    assert calls == []

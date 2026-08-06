"""Public Potential facade.

``Potential`` is a lightweight specification until ``calculator()`` or
``validate()`` is called. This keeps imports cheap and prevents accidental model
initialization during API exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Potential:
    """Describe a calculator or machine-learned interatomic potential lazily.

    The facade stores provider, model, device, and provider-specific options.
    ``device`` accepts any backend device string; ``"auto"`` resolves to the
    GPU when one is visible and to the CPU otherwise
    without importing or initializing the underlying calculator. Construction is
    deferred until ``calculator()``, ``validate()``, or another calculator-
    backed workflow is invoked.

    Successful construction establishes runtime availability only. It does not by
    itself establish scientific suitability for a material system.
    """

    family: str
    model: str | None = None
    device: str = "cpu"
    options: Dict[str, Any] = field(default_factory=dict)
    _calculator: Any | None = field(default=None, repr=False, compare=False)
    _external_calculator: bool = field(default=False, repr=False, compare=False)
    quiet: bool = False

    @property
    def name(self) -> str:
        return f"{self.family}:{self.model}" if self.model else str(self.family)

    @classmethod
    def grace(
        cls,
        model: str = "GRACE-1L-OMAT",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        return cls("grace", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def mace(
        cls,
        model: str = "medium-mpa-0",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        return cls("mace", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def chgnet(
        cls,
        model: str = "0.3.0",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        return cls("chgnet", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def sevenn(
        cls,
        model: str = "7net-0",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        return cls("sevenn", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def mattersim(
        cls,
        model: str = "default",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        return cls("mattersim", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def orb(
        cls,
        model: str = "orb-v3-conservative-inf-omat",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        return cls("orb", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def nequip(
        cls,
        model: str,
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        """NequIP/Allegro compiled model (foundation or bespoke) from a file path."""
        return cls("nequip", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def allegro(
        cls,
        model: str = "Allegro-OAM-L",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        """Allegro model: a nequip.net foundation name (default Allegro-OAM-L)
        or a path to a bespoke compiled artifact (.nequip.pth/.nequip.pt2)."""
        return cls("nequip", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def uma(
        cls,
        model: str = "uma-s-1p1",
        task: str = "omat",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        """Meta/FAIR-Chem UMA model. ``task`` selects the scientific domain
        head (omat/omol/oc20/odac/omc); the public constructor defaults to
        ``omat`` since CALM workflows target bulk inorganic materials."""
        options = dict(kwargs)
        options["task"] = task
        return cls("uma", model, device=device, options=options, quiet=quiet)

    @classmethod
    def petmad(
        cls,
        model: str = "latest",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        """PET-MAD universal potential: a released version tag or a path to
        a bespoke checkpoint."""
        return cls("petmad", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def matgl(
        cls,
        model: str = "M3GNet-PES-MatPES-PBE-2025.2",
        *,
        device: str = "cpu",
        quiet: bool = False,
        **kwargs,
    ) -> "Potential":
        return cls("matgl", model, device=device, options=dict(kwargs), quiet=quiet)

    @classmethod
    def from_ase(
        cls, calculator: Any, *, name: str = "ase-calculator", quiet: bool = False
    ) -> "Potential":
        if calculator is None:
            raise TypeError("Potential.from_ase requires a live calculator.")
        return cls(
            "ase",
            name,
            _calculator=calculator,
            _external_calculator=True,
            quiet=quiet,
        )

    @classmethod
    def available(cls):
        from calm.calculators.api import list_providers

        return tuple(list_providers())

    @classmethod
    def info(cls, name: str) -> dict[str, Any]:
        return {"name": name, "available_providers": cls.available()}

    def to_spec(self) -> dict[str, Any]:
        """Return the exact current CalculatorSpec mapping without importing backends."""

        if self._external_calculator:
            from calm.calculators.exceptions import CalculatorProvenanceError

            raise CalculatorProvenanceError(
                "Potential.from_ase(...) wraps a live calculator but does not "
                "define a reconstructible CalculatorSpec. Use a provider-backed "
                "Potential for persisted workflows or pass the live calculator "
                "only to a non-persisted calculation."
            )

        return {
            "schema_version": 1,
            "family": self.family,
            "model": self.model,
            "version": None,
            "source": None,
            "device": self.device,
            "dtype": None,
            "options": dict(self.options),
        }

    def calculator(self):
        if self._calculator is not None:
            return self._calculator

        # Construction is lazy but exact: validate the complete public mapping,
        # construct the requested calculator once, and cache only a successful
        # result. Quiet mode changes presentation only; it does not select a
        # fallback provider or retry with different settings.
        from calm.calculators.runtime import construct_calculator
        from calm.calculators.spec import CalculatorSpec

        spec_obj = CalculatorSpec.from_dict(self.to_spec())
        calc = construct_calculator(spec_obj, quiet=self.quiet)
        object.__setattr__(self, "_calculator", calc)
        return calc

    def check(
        self, structure: Any | None = None, *, elements: list[str] | None = None
    ) -> bool:
        if elements is None and structure is not None:
            atoms = (
                structure.to_ase()
                if hasattr(structure, "to_ase")
                else getattr(structure, "atoms", structure)
            )
            if hasattr(atoms, "get_chemical_symbols"):
                elements = sorted(set(atoms.get_chemical_symbols()))
        # Provider-specific elemental support is not yet centralized; for now
        # check only that the calculator can be constructed.
        self.calculator()
        return True

    def validate(
        self, structure: Any | None = None, *, elements: list[str] | None = None
    ) -> "Potential":
        """Validate the potential by attempting to construct the underlying calculator.

        This is a public, descriptive API for examples and user code. It raises
        on failure and returns self on success to support fluent usage.
        """
        self.check(structure, elements=elements)
        return self

    def supports(self, structure: Any) -> bool:
        """Return true after exact provider validation.

        CALM does not yet own a provider-independent element-support registry.
        Configuration, dependency, and construction failures therefore
        propagate instead of being collapsed into a false support result.
        """

        self.check(structure)
        return True

    def unsupported_elements(self, structure: Any) -> tuple[str, ...]:
        self.check(structure)
        return ()

    def fingerprint(self) -> str:
        if self._external_calculator:
            from calm.calculators.identity import get_calculator_uid

            uid = get_calculator_uid(self._calculator)
            return uid.removeprefix("calc:")

        from calm.calculators.spec import CalculatorSpec

        return CalculatorSpec.from_dict(self.to_spec()).fingerprint()

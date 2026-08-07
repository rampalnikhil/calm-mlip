"""Calculator provider implementations and helpers.

This module contains concrete provider implementations for external
calculator backends (MACE, CHGNet, GRACE, LAMMPS) as well as small helper functions
used when adapting user options to third-party constructors.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from .exceptions import (
    CalculatorBuildError,
    CalculatorSpecError,
    OptionalDependencyError,
)
from .registry import CalculatorProvider
from .spec import CalculatorSpec


def resolve_device(device: str | None) -> str:
    """Resolve a requested device string to a concrete backend device.

    ``None`` and ``"auto"`` are the only values interpreted here; every other
    value is passed through untouched so each backend keeps its own device
    validation. ``"auto"`` selects ``"cuda"`` when PyTorch reports a visible
    accelerator and falls back to ``"cpu"`` otherwise, which keeps a single
    script portable between CPU-only and GPU nodes.

    ROCm builds of PyTorch expose AMD GPUs through the same ``torch.cuda``
    namespace as CUDA builds, so this one check covers both vendors; the builds
    are told apart by ``torch.version.hip``, not by the device string.
    """

    if device is None:
        return "cpu"
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - driver probing is environment specific
        return "cpu"


class MACEProvider(CalculatorProvider):
    """Provider for MACE foundation models via `mace.calculators.mace_mp`."""

    DEFAULT_MODEL = "medium-mpa-0"

    VALID_MODELS = {
        "small",
        "medium",
        "large",
        "medium-mpa-0",
        "medium-omat-0",
        "matpes-pbe",
        "matpes-r2scan",
        "small-off",
        "medium-off",
        "large-off",
    }

    @property
    def family(self) -> str:
        return "mace"

    def describe(self) -> str:
        return "MACE foundation model calculators via mace.calculators.mace_mp"

    def required_extras(self) -> tuple[str, ...]:
        return ("mace",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("mace") is not None

    def list_models(self) -> tuple[str, ...]:
        return tuple(sorted(self.VALID_MODELS))

    def create(self, spec: CalculatorSpec) -> Any:
        if not self.is_available():
            raise OptionalDependencyError(
                "MACE is not installed. Install it (e.g. `pip install mace-torch`) to use MACE calculators."
            )

        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise OptionalDependencyError(
                "MACE is installed but CALM could not import "
                "mace.calculators.mace_mp. Install or upgrade `mace-torch`."
            ) from exc

        device = resolve_device(spec.device)
        options = spec.options_dict()

        # Guard against ambiguous double-specification.
        if "model" in options:
            raise CalculatorSpecError(
                "MACEProvider does not allow 'model' inside options; use spec.model."
            )
        if "device" in options:
            raise CalculatorSpecError(
                "MACEProvider does not allow 'device' inside options; use spec.device."
            )

        # CALM requests stress by default for MACE-MP calculations.
        options.setdefault("stress", True)

        try:
            return mace_mp(model=spec.model, device=device, **options)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct MACE calculator via mace_mp(model={spec.model!r}, device={device!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e


class CHGNetProvider(CalculatorProvider):
    """Provider for CHGNet pretrained models via ``CHGNetCalculator``.

    The provider keeps import-time behavior lightweight: importing CALM only
    registers this provider, while importing CHGNet and loading the requested
    model happens only inside :meth:`create`.
    """

    DEFAULT_MODEL = "0.3.0"

    VALID_MODELS = (
        "0.3.0",
        "0.2.0",
        "r2scan",
    )

    @property
    def family(self) -> str:
        return "chgnet"

    def describe(self) -> str:
        return "CHGNet pretrained models via chgnet.model.dynamics.CHGNetCalculator"

    def required_extras(self) -> tuple[str, ...]:
        return ("chgnet",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("chgnet") is not None

    def list_models(self) -> tuple[str, ...]:
        return self.VALID_MODELS

    def normalize_spec(self, spec: CalculatorSpec) -> CalculatorSpec:
        if spec.model == "default":
            raise CalculatorSpecError(
                f"CHGNet model 'default' is retired; use {self.DEFAULT_MODEL!r}."
            )
        if spec.model not in self.VALID_MODELS:
            raise CalculatorSpecError(
                f"Unsupported CHGNet model {spec.model!r}. "
                f"Supported models are: {', '.join(self.VALID_MODELS)}."
            )
        return spec

    def create(self, spec: CalculatorSpec) -> Any:
        spec = self.normalize_spec(spec)
        if not self.is_available():
            raise OptionalDependencyError(
                "CHGNet is not installed. Install it (e.g. `pip install chgnet`) to use CHGNet calculators."
            )

        try:
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import CHGNetCalculator
        except ImportError as exc:
            raise OptionalDependencyError(
                "CHGNet is installed but CALM could not import CHGNet/CHGNetCalculator. "
                "Install or upgrade the `chgnet` package."
            ) from exc

        options = spec.options_dict()
        for forbidden in ("model", "model_name", "device", "use_device"):
            if forbidden in options:
                raise CalculatorSpecError(
                    f"CHGNetProvider does not allow {forbidden!r} inside options; "
                    "use CalculatorSpec fields instead."
                )

        load_kwargs: dict[str, Any] = {}
        if spec.model:
            load_kwargs["model_name"] = spec.model

        try:
            model = CHGNet.load(**load_kwargs)
        except TypeError as exc:
            raise CalculatorBuildError(
                f"Failed to load CHGNet model {spec.model!r}. "
                "The installed CHGNet version may not support this model name."
            ) from exc

        calc_kwargs = dict(options)
        if spec.device:
            calc_kwargs["use_device"] = resolve_device(spec.device)

        try:
            return CHGNetCalculator(model=model, **calc_kwargs)
        except TypeError as exc:
            raise CalculatorBuildError(
                f"Failed to construct CHGNetCalculator for model={spec.model!r}. "
                f"Unsupported option keys: {sorted(options)}"
            ) from exc


class GRACEProvider(CalculatorProvider):
    """Provider for GRACE foundation models via `tensorpotential.calculator.grace_fm`."""

    DEFAULT_MODEL = "GRACE-1L-OMAT"

    VALID_MODELS = {
        "GRACE-1L-OMAT",
        "GRACE-2L-OMAT",
        "GRACE-1L-OAM",
        "GRACE-2L-OAM",
        "GRACE-1L-MP-r6",
        "GRACE-2L-MP-r5",
        "GRACE-2L-MP-r6",
    }

    @property
    def family(self) -> str:
        return "grace"

    def describe(self) -> str:
        return (
            "GRACE foundation model calculators via tensorpotential.calculator.grace_fm"
        )

    def required_extras(self) -> tuple[str, ...]:
        return ("tensorpotential",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("tensorpotential") is not None

    def list_models(self) -> tuple[str, ...]:
        return tuple(sorted(self.VALID_MODELS))

    def create(self, spec: CalculatorSpec) -> Any:
        if not self.is_available():
            raise OptionalDependencyError(
                "GRACE is not installed. Install the tensorpotential package to use GRACE calculators."
            )

        try:
            from tensorpotential.calculator import grace_fm
        except ImportError as exc:
            raise OptionalDependencyError(
                "GRACE is installed but CALM could not import "
                "tensorpotential.calculator.grace_fm. Install or upgrade "
                "`tensorpotential`."
            ) from exc

        options = spec.options_dict()
        if "model" in options:
            raise CalculatorSpecError(
                "GRACEProvider does not allow 'model' inside options; use spec.model."
            )

        # GRACE runs on TensorFlow, which places ops through its own
        # visible-device configuration rather than an explicit argument, so
        # grace_fm has no device parameter to forward to. Silently dropping an
        # explicit device request would run on CPU while reporting success, so
        # reject it. "auto" is advisory and is allowed to fall back to CPU.
        if spec.device and spec.device != "auto":
            if resolve_device(spec.device) != "cpu":
                raise CalculatorSpecError(
                    f"GRACEProvider cannot honor device={spec.device!r}. GRACE "
                    "uses TensorFlow, which selects devices through its own "
                    "visible-device configuration (for example the "
                    "CUDA_VISIBLE_DEVICES environment variable) rather than a "
                    "device argument. Use device='cpu' or device='auto' and "
                    "control GPU placement through the environment."
                )
        try:
            return grace_fm(spec.model, **options)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct GRACE calculator via grace_fm(model={spec.model!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e


def _as_str_list(values: Any) -> list[str]:
    """Normalize a list-ish input into a list[str]."""

    if values is None:
        return []
    if isinstance(values, (str, Path)):
        return [str(values)]
    if isinstance(values, (list, tuple, set)):
        return [str(v) for v in values]
    return [str(values)]


def _validate_kwargs_for_callable(
    callable_obj: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return ``kwargs`` after rejecting silently unsupported fields.

    ASE constructor signatures can change across versions. CALM must not adapt
    to those changes by dropping requested settings because that would create a
    calculator different from the persisted specification.
    """

    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    # If the callable accepts **kwargs, don't filter.
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return dict(kwargs)

    allowed = set(sig.parameters.keys())
    unsupported = sorted(set(kwargs) - allowed)
    if unsupported:
        raise CalculatorSpecError(
            f"{callable_obj!r} does not accept requested calculator option(s): "
            f"{unsupported}."
        )
    return dict(kwargs)


class LAMMPSProvider(CalculatorProvider):
    """LAMMPS calculators via ASE.

    This provider is intentionally low-level and forwards options directly into
    ASE's LAMMPS calculators.

    Conventions
    -----------
    * ``spec.model`` is treated as a LAMMPS ``pair_style`` (unless ``model`` is
      ``"custom"``).
    * Provide LAMMPS inputs in ``spec.options`` using either:

      - ``parameters``: a dict of LAMMPS key/value inputs (e.g. ``pair_style``,
        ``pair_coeff``), plus any ASE LAMMPS calculator kwargs.
      - Convenience top-level keys like ``pair_style``, ``pair_coeff``,
        ``parameters_file``, and ``files``.

    Backend selection
    -----------------
    By default, CALM uses :class:`ase.calculators.lammpsrun.LAMMPS` (subprocess
    backend). To use the in-process backend, set ``options={"backend": "lib"}``.
    """

    @property
    def family(self) -> str:
        return "lammps"

    def describe(self) -> str:
        return "LAMMPS calculators via ASE (lammpsrun/lammpslib)"

    def required_extras(self) -> tuple[str, ...]:
        # ASE is a core dependency of CALM.
        return ()

    def is_available(self) -> bool:
        return importlib.util.find_spec("ase") is not None and (
            importlib.util.find_spec("ase.calculators.lammpsrun") is not None
            or importlib.util.find_spec("ase.calculators.lammpslib") is not None
        )

    def list_models(self) -> tuple[str, ...]:
        # LAMMPS supports many pair styles; `model` is treated as pair_style.
        return ("custom",)

    def create(self, spec: CalculatorSpec) -> Any:
        if not self.is_available():
            raise OptionalDependencyError(
                "ASE LAMMPS calculator bindings are not available. Install/upgrade ASE."
            )

        options = spec.options_dict()
        backend = options.pop("backend", "run")
        if not isinstance(backend, str):
            raise CalculatorSpecError("LAMMPS option 'backend' must be a string.")

        # Backend-agnostic convenience inputs.
        parameters = options.pop("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise CalculatorSpecError("LAMMPS option 'parameters' must be a dict.")

        files = _as_str_list(options.pop("files", []))
        parameters_file = options.pop("parameters_file", None)
        if parameters_file is not None:
            files.extend(_as_str_list(parameters_file))

        # Promote common LAMMPS inputs from options into parameters.
        for key in (
            "pair_style",
            "pair_coeff",
            "units",
            "atom_style",
            "boundary",
            "kspace_style",
            "kspace_modify",
            "neighbor",
            "neigh_modify",
            "special_bonds",
        ):
            if key in options and key in parameters:
                raise CalculatorSpecError(
                    f"LAMMPS option {key!r} is specified both at the top level "
                    "and inside options['parameters']."
                )
            if key in options:
                parameters[key] = options.pop(key)

        # Treat spec.model as the sole pair_style owner unless the explicit
        # custom mode is selected.
        if spec.model and spec.model != "custom":
            if "pair_style" in parameters:
                raise CalculatorSpecError(
                    "LAMMPS pair_style must be specified either by spec.model "
                    "or in options['parameters'], not both. Use model='custom' "
                    "when parameters owns pair_style."
                )
            parameters["pair_style"] = spec.model

        # Normalize pair_coeff to the common ASE format (list[str]).
        if "pair_coeff" in parameters and isinstance(parameters["pair_coeff"], str):
            parameters["pair_coeff"] = [parameters["pair_coeff"]]

        if backend == "run":
            try:
                from ase.calculators.lammpsrun import LAMMPS as _LAMMPS
            except ImportError as exc:
                raise OptionalDependencyError(
                    "ASE does not expose ase.calculators.lammpsrun.LAMMPS. "
                    "Install/upgrade ASE or use backend='lib' if available."
                ) from exc

            init_kwargs: dict[str, Any] = {
                "parameters": parameters,
                "files": files,
                **options,
            }
            init_kwargs = _validate_kwargs_for_callable(_LAMMPS, init_kwargs)
            return _LAMMPS(**init_kwargs)

        if backend == "lib":
            try:
                from ase.calculators.lammpslib import LAMMPSlib as _LAMMPSlib
            except ImportError as exc:
                raise OptionalDependencyError(
                    "ASE does not expose ase.calculators.lammpslib.LAMMPSlib. "
                    "Install/upgrade ASE or use backend='run'."
                ) from exc

            lmpcmds = options.pop("lmpcmds", None)
            if lmpcmds is not None:
                lmpcmds = _as_str_list(lmpcmds)

            init_kwargs = {
                "lmpcmds": lmpcmds,
                "parameters": parameters,
                "files": files,
                **options,
            }
            # Drop None values to avoid noisy kwargs.
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
            init_kwargs = _validate_kwargs_for_callable(_LAMMPSlib, init_kwargs)
            return _LAMMPSlib(**init_kwargs)

        raise CalculatorSpecError(
            f"Unknown LAMMPS backend {backend!r}. Expected exactly 'run' or 'lib'."
        )


class SevenNetProvider(CalculatorProvider):
    """Provider for SevenNet foundation models via ``sevenn.calculator``."""

    DEFAULT_MODEL = "7net-0"

    @property
    def family(self) -> str:
        return "sevenn"

    def describe(self) -> str:
        return "SevenNet foundation models via sevenn.calculator.SevenNetCalculator"

    def required_extras(self) -> tuple[str, ...]:
        return ("sevenn",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("sevenn") is not None

    def normalize_spec(self, spec: CalculatorSpec) -> CalculatorSpec:
        model = (spec.model or "").strip()
        if not model or model == "default":
            return spec.replace(model=self.DEFAULT_MODEL)
        return spec

    def create(self, spec: CalculatorSpec) -> Any:
        spec = self.normalize_spec(spec)
        if not self.is_available():
            raise OptionalDependencyError(
                "SevenNet is not installed. Install it (e.g. `pip install sevenn`) to use SevenNet calculators."
            )
        from sevenn.calculator import SevenNetCalculator

        options = spec.options_dict()
        try:
            return SevenNetCalculator(model=spec.model, device=resolve_device(spec.device), **options)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct SevenNet calculator (model={spec.model!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e


class MatterSimProvider(CalculatorProvider):
    """Provider for MatterSim foundation models via ``mattersim.forcefield``."""

    DEFAULT_MODEL = "default"

    @property
    def family(self) -> str:
        return "mattersim"

    def describe(self) -> str:
        return "MatterSim foundation models via mattersim.forcefield.MatterSimCalculator"

    def required_extras(self) -> tuple[str, ...]:
        return ("mattersim",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("mattersim") is not None

    def create(self, spec: CalculatorSpec) -> Any:
        if not self.is_available():
            raise OptionalDependencyError(
                "MatterSim is not installed. Install it (e.g. `pip install mattersim`) to use MatterSim calculators."
            )
        from mattersim.forcefield import MatterSimCalculator

        options = spec.options_dict()
        model = (spec.model or "").strip()
        if model and model != "default":
            options.setdefault("load_path", model)
        try:
            return MatterSimCalculator(device=resolve_device(spec.device), **options)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct MatterSim calculator (model={spec.model!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e


class OrbProvider(CalculatorProvider):
    """Provider for Orb foundation models via ``orb_models``.

    ``spec.model`` names a pretrained factory in ``orb_models.forcefield.pretrained``
    (dashes are normalized to underscores, e.g. ``orb-v3-conservative-inf-omat``).
    ``torch.compile`` is disabled by default: it fails on CPU-only nodes and can
    be re-enabled per-spec with ``options={"compile": True}``.
    """

    DEFAULT_MODEL = "orb-v3-conservative-inf-omat"

    @property
    def family(self) -> str:
        return "orb"

    def describe(self) -> str:
        return "Orb foundation models via orb_models pretrained factories"

    def required_extras(self) -> tuple[str, ...]:
        return ("orb-models",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("orb_models") is not None

    def normalize_spec(self, spec: CalculatorSpec) -> CalculatorSpec:
        model = (spec.model or "").strip()
        if not model or model == "default":
            return spec.replace(model=self.DEFAULT_MODEL)
        return spec

    def create(self, spec: CalculatorSpec) -> Any:
        spec = self.normalize_spec(spec)
        if not self.is_available():
            raise OptionalDependencyError(
                "Orb is not installed. Install it (e.g. `pip install orb-models`) to use Orb calculators."
            )
        from orb_models.forcefield import pretrained

        # Requires orb-models >= 0.7 (see the `orb` extra): earlier releases
        # both place ORBCalculator elsewhere and return a bare model from the
        # pretrained factories instead of a (model, adapter) pair.
        from orb_models.forcefield.inference.calculator import ORBCalculator

        factory_name = spec.model.replace("-", "_")
        factory = getattr(pretrained, factory_name, None)
        if factory is None:
            raise CalculatorBuildError(
                f"Unknown Orb pretrained model {spec.model!r}: no factory "
                f"orb_models.forcefield.pretrained.{factory_name}"
            )
        options = spec.options_dict()
        device = resolve_device(spec.device)
        compile_flag = options.pop("compile", False)
        try:
            model, adapter = factory(device=device, compile=compile_flag)
            return ORBCalculator(model, adapter, device=device, **options)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct Orb calculator (model={spec.model!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e


class MatGLProvider(CalculatorProvider):
    """Provider for MatGL (M3GNet/CHGNet-arch) models via ``matgl.load_model``."""

    DEFAULT_MODEL = "M3GNet-PES-MatPES-PBE-2025.2"

    @property
    def family(self) -> str:
        return "matgl"

    def describe(self) -> str:
        return "MatGL pretrained PES models via matgl.load_model + PESCalculator"

    def required_extras(self) -> tuple[str, ...]:
        return ("matgl",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("matgl") is not None

    def normalize_spec(self, spec: CalculatorSpec) -> CalculatorSpec:
        model = (spec.model or "").strip()
        if not model or model == "default":
            return spec.replace(model=self.DEFAULT_MODEL)
        return spec

    def create(self, spec: CalculatorSpec) -> Any:
        spec = self.normalize_spec(spec)
        if not self.is_available():
            raise OptionalDependencyError(
                "MatGL is not installed. Install it (e.g. `pip install matgl`) to use MatGL calculators."
            )
        import matgl
        from matgl.ext.ase import PESCalculator

        options = spec.options_dict()
        try:
            return PESCalculator(matgl.load_model(spec.model), **options)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct MatGL calculator (model={spec.model!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e


class NequIPProvider(CalculatorProvider):
    """Provider for NequIP/Allegro models in two modes.

    1. **Foundation models**: ``spec.model`` is a nequip.net registry
       identifier (``nequip.net:mir-group/NequIP-OAM-S:0.1``) or a known
       shorthand (``NequIP-OAM-S``, ``Allegro-OAM-L``, ...). The provider
       invokes ``nequip-compile --mode aotinductor --target ase`` to
       download (Zenodo, CC-BY-4.0) and compile for the requested device,
       caching under ``$XDG_CACHE_HOME/calm-nequip`` (compile on the
       hardware class where inference runs). LAMMPS deployment uses
       separately compiled ``--target pair_allegro``/ML-IAP artifacts.
    2. **Bespoke models**: ``spec.model`` is a filesystem path to an
       existing compiled artifact — a portable TorchScript
       ``.nequip.pth`` or a device-specific AOTInductor ``.nequip.pt2``
       — e.g. a lab-trained universal Allegro model.

    Both modes end in ``NequIPCalculator.from_compiled_model``; the
    resolved artifact path participates in run identity.
    """

    FOUNDATION_MODELS = {
        "NequIP-OAM-S": "nequip.net:mir-group/NequIP-OAM-S:0.1",
        "NequIP-OAM-M": "nequip.net:mir-group/NequIP-OAM-M:0.1",
        "NequIP-OAM-L": "nequip.net:mir-group/NequIP-OAM-L:0.1",
        "NequIP-OAM-XL": "nequip.net:mir-group/NequIP-OAM-XL:0.1",
        "NequIP-MP-L": "nequip.net:mir-group/NequIP-MP-L:0.1",
        "Allegro-OAM-L": "nequip.net:mir-group/Allegro-OAM-L:0.1",
        "Allegro-MP-L": "nequip.net:mir-group/Allegro-MP-L:0.1",
    }

    @property
    def family(self) -> str:
        return "nequip"

    def describe(self) -> str:
        return "NequIP/Allegro compiled models via nequip.integrations.ase.NequIPCalculator"

    def required_extras(self) -> tuple[str, ...]:
        return ("nequip",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("nequip") is not None

    def create(self, spec: CalculatorSpec) -> Any:
        if not self.is_available():
            raise OptionalDependencyError(
                "NequIP is not installed. Install it (e.g. `pip install nequip`) to use NequIP/Allegro calculators."
            )
        model = (spec.model or "").strip()
        if not model or model == "default":
            raise CalculatorBuildError(
                "NequIPProvider requires spec.model: either a nequip.net "
                "foundation-model name (e.g. 'Allegro-OAM-L') or a path to a "
                "compiled artifact (.nequip.pth / .nequip.pt2). Known "
                f"foundation names: {sorted(self.FOUNDATION_MODELS)}"
            )

        registry_id = self.FOUNDATION_MODELS.get(model) or (
            model if model.startswith("nequip.net:") else None
        )
        if registry_id is not None:
            path = self._compiled_foundation_artifact(registry_id, resolve_device(spec.device))
        else:
            path = Path(model).expanduser()
            if not path.exists():
                raise CalculatorBuildError(
                    f"NequIP model {model!r} is neither a known foundation-model "
                    f"name ({sorted(self.FOUNDATION_MODELS)}), a nequip.net: "
                    "registry id, nor an existing compiled-artifact path."
                )

        from nequip.integrations.ase import NequIPCalculator

        options = spec.options_dict()
        try:
            return NequIPCalculator.from_compiled_model(
                compile_path=str(path),
                device=resolve_device(spec.device),
                **options,
            )
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct NequIP calculator from {path}. "
                f"Unsupported option keys: {sorted(options)}"
            ) from e


    def _compiled_foundation_artifact(self, registry_id: str, device: str) -> Path:
        """Return a cached compiled artifact for a nequip.net registry id.

        Compiles on first use via ``nequip-compile``. TorchScript mode is
        used for CPU (portable across nodes); AOTInductor for GPU devices
        (device-specific, matching nequip.net guidance to compile on the
        inference hardware).
        """
        import os
        import subprocess

        cache_root = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        ) / "calm-nequip"
        cache_root.mkdir(parents=True, exist_ok=True)

        safe = registry_id.replace("nequip.net:", "").replace("/", "__").replace(":", "__")
        # Two compile modes exist. AOTInductor produces the .nequip.pt2
        # artifact and is substantially faster at inference (measured ~11x on
        # Allegro and ~2.7x on NequIP versus TorchScript on CPU), so it is
        # tried first. TorchScript produces .nequip.pth and is the fallback:
        # inductor code generation segfaults on ROCm builds for these models,
        # and TorchScript compiles and runs there. Note the --target
        # distinction: this provider compiles for ASE, while LAMMPS deployment
        # (pair_allegro / ML-IAP) uses separately compiled --target artifacts.
        #
        # Compiled artifacts embed code tied to the torch build that produced
        # them, so the cache key carries the full torch version string. That
        # string includes the accelerator build tag ("2.7.0+rocm6.3" vs
        # "2.13.0+cu130"), which also keeps CUDA- and ROCm-compiled artifacts
        # from colliding on a shared filesystem.
        import torch

        build = torch.__version__.replace("+", "-").replace("/", "-")
        modes = (("aotinductor", "nequip.pt2"), ("torchscript", "nequip.pth"))
        candidates = [
            (mode, cache_root / f"{safe}__{device}__torch{build}.{suffix}")
            for mode, suffix in modes
        ]

        for _, cached in candidates:
            if cached.exists():
                return cached

        failures: list[str] = []
        for mode, artifact in candidates:
            cmd = [
                "nequip-compile",
                registry_id,
                str(artifact),
                "--mode",
                mode,
                "--device",
                device,
                "--target",
                "ase",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            except FileNotFoundError as e:
                raise OptionalDependencyError(
                    "nequip-compile CLI not found; is the nequip package installed in this environment?"
                ) from e
            except subprocess.TimeoutExpired:
                artifact.unlink(missing_ok=True)
                failures.append(f"[{mode}] timed out after 3600s")
                continue

            if proc.returncode == 0 and artifact.exists():
                return artifact

            # A failed compile can leave a partial archive behind -- notably a
            # segfault part-way through writing it. Kept, it would be returned
            # by the cache probe above on the next call and fail at load time
            # instead of recompiling, so discard it before trying the next mode.
            artifact.unlink(missing_ok=True)

            tail = "\n".join(
                (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            )
            failures.append(f"[{mode}] exit {proc.returncode}\n{tail}")

        raise CalculatorBuildError(
            f"nequip-compile failed for {registry_id!r} (device={device!r}) in "
            "every available mode.\n" + "\n\n".join(failures)
        )


class UMAProvider(CalculatorProvider):
    """Provider for Meta/FAIR-Chem UMA models via ``fairchem-core``.

    ``spec.model`` is a UMA checkpoint name (e.g. ``uma-s-1p1``,
    ``uma-m-1p1``). UMA is multi-task: the same checkpoint serves
    different scientific domains, so the task head must be chosen
    explicitly via ``spec.options["task"]`` (e.g. ``omat`` for inorganic
    materials, ``omol`` molecules, ``oc20`` catalysis, ``odac`` MOFs,
    ``omc`` molecular crystals). CALM does not default this scientific
    choice.

    UMA checkpoints are license-gated on Hugging Face: each user must
    accept the model license and authenticate (``huggingface-cli login``)
    before first download.
    """

    VALID_TASKS = ("omat", "omol", "oc20", "odac", "omc")

    @property
    def family(self) -> str:
        return "uma"

    def describe(self) -> str:
        return "Meta/FAIR-Chem UMA models via fairchem.core.FAIRChemCalculator"

    def required_extras(self) -> tuple[str, ...]:
        return ("fairchem-core",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("fairchem") is not None

    def create(self, spec: CalculatorSpec) -> Any:
        if not self.is_available():
            raise OptionalDependencyError(
                "fairchem-core is not installed. Install it (e.g. `pip install fairchem-core`) "
                "to use UMA calculators."
            )
        model = (spec.model or "").strip()
        if not model or model == "default":
            raise CalculatorBuildError(
                "UMAProvider requires spec.model to name a UMA checkpoint (e.g. 'uma-s-1p1')."
            )

        options = spec.options_dict()
        task = options.pop("task", None)
        if not task:
            raise CalculatorBuildError(
                "UMAProvider requires options['task'] — UMA is multi-task and CALM "
                f"does not default the scientific domain. Choose one of {self.VALID_TASKS} "
                "(use 'omat' for bulk inorganic materials workflows)."
            )
        if task not in self.VALID_TASKS:
            raise CalculatorBuildError(
                f"Unknown UMA task {task!r}; expected one of {self.VALID_TASKS}."
            )

        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        try:
            predictor = pretrained_mlip.get_predict_unit(model, device=resolve_device(spec.device))
            return FAIRChemCalculator(predictor, task_name=task, **options)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct UMA calculator (model={model!r}, task={task!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e
        except Exception as e:
            msg = str(e)
            if "gated" in msg.lower() or "401" in msg or "403" in msg or "authent" in msg.lower():
                raise CalculatorBuildError(
                    f"Could not download UMA checkpoint {model!r}: the weights are "
                    "license-gated on Hugging Face. Accept the license on the model page "
                    "and run `huggingface-cli login` first."
                ) from e
            raise


class PETMADProvider(CalculatorProvider):
    """Provider for PET-MAD universal potentials via ``pet-mad``.

    ``spec.model`` is either a released PET-MAD version tag (e.g.
    ``latest``, ``1.1``) or a filesystem path to a bespoke checkpoint
    (loaded via ``checkpoint_path``) — the same foundation/bespoke duality
    as the ``nequip`` family. Weights download from Hugging Face
    (ungated) on first use.
    """

    DEFAULT_MODEL = "latest"

    @property
    def family(self) -> str:
        return "petmad"

    def describe(self) -> str:
        return "PET-MAD universal potentials via pet_mad.calculator.PETMADCalculator"

    def required_extras(self) -> tuple[str, ...]:
        return ("pet-mad",)

    def is_available(self) -> bool:
        return importlib.util.find_spec("pet_mad") is not None

    def normalize_spec(self, spec: CalculatorSpec) -> CalculatorSpec:
        model = (spec.model or "").strip()
        if not model or model == "default":
            return spec.replace(model=self.DEFAULT_MODEL)
        return spec

    def create(self, spec: CalculatorSpec) -> Any:
        spec = self.normalize_spec(spec)
        if not self.is_available():
            raise OptionalDependencyError(
                "pet-mad is not installed. Install it (e.g. `pip install pet-mad`) "
                "to use PET-MAD calculators."
            )
        from pet_mad.calculator import PETMADCalculator

        options = spec.options_dict()
        model = spec.model
        kwargs: dict[str, Any] = {"device": resolve_device(spec.device), **options}
        path = Path(model).expanduser()
        if path.exists():
            kwargs["checkpoint_path"] = str(path)
        else:
            kwargs["version"] = model
        try:
            return PETMADCalculator(**kwargs)
        except TypeError as e:
            raise CalculatorBuildError(
                f"Failed to construct PET-MAD calculator (model={model!r}). "
                f"Unsupported option keys: {sorted(options)}"
            ) from e

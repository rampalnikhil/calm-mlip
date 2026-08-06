# Install CALM

<p class="calm-lede">
Install CALM from a tagged source checkout in a Python 3.10–3.12 environment, choose only the dependency layer required by your workflow, and validate that exact environment before beginning a study.
</p>

<div class="calm-page-facts" markdown>
- **Supported Python:** 3.10, 3.11, or 3.12
- **Current distribution:** source checkout
- **Recommended first environment:** baseline science
- **Calculator providers:** installed and validated separately
</div>

## Choose an environment

CALM separates installation into three layers.

| Layer | Use it for | Included capabilities |
| --- | --- | --- |
| **Base** | Create, reopen, inspect, query, and export project metadata | NumPy, SQLAlchemy, the project API, manifests, and lightweight utilities |
| **Science** | Run the geometry tutorials and normal interface-construction workflows | Base plus SciPy, ASE, and spglib |
| **Calculator provider** | Optimize, refine with energies, relax structures, or evaluate energies | Science capabilities plus one registered provider stack and any external runtime or model files it needs |

Most new users should begin with the **science** environment. It supports material and surface handling, coherent matching, candidate inspection, and interface construction without installing a machine-learned potential.

## Install from a source checkout

Obtain a tagged CALM source checkout and run the commands from its repository root.

### Recommended: baseline science environment

The maintained Conda recipe installs the compiled scientific stack from `conda-forge`:

```bash
conda env create -f environments/calm-science.yml
conda activate calm-science
python -m pip install -e . --no-deps
```

Validate the environment:

```bash
python environments/validate_environment.py --profile science
```

This is the supported environment for the two calculator-free tutorials and the ASE baseline provider.

### Alternative: install the science extra with pip

In an existing compatible Python environment:

```bash
python -m pip install -e ".[science]"
python environments/validate_environment.py --profile science
```

This route asks pip to resolve SciPy, ASE, and spglib. The maintained Conda science recipe is generally more predictable when compiled dependencies must be solved together.

### Import-light base installation

Use the base package only when you need project creation, reopening, queries, manifests, or lightweight inspection without atomistic geometry operations:

```bash
python -m pip install -e .
python environments/validate_environment.py --profile base
```

The base installation does not provide ASE, SciPy, spglib, plotting, notebooks, documentation tools, or calculator runtimes.

## Validate the environment

The environment validator checks:

- the supported Python range;
- required package and import availability;
- CALM version identity;
- whether `calm` is imported from the active checkout;
- provider registration and dependency visibility when requested; and
- a disposable project create/reopen smoke test.

It does **not** run model inference or certify that a calculator is scientifically suitable.

A successful science validation should be followed by a direct import check:

```bash
python -c "import calm, ase, scipy, spglib; print(calm.__version__)"
```

When multiple CALM checkouts or environments exist, inspect the imported path:

```bash
python -c "import calm; print(calm.__file__)"
```

The path should point into the checkout you intended to install.

## Run the first workflow

Keep generated tutorial projects under `examples/work/`:

```bash
python examples/tutorials/first_interface.py \
  --work-dir examples/work/first-interface \
  --reset
```

Then inspect the run summary:

```bash
python -m json.tool \
  examples/work/first-interface/outputs/run-summary.json
```

The first tutorial requires no calculator. It should produce one saved, constructed, and unrelaxed interface. See [Build your first interface](learn/first-interface.md) for the full interpretation.

## Add a calculator provider

Use one primary provider stack per environment. Provider registration, dependency availability, technical compatibility, and scientific suitability are separate questions.

### ASE baseline calculators

The `ase` family is available in the science environment. Validate that provider boundary explicitly:

```bash
python environments/validate_environment.py --provider ase
```

The calculator-backed tutorial uses ASE EMT with package-owned Cu and Ni structures:

```bash
python examples/tutorials/refine_relax_evaluate.py \
  --work-dir examples/work/refine-relax-evaluate \
  --reset
```

EMT demonstrates the CALM workflow and reference bookkeeping. It is not a general interface potential.

### Maintained ML-provider environments

CALM maintains separate source-checkout recipes for the three pip-managed ML provider families:

| Family | Environment recipe | Validation command |
| --- | --- | --- |
| `chgnet` | `environments/calm-chgnet.yml` | `python environments/validate_environment.py --provider chgnet` |
| `grace` | `environments/calm-grace.yml` | `python environments/validate_environment.py --provider grace` |
| `mace` | `environments/calm-mace.yml` | `python environments/validate_environment.py --provider mace` |
| `matgl` | `environments/calm-matgl.yml` | `python environments/validate_environment.py --provider matgl` |
| `mattersim` | `environments/calm-mattersim.yml` | `python environments/validate_environment.py --provider mattersim` |
| `nequip` | `environments/calm-nequip.yml` | `python environments/validate_environment.py --provider nequip` |
| `orb` | `environments/calm-orb.yml` | `python environments/validate_environment.py --provider orb` |
| `petmad` | `environments/calm-petmad.yml` | `python environments/validate_environment.py --provider petmad` |
| `sevenn` | `environments/calm-sevenn.yml` | `python environments/validate_environment.py --provider sevenn` |
| `uma` | `environments/calm-uma.yml` | `python environments/validate_environment.py --provider uma` |

For example:

```bash
conda env create -f environments/calm-grace.yml
conda activate calm-grace
python -m pip install -e . --no-deps
python environments/validate_environment.py --provider grace
```

The recipe installs a provider package; it does not select a scientifically appropriate model or prove accuracy for a material system.

### LAMMPS

The `lammps` provider begins with the science environment and additionally requires a compatible external LAMMPS executable or library plus the required potential files. Validate the CALM adapter boundary with:

```bash
python environments/validate_environment.py --provider lammps
```

That command cannot verify external executables, shared libraries, licensed files, parameter files, accelerator drivers, or scientific parameterization.

### Accelerator-specific stacks

CALM does not publish one universal GPU environment. CUDA, ROCm, Apple MPS, TensorFlow, PyTorch, and provider-specific accelerator requirements must follow the selected provider’s upstream instructions.

After installing that stack in a dedicated environment, install the checkout without resolving dependencies again:

```bash
python -m pip install -e . --no-deps
python environments/validate_environment.py --provider <family>
```

#### Selecting a device

Every provider that wraps a PyTorch backend accepts `device`:

```python
Potential.chgnet(device="cuda")   # explicit accelerator
Potential.chgnet(device="cpu")    # explicit CPU
Potential.chgnet(device="auto")   # accelerator when visible, else CPU
```

`"auto"` keeps one script portable between CPU-only and GPU nodes. ROCm builds
of PyTorch expose AMD GPUs through the same `torch.cuda` namespace as CUDA
builds, so `device="cuda"` and `device="auto"` are correct on both vendors; the
two builds are distinguished by `torch.version.hip`, not by the device string.

Two providers cannot honor a device argument. GRACE runs on TensorFlow, which
selects devices through its own visible-device configuration, so an explicit
non-CPU device is rejected rather than silently ignored. MatGL accepts the
argument but currently executes on CPU regardless.

#### Worked example: AMD ROCm

The default PyTorch wheel on PyPI is built for CUDA, so a plain
`pip install <backend>` on an AMD machine produces a stack where
`torch.cuda.is_available()` is `False`. Install PyTorch from the ROCm index
first, then constrain the backend resolve so it cannot pull the CUDA wheel back
in:

```bash
pip install torch==2.7.0+rocm6.3 --index-url https://download.pytorch.org/whl/rocm6.3
echo "torch==2.7.0+rocm6.3" > constraints.txt
pip install <backend> -c constraints.txt \
  --extra-index-url https://download.pytorch.org/whl/rocm6.3
python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"
```

Ahead-of-time compilation (the NequIP and Allegro foundation-model path) needs
the HIP toolchain headers and writes large compiler caches. Point `ROCM_PATH` at
an installed ROCm tree, and redirect the caches if the home directory is small:

```bash
export ROCM_PATH=/opt/rocm-6.3.1          # no /opt/rocm symlink on many sites
export TRITON_CACHE_DIR=$SCRATCH/triton   # Triton ignores XDG_CACHE_HOME
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/torchinductor
export MIOPEN_USER_DB_PATH=$SCRATCH/miopen
```

On PyTorch 2.7.0+rocm6.3, `nequip-compile --mode aotinductor --device cuda`
reaches code generation and then aborts with a segmentation fault, having
reported `aten._linalg_det.default is missing a c-shim implementation`. That is
a gap in the inductor ROCm backend rather than a CALM or NequIP defect.

The provider recovers from this: it compiles with AOTInductor first and falls
back to TorchScript when that fails, so NequIP and Allegro do run on GPU there.
AOTInductor is tried first because its compiled artifacts evaluate considerably
faster -- roughly 11x for Allegro and 2.7x for NequIP in CPU single-point
timings -- so the slower mode is used only when the faster one is unavailable.
Both artifacts are cached side by side, keyed by device and torch build.

Backends do not always share an environment. MACE requires `e3nn` 0.4.x while
SevenNet requires 0.6.x, so they need separate environments -- which is why the
recipes are per family rather than one combined stack.

## Installation boundaries

The current documented user distribution is a tagged source checkout. A package-registry command such as `pip install calm` is not yet a supported installation instruction.

Installation establishes only that the software and selected dependencies are available. It does not establish that:

- a model covers the required elements or configurations;
- stress is reliable enough for variable-cell work;
- surface, strained-cell, or short-contact environments are in domain;
- energy references are mutually compatible; or
- a result is scientifically converged or validated.

Use [Calculator support](reference/calculator-support.md) before beginning a calculator-backed study.

## Troubleshooting

### Python version is rejected

Create an environment using Python 3.10, 3.11, or 3.12. CALM currently declares `>=3.10,<3.13`.

### `ase` or `spglib` is missing

The base package is installed, but the science dependencies are not. Activate the science environment or install the `science` extra.

### `calm` imports from the wrong checkout

Activate the intended environment and reinstall from the correct repository root:

```bash
python -m pip install -e . --no-deps
python -c "import calm; print(calm.__file__)"
```

### A provider is registered but unavailable

Install the provider-specific environment or dependency extra, then run the corresponding `--provider` validation. Registration alone does not install the upstream package.

### Validation passes but a calculation fails

The remaining issue may be model configuration, element coverage, device selection, external files, structure compatibility, memory, or scientific suitability. Continue with [Calculation and data troubleshooting](reference/troubleshooting-calculations.md).

## Related documentation

- [Build your first interface](learn/first-interface.md)
- [Calculator support](reference/calculator-support.md)
- [Supported scientific scope](reference/supported-scope.md)
- [Materials](use/materials.md)
- [Relax and evaluate](use/relax-evaluate.md)

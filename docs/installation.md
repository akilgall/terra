# Installation

TERRA is published on PyPI as `terra-st` (the import name is `terra`). It runs on
NVIDIA GPUs (CUDA), Apple Silicon GPUs (MPS) and CPU. An NVIDIA GPU is
recommended for pretraining and large-scale embedding. Fine-tuning, count
decoding and inference also run on a Mac. Install in two steps — PyTorch first,
so it matches your hardware, then TERRA. We recommend
[uv](https://docs.astral.sh/uv/).

## 1. Install PyTorch for your hardware

**NVIDIA GPU (Linux / Windows).** Install the [PyTorch](https://pytorch.org)
build that matches your GPU driver **before** installing TERRA, so the correct
CUDA wheel is used (otherwise a plain install pulls the default wheel, which may
not match your driver). Run `nvidia-smi` and read the "CUDA Version" in the
top-right, then install the matching CUDA build — see the
[official PyTorch install guide](https://pytorch.org/get-started/locally/) — e.g.:

```shell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Apple Silicon (macOS).** The default PyTorch wheel includes MPS support:

```shell
uv pip install torch
```

**CPU only.** On Linux, install the CPU wheel
(`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`); on
macOS and Windows the default wheel works.

## 2. Install TERRA

```shell
uv pip install terra-st
```

Plain `pip install terra-st` works too. Verify the install (PyTorch sees your
accelerator, and TERRA imports):

```shell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
python -c "import terra; print(terra.__version__)"
```

On an NVIDIA machine the first boolean should be `True`; if it prints `False`,
PyTorch can't see your GPU (usually a CUDA build that doesn't match your driver
— revisit step 1). On Apple Silicon the second boolean should be `True`. If
both are `False`, TERRA runs on the CPU, which is fine for small datasets but
slow for large ones.

## Choosing a device

TERRA picks the device automatically, preferring CUDA, then MPS, then CPU. To
override it:

- **Embedding:** pass `device=` to `harmonize_tokenize_embed_pipeline`,
  `embed_dataset` or `gene_embed_dataset`, e.g. `device="cpu"` or `device="mps"`.
- **Fine-tuning:** set `device` in the `finetune` section of the config used by
  `terra.training.finetune` / `terra.training.finetune_self_supervised`:

  ```yaml
  finetune:
    device: mps   # auto (default), cpu, mps, cuda or cuda:N
  ```

- **Count decoder:** `python -m terra.training.decode ... --device mps`, or
  `apply_count_decoder(..., device="mps")`. Legacy GPU ids still work
  (`--device -1` selects the CPU).

Notes for Apple Silicon:

- `use_bfloat16` enables bfloat16 mixed precision on MPS, as on CUDA. On the CPU
  TERRA always runs in float32.
- macOS starts `DataLoader` workers with `spawn`, so each worker re-imports
  TERRA's dependencies. The shipped configs use `num_workers: 16`; on a Mac,
  0–4 workers is usually faster.
- If PyTorch lacks an MPS kernel for an operation it raises
  `NotImplementedError`. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` runs such
  operations on the CPU instead.
- Pretraining (`terra.training.main`) still requires NVIDIA GPUs (it uses DDP
  with NCCL).

## Development install

For a development install from a clone of the
[repository](https://github.com/Lotfollahi-lab/terra) (after step 1 above):

```shell
git clone https://github.com/Lotfollahi-lab/terra.git
cd terra
uv pip install -e ".[dev,test,doc]"
```

## Optional extras

TERRA ships several optional dependency groups:

| Extra | Install | Purpose |
| --- | --- | --- |
| `perturb` | `uv pip install "terra-st[perturb]"` | In-silico perturbation scoring (`infer_token_distance`, `summarize_*_by_label`). Installs [GeomLoss](https://www.kernel-operations.io/geomloss/) for the point-cloud (Sinkhorn/energy/MMD) distances. **Required by the perturbation and spatial-mapping tutorials.** |
| `hub` | `uv pip install "terra-st[hub]"` | Publish/download model bundles on the Hugging Face Hub (`terra-hub`). |
| `notebook` | `uv pip install "terra-st[notebook]"` | JupyterLab, ipykernel, and `gdown` (demo-data download) to run the tutorial notebooks. |
| `eval` | `uv pip install "terra-st[eval]"` | Evaluation utilities (CellPhoneDB, Omnipath). |
| `doc` | `uv pip install "terra-st[doc]"` | Build the documentation. |
| `test` | `uv pip install "terra-st[test]"` | Run the test suite. |

## Reproducible environment

For the exact, fully-pinned environment TERRA is developed and tested against, use
the committed lockfile with [uv](https://docs.astral.sh/uv/):

```shell
uv sync
```

## RAPIDS GPU acceleration (advanced)

Large downstream analyses (neighbors, Leiden, UMAP) can run on the GPU with
NVIDIA [RAPIDS](https://rapids.ai/) via
[`rapids-singlecell`](https://rapids-singlecell.readthedocs.io/). This is
**optional and environment-level**: RAPIDS is CUDA-specific and installed from
NVIDIA's package index, not as a `terra-st` extra, and TERRA never requires it
(`cuml`, `cudf`, and `rapids-singlecell` are always optional).

In a fresh environment, install the RAPIDS and PyTorch wheels — match the
`-cu12` suffix, the RAPIDS version, and the PyTorch CUDA index to your hardware
(see the [RAPIDS install selector](https://docs.rapids.ai/install/)):

```shell
uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --only-binary rapids-singlecell \
    torch rapids-singlecell \
    "cudf-cu12==25.10.*" "cuml-cu12==25.10.*" "cugraph-cu12==25.10.*" "cuvs-cu12==25.10.*"
```

**Pin `scikit-learn` into `cuml`'s supported range.** `cuml`'s `cuml.accel`
layer wraps scikit-learn internals, so a too-new scikit-learn makes `import
cuml` (and therefore `rapids-singlecell`) fail with `AttributeError: type object
'BaseEstimator' has no attribute '_get_default_requests'`. RAPIDS 25.10 works
with scikit-learn 1.6–1.7. Enforce this with a constraints file so it holds
while TERRA and the rest of the stack (`scanpy`, `squidpy`, …) install —
otherwise `scanpy` pulls scikit-learn forward again and re-breaks `cuml`:

```shell
echo "scikit-learn<1.8" > rapids-constraints.txt
uv pip install -c rapids-constraints.txt terra-st
```

For other RAPIDS versions, check which scikit-learn range they support. Then
verify RAPIDS and TERRA both import:

```shell
python -c "import rapids_singlecell; print('rapids-singlecell ok')"
python -c "import terra; print('terra', terra.__version__)"
```

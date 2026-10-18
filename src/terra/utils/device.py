"""Device selection and mixed-precision helpers.

Lets TERRA run on NVIDIA CUDA, Apple Silicon (MPS), or CPU.
"""

import contextlib
import functools
import logging

import torch

logger = logging.getLogger(__name__)


__all__ = [
    "get_device",
    "autocast",
    "should_pin_memory",
]

_SUPPORTED_DEVICE_TYPES = ("cpu", "cuda", "mps")
# Device types for which ``use_bfloat16`` enables autocast. On CPU autocast
# stays off, matching TERRA's historical fp32 CPU behaviour.
_AUTOCAST_DEVICE_TYPES = ("cuda", "mps")


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def _auto_device(local_rank: int | None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank or 0)
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_device(device: str | torch.device,
                  local_rank: int | None) -> torch.device:
    try:
        parsed = torch.device(device)
    except RuntimeError as e:
        raise ValueError(
            f"Unsupported device '{device}'. Use one of 'auto', 'cpu', "
            "'mps', 'cuda' or 'cuda:N'.") from e
    if parsed.type not in _SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"Unsupported device '{device}'. Use one of 'auto', 'cpu', "
            "'mps', 'cuda' or 'cuda:N'.")
    if parsed.type == "cuda" and parsed.index is None:
        parsed = torch.device("cuda", local_rank or 0)
    return parsed


def _check_available(device: torch.device) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{device}' but CUDA is not available. "
                "Use device='auto' to pick the best available device.")
        if device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested device '{device}' but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible.")
    elif device.type == "mps" and not _mps_available():
        raise ValueError(
            f"Requested device '{device}' but MPS is not available (it "
            "needs Apple Silicon and a PyTorch build with MPS support). Use "
            "device='auto' to pick the best available device.")


def get_device(device: str | torch.device | None = None,
               local_rank: int | None = None) -> torch.device:
    """
    Resolve the torch device to run on.

    Parameters
    ----------
    device:
        ``None`` or ``'auto'`` picks the best available device in the order
        CUDA > MPS > CPU. Otherwise one of ``'cpu'``, ``'mps'``, ``'cuda'``,
        ``'cuda:N'`` (or an equivalent ``torch.device``).
    local_rank:
        CUDA device index to use when ``device`` is ``None``, ``'auto'`` or
        ``'cuda'`` without an index (e.g. the ``LOCAL_RANK`` of a
        distributed run). Defaults to 0.

    Returns
    -------
    device:
        The resolved device. For CUDA it is also made the current device.

    Raises
    ------
    ValueError
        If ``device`` names an unsupported or unavailable backend.
    """
    if device is None or device == "auto":
        resolved = _auto_device(local_rank)
    else:
        resolved = _parse_device(device, local_rank)
        _check_available(resolved)

    if resolved.type == "cuda":
        torch.cuda.set_device(resolved)
    logger.info(f"Using device: {resolved}.")
    return resolved


@functools.cache
def _autocast_supported(device_type: str, dtype: torch.dtype) -> bool:
    try:
        torch.autocast(device_type=device_type, dtype=dtype)
    except RuntimeError as e:
        logger.warning(
            f"{dtype} autocast is not supported on '{device_type}' with "
            f"torch {torch.__version__} ({e}); running in float32 instead.")
        return False
    return True


def autocast(device: torch.device,
             enabled: bool,
             dtype: torch.dtype = torch.bfloat16
             ) -> contextlib.AbstractContextManager:
    """
    Mixed-precision context for ``device``.

    Parameters
    ----------
    device:
        Device the model runs on.
    enabled:
        Whether mixed precision is requested (e.g. ``use_bfloat16``).
    dtype:
        Lower-precision dtype to autocast to.

    Returns
    -------
    context:
        ``torch.autocast`` on CUDA and MPS. A no-op context on CPU, and on
        MPS when the installed torch rejects MPS autocast (logged once). On
        CUDA, errors (e.g. bfloat16 on a GPU without bf16 support) propagate
        as before.
    """
    if device.type not in _AUTOCAST_DEVICE_TYPES:
        return contextlib.nullcontext()
    if (enabled and device.type == "mps"
            and not _autocast_supported(device.type, dtype)):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype,
                          enabled=enabled)


def should_pin_memory(device: torch.device, requested: bool) -> bool:
    """
    Whether a ``DataLoader`` should pin host memory for ``device``.

    Pinned memory only speeds up host-to-CUDA copies. MPS shares memory with
    the CPU, and recent torch releases error or warn when pinning there.
    """
    return requested and device.type == "cuda"

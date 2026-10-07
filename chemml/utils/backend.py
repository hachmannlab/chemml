"""
Helpers to detect available deep learning backends (TensorFlow / PyTorch) and
resolve the ``engine`` requested by chemml's neural network models, falling
back to PyTorch when TensorFlow cannot be imported (e.g. no TensorFlow build
available yet for the running Python version, such as 3.14+).
"""

import warnings

_TENSORFLOW_AVAILABLE = None


def is_tensorflow_available():
    """Check whether TensorFlow can be imported in the current environment.

    The result is cached after the first call since the availability of a
    package cannot change during the lifetime of the process.

    Returns
    -------
    bool
        True if ``import tensorflow`` succeeds, False otherwise.
    """
    global _TENSORFLOW_AVAILABLE
    if _TENSORFLOW_AVAILABLE is None:
        try:
            import tensorflow  # noqa: F401
            _TENSORFLOW_AVAILABLE = True
        except Exception:
            _TENSORFLOW_AVAILABLE = False
    return _TENSORFLOW_AVAILABLE


def resolve_device(device=None):
    """Resolve the PyTorch device string.

    Works for both CUDA and ROCm builds of PyTorch (both use the 'cuda' device type).

    Parameters
    ----------
    device: str or None
        None or 'cpu' -> 'cpu'; 'auto' -> 'cuda' if available else 'cpu';
        'cuda' or 'cuda:N' -> that device, falling back to 'cpu' with a warning if unavailable.

    Returns
    -------
    str
    """
    if device is None:
        return 'cpu'
    device = str(device).lower()
    if device == 'cpu':
        return 'cpu'

    import torch
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if not device.startswith('cuda'):
        raise ValueError("device has to be None, 'auto', 'cpu', 'cuda' or 'cuda:N'")
    if not torch.cuda.is_available():
        warnings.warn(
            "A GPU (CUDA/ROCm) was requested but is not available. Falling back to 'cpu'.",
            UserWarning,
            stacklevel=3,
        )
        return 'cpu'
    return device


def resolve_engine(engine, default='tensorflow'):
    """Validate the requested ``engine`` and fall back to PyTorch if needed.

    If ``engine`` is 'tensorflow' but TensorFlow is not installed/importable,
    a warning is raised and 'pytorch' is used instead.

    Parameters
    ----------
    engine: str or None
        Requested engine, either 'tensorflow' or 'pytorch'. If None, ``default`` is used.

    default: str, optional (default='tensorflow')
        Engine to use when ``engine`` is None.

    Returns
    -------
    str
        The resolved engine, either 'tensorflow' or 'pytorch'.
    """
    if engine is None:
        engine = default

    if engine not in ('tensorflow', 'pytorch'):
        raise ValueError("engine has to be 'tensorflow' or 'pytorch'")

    if engine == 'pytorch':
        return engine
    
    elif engine == 'tensorflow' and not is_tensorflow_available():
        warnings.warn(
            "TensorFlow is not installed or could not be imported in this "
            "environment (it may not yet support your Python version). "
            "Falling back to the 'pytorch' engine instead.",
            UserWarning,
            stacklevel=3,
        )
        engine = 'pytorch'

    return engine

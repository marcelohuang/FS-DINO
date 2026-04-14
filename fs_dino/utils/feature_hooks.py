"""Feature extraction hooks for DINOv2 intermediate layers.

Provides both a passive ``FeatureExtractHook`` (captures output without
stopping the forward pass) and an ``EarlyExitHook`` that raises a
sentinel exception to stop computation after the target block — saving
~50% of forward-pass compute for frozen DINOv2.
"""

import torch
import torch.nn as nn


class EarlyExitException(Exception):
    """Sentinel raised by EarlyExitHook to stop forward execution."""

    def __init__(self, feature: torch.Tensor):
        self.feature = feature


class FeatureExtractHook:
    """Context manager that captures a module's output.

    Usage::

        with FeatureExtractHook(model.blocks[2]) as hook:
            _ = model(x)
        feat = hook.feature  # output of blocks[2]
    """

    def __init__(self, module: nn.Module):
        self._module = module
        self._handle = None
        self.feature: torch.Tensor | None = None

    def _hook_fn(self, module, input, output):
        self.feature = output

    def __enter__(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()


class EarlyExitHook:
    """Context manager that captures output AND stops forward execution.

    Raises ``EarlyExitException`` after the hook fires, so all blocks
    after the target are skipped. The caller must catch the exception.

    Usage::

        try:
            with EarlyExitHook(model.blocks[2]) as hook:
                model.forward_features(x)
        except EarlyExitException as e:
            raw_feat = e.feature  # (B, N+1, C) with CLS at index 0
    """

    def __init__(self, module: nn.Module):
        self._module = module
        self._handle = None
        self.feature: torch.Tensor | None = None

    def _hook_fn(self, module, input, output):
        self.feature = output
        raise EarlyExitException(output)

    def __enter__(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, *args):
        if self._handle is not None:
            self._handle.remove()
        # Suppress EarlyExitException so it doesn't propagate
        if exc_type is EarlyExitException:
            return True
        return False

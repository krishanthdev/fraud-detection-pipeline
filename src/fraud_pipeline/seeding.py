"""One place to make a run reproducible.

Call :func:`set_global_seed` at the top of every stage. It seeds the standard library,
numpy and, if it is installed, torch. Torch is imported lazily so that the light stages
(ingest, validate) do not pay the cost of loading a deep learning framework.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> int:
    """Seed every random source the pipeline uses and return the seed.

    Args:
        seed: the integer seed, normally ``config.project.seed``.
        deterministic_torch: ask cuDNN for deterministic kernels. This is slower but it
            means two runs on the same machine produce the same numbers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("torch is not installed, skipping its seed")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.debug("global seed set to %s", seed)
    return seed

"""Logging setup shared by every stage and by the serving layer.

The pipeline prints to the console through rich, so a long training run stays readable,
and it writes the same lines to ``reports/pipeline.log`` so a run can be reviewed later.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

from fraud_pipeline.paths import ensure_dir

_CONFIGURED = False
_LOG_FILE = "pipeline.log"


def setup_logging(level: str = "INFO", log_dir: str | Path = "reports") -> logging.Logger:
    """Configure the root logger once and return the pipeline logger.

    Calling this more than once is safe. The second call is ignored, which matters because
    the FastAPI app and the command line can both start the same process.
    """
    global _CONFIGURED
    root = logging.getLogger()

    if _CONFIGURED:
        root.setLevel(level.upper())
        return logging.getLogger("fraud_pipeline")

    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console)

    try:
        target = ensure_dir(log_dir) / _LOG_FILE
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
    except OSError:
        # A read only file system (some container setups) should not stop the run.
        root.warning("could not open the log file, logging to the console only")

    # These libraries are chatty at INFO and drown out the pipeline's own output.
    for noisy in ("matplotlib", "PIL", "urllib3", "git", "mlflow.utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger("fraud_pipeline")


def stage_banner(logger: logging.Logger, stage: str, detail: str = "") -> None:
    """Print a clear separator so each stage is easy to find in a long log."""
    line = f"STAGE {stage.upper()}"
    if detail:
        line = f"{line}  |  {detail}"
    logger.info("=" * 70)
    logger.info(line)
    logger.info("=" * 70)

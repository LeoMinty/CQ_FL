"""CQ-FL experiment-one implementation.

The legacy demo files in the repository are intentionally kept untouched.  This
package contains the reproducible implementation used by ``run_experiment1.py``.
"""

from .config import DATASET_CONFIGS, METHOD_NAMES, ExperimentConfig

__all__ = ["DATASET_CONFIGS", "METHOD_NAMES", "ExperimentConfig"]

"""harness package root."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import random
import warnings

import numpy as np
import torch
from sklearn.utils import check_random_state

__version__ = importlib_metadata.version('harness')


# Ignore warnings from pydantic. The specific warning is
# "Expected `Union[$ALL_TASK_CONFIGS]` but got `CONFIG_INSTANCE`
# - serialized value may not be as expected" This is core from the
# use of inheritance in the task configuration classes. I am silencing
# this warning as it does not affect the functionality of the configs.
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')


# Globally set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
SKLEARN_RANDOM_STATE = check_random_state(SEED)

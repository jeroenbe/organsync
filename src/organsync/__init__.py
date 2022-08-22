import multiprocessing
import os

from . import models  # noqa: F401
from . import policies  # noqa: F401
from . import simulation  # noqa: F401
from . import survival_analysis  # noqa: F401
from . import version  # noqa: F401

cores_env = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = cores_env
os.environ["OPENBLAS_NUM_THREADS"] = cores_env
os.environ["MKL_NUM_THREADS"] = cores_env
os.environ["VECLIB_MAXIMUM_THREADS"] = cores_env
os.environ["NUMEXPR_NUM_THREADS"] = cores_env

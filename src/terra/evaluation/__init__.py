import importlib

from . import alignment
from .knn import knn_classifier
from .linear import linear_classifier, linear_regressor
from .utils import compute_neighborhood_composition, plot_roc_curve

# Ligand-receptor helpers need the optional `eval` extra (cellphonedb,
# omnipath); import them on first use so the rest of the package works
# without it.
_LAZY = {"get_adata_cellphonedb_lr_pairs": "lr_interactions",
         "get_adata_omnipath_lr_pairs": "lr_interactions"}


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Dataset registry and loading interfaces."""

from creditrep.datasets.loader import load_dataset
from creditrep.datasets.models import LoadedDataset
from creditrep.datasets.registry import load_registry

__all__ = ["LoadedDataset", "load_dataset", "load_registry"]

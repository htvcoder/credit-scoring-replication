"""Python foundation for the credit scoring replication project."""

from creditrep.datasets.loader import load_dataset
from creditrep.datasets.models import LoadedDataset

__all__ = ["LoadedDataset", "load_dataset"]

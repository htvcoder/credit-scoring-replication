"""Dataset loading exceptions with fail-fast messages."""


class DatasetError(ValueError):
    """Base error for dataset registry and loading failures."""


class RegistryError(DatasetError):
    """Raised when the dataset registry is missing or malformed."""


class DatasetNotFoundError(DatasetError):
    """Raised when a requested dataset ID is not present in the registry."""


class DatasetFileError(DatasetError):
    """Raised when a configured dataset file path cannot be used."""


class DatasetSchemaError(DatasetError):
    """Raised when dataset columns or target values violate the registry."""

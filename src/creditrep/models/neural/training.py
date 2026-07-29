"""Small CPU-first trainer for P6A tests, without CV or preprocessing ownership."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from creditrep.models.neural.config import MLPConfig
from creditrep.models.neural.exceptions import NeuralInputError, NeuralTrainingError
from creditrep.models.neural.network import ConfigurableMLP, architecture_metadata
from creditrep.models.neural.runtime import RuntimeMetadata, configure_runtime, require_torch


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float
    duration_seconds: float
    improved: bool
    finite: bool


@dataclass(frozen=True)
class TrainingHistory:
    epochs: tuple[EpochRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"epochs": [asdict(record) for record in self.epochs]}


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float
    best_value: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0
    triggered: bool = False
    stop_reason: str | None = None

    def update(self, value: float, epoch: int) -> bool:
        if not math.isfinite(value):
            self.triggered = True
            self.stop_reason = "non_finite_validation_loss"
            return False
        improved = self.best_value is None or value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.triggered = True
                self.stop_reason = "early_stopping_patience_exhausted"
        return improved

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingSummary:
    runtime: RuntimeMetadata
    epochs_requested: int
    epochs_completed: int
    best_epoch: int | None
    best_validation_loss: float | None
    stop_reason: str
    early_stopping_triggered: bool
    best_weights_restored: bool
    parameter_count: int
    total_training_duration_seconds: float
    model_architecture: dict[str, Any]
    optimizer: dict[str, Any]
    loss_function: str = "BCEWithLogitsLoss"
    checkpoint_saved: bool = False
    warnings: tuple[str, ...] = ()
    publishable: bool = False
    result_scope: str = "mlp_training_validation"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime"] = self.runtime.to_dict()
        return payload


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    history: TrainingHistory
    summary: TrainingSummary

    def predict_probabilities(self, X: Any) -> np.ndarray:
        values = _validate_features(X, expected_features=self.summary.model_architecture["input_dimension"], context="prediction")
        torch = require_torch()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.as_tensor(values, dtype=torch.float32, device=self.summary.runtime.resolved_device))
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise NeuralTrainingError("Probability prediction is non-finite or outside [0, 1].")
        return probabilities.astype(float)


def _validate_features(values: Any, *, expected_features: int | None, context: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise NeuralInputError(f"{context} X must be a two-dimensional matrix.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise NeuralInputError(f"{context} X must contain at least one row and one feature.")
    if expected_features is not None and array.shape[1] != expected_features:
        raise NeuralInputError(f"{context} feature dimension mismatch: {array.shape[1]} != {expected_features}.")
    if not np.isfinite(array).all():
        raise NeuralInputError(f"{context} X contains NaN or Infinity.")
    return array


def _validate_targets(values: Any, *, rows: int, context: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim not in {1, 2} or (array.ndim == 2 and array.shape[1] != 1):
        raise NeuralInputError(f"{context} y must be a one-dimensional binary vector.")
    array = array.reshape(-1)
    if len(array) != rows:
        raise NeuralInputError(f"{context} X and y row counts differ: {rows} != {len(array)}.")
    if not np.isfinite(array.astype(float)).all():
        raise NeuralInputError(f"{context} y contains NaN or Infinity.")
    if not np.isin(array, [0, 1]).all():
        raise NeuralInputError(f"{context} y must contain only 0 and 1.")
    return array.astype(np.float32)


def train_mlp(X_train: Any, y_train: Any, config: MLPConfig, *, X_validation: Any | None = None, y_validation: Any | None = None) -> TrainingResult:
    """Train an in-memory MLP. Inputs must already be train-only preprocessed/scaled."""
    if config.early_stopping.enabled and (X_validation is None or y_validation is None):
        raise NeuralInputError("Early stopping requires validation data; outer test folds must not be used for this purpose.")
    train_X = _validate_features(X_train, expected_features=config.input_dimension, context="training")
    train_y = _validate_targets(y_train, rows=len(train_X), context="training")
    validation_X = validation_y = None
    if X_validation is not None or y_validation is not None:
        if X_validation is None or y_validation is None:
            raise NeuralInputError("Both X_validation and y_validation must be supplied together.")
        validation_X = _validate_features(X_validation, expected_features=train_X.shape[1], context="validation")
        validation_y = _validate_targets(y_validation, rows=len(validation_X), context="validation")
    runtime = configure_runtime(seed=config.random_seed, device_policy=config.device_policy, deterministic=config.deterministic)
    torch = require_torch()
    model = ConfigurableMLP(train_X.shape[1], config).to(runtime.resolved_device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer_type = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.as_tensor(train_X), torch.as_tensor(train_y)), batch_size=config.batch_size, shuffle=True, num_workers=config.dataloader_workers)
    stopping = EarlyStoppingState(config.early_stopping.patience, float(config.early_stopping.min_delta))
    best_weights = None
    records: list[EpochRecord] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X.to(runtime.resolved_device)).reshape(-1)
            loss = criterion(logits, batch_y.to(runtime.resolved_device))
            if not torch.isfinite(loss):
                raise NeuralTrainingError("Training loss is NaN or Infinity.")
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss: float | None = None
        improved = False
        finite = all(math.isfinite(loss) for loss in losses)
        if validation_X is not None and validation_y is not None:
            model.eval()
            with torch.no_grad():
                loss = criterion(model(torch.as_tensor(validation_X, device=runtime.resolved_device)).reshape(-1), torch.as_tensor(validation_y, device=runtime.resolved_device))
                validation_loss = float(loss.detach().cpu())
            improved = stopping.update(validation_loss, epoch)
            finite = finite and math.isfinite(validation_loss)
            if improved:
                best_weights = copy.deepcopy(model.state_dict())
        records.append(EpochRecord(epoch, float(np.mean(losses)), validation_loss, float(optimizer.param_groups[0]["lr"]), time.perf_counter() - epoch_started, improved, finite))
        if stopping.triggered:
            break
    restored = False
    if best_weights is not None:
        try:
            model.load_state_dict(best_weights)
            restored = True
        except RuntimeError as exc:
            raise NeuralTrainingError("Failed to restore best model weights.") from exc
    stop_reason = stopping.stop_reason or "max_epochs_reached"
    architecture = architecture_metadata(model, input_dimension=train_X.shape[1], config=config)
    summary = TrainingSummary(runtime, config.max_epochs, len(records), stopping.best_epoch, stopping.best_value, stop_reason, stopping.triggered, restored, architecture["trainable_parameter_count"], time.perf_counter() - started, architecture, {"name": config.optimizer, "learning_rate": config.learning_rate, "weight_decay": config.weight_decay})
    return TrainingResult(model, TrainingHistory(tuple(records)), summary)

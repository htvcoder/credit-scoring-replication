"""Protocol A preprocessing pipeline composition for P3B."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditrep.preprocessing.exceptions import PreprocessingError
from creditrep.preprocessing.protocol import LeakageSafePreprocessor, ProtocolAConfig
from creditrep.preprocessing.scaling import TrainOnlyStandardScaler
from creditrep.preprocessing.vif import IterativeVIFSelector
from creditrep.preprocessing.woe import WeightOfEvidenceEncoder

PIPELINE_VERSION = "0.1.0"


class ProtocolAPreprocessingPipeline:
    """Compose P3A imputation, categorical WOE, VIF and optional scaling."""

    def __init__(
        self,
        *,
        dataset_metadata: dict[str, Any] | None = None,
        numeric_features: list[str] | tuple[str, ...] | None = None,
        categorical_features: list[str] | tuple[str, ...] | None = None,
        config: ProtocolAConfig | None = None,
    ) -> None:
        self.config = config or ProtocolAConfig()
        self.imputer = LeakageSafePreprocessor(
            dataset_metadata=dataset_metadata,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=self.config,
        )
        self.woe: WeightOfEvidenceEncoder | None = None
        self.vif: IterativeVIFSelector | None = None
        self.scaler: TrainOnlyStandardScaler | None = None
        self._fitted = False
        self._input_feature_order: list[str] = []
        self._final_output_feature_order: list[str] = []
        self._fitted_row_count = 0

    def fit(self, X_train: pd.DataFrame, y_train: Any) -> "ProtocolAPreprocessingPipeline":
        """Fit every enabled preprocessing step on training data only."""

        if y_train is None:
            raise PreprocessingError("Protocol A pipeline fit() requires y_train for WOE.")
        self._input_feature_order = [str(column) for column in X_train.columns]
        imputed = self.imputer.fit_transform(X_train, y_train=y_train)
        imputer_metadata = self.imputer.get_metadata()
        numeric = list(imputer_metadata["numeric_features"])
        categorical = list(imputer_metadata["categorical_features"])

        current = imputed
        if self.config.woe_enabled:
            if self.config.woe_scope != "categorical":
                raise PreprocessingError(f"Unsupported WOE scope {self.config.woe_scope!r}.")
            self.woe = WeightOfEvidenceEncoder(
                categorical_features=categorical,
                passthrough_features=numeric,
                smoothing=self.config.woe_smoothing,
                unknown_value=self.config.woe_unknown_value,
                sign_convention=self.config.woe_sign_convention,
            )
            current = self.woe.fit_transform(current, y_train)
        else:
            self.woe = None
            if categorical:
                raise PreprocessingError("Protocol A pipeline requires WOE enabled when categorical features exist.")

        if not np.isfinite(current.to_numpy(dtype=float)).all():
            raise PreprocessingError("Protocol A pipeline matrix before VIF contains non-finite values.")

        if self.config.vif_enabled:
            self.vif = IterativeVIFSelector(
                threshold=self.config.vif_threshold,
                minimum_features_to_keep=self.config.vif_minimum_features_to_keep,
                tie_break=self.config.vif_tie_break,
            )
            current = self.vif.fit_transform(current)
        else:
            self.vif = None

        self.scaler = TrainOnlyStandardScaler(
            enabled=self.config.scaling_enabled,
            strategy=self.config.scaling_strategy,
        )
        current = self.scaler.fit_transform(current)

        self._final_output_feature_order = [str(column) for column in current.columns]
        self._fitted_row_count = int(len(X_train))
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted pipeline state without side effects."""

        transformed, _ = self.transform_with_diagnostics(X)
        return transformed

    def transform_with_diagnostics(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Transform and return per-call diagnostics outside reproducibility metadata."""

        if not self._fitted:
            raise PreprocessingError("ProtocolAPreprocessingPipeline.transform() called before fit().")
        current, imputation_diagnostics = self.imputer.transform_with_diagnostics(X)
        diagnostics: dict[str, Any] = {"imputation": imputation_diagnostics}

        if self.woe is not None:
            current, woe_diagnostics = self.woe.transform_with_diagnostics(current)
            diagnostics["woe"] = woe_diagnostics

        if self.vif is not None:
            current = self.vif.transform(current)

        if self.scaler is None:
            raise PreprocessingError("Protocol A pipeline scaler state is missing.")
        current = self.scaler.transform(current)
        if not np.isfinite(current.to_numpy(dtype=float)).all():
            raise PreprocessingError("Protocol A pipeline output contains NaN or infinite values.")
        return current.loc[:, self._final_output_feature_order], diagnostics

    def fit_transform(self, X_train: pd.DataFrame, y_train: Any) -> pd.DataFrame:
        """Fit on training data and return transformed training matrix."""

        return self.fit(X_train, y_train).transform(X_train)

    def get_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable fitted pipeline state."""

        return {
            "pipeline": "ProtocolAPreprocessingPipeline",
            "version": PIPELINE_VERSION,
            "fitted": self._fitted,
            "protocol_name": self.config.protocol_name,
            "protocol_version": self.config.protocol_version,
            "input_feature_order": list(self._input_feature_order),
            "imputation": self.imputer.get_metadata(),
            "woe_enabled": self.config.woe_enabled,
            "woe": self.woe.get_metadata() if self.woe is not None else None,
            "vif_enabled": self.config.vif_enabled,
            "vif": self.vif.get_metadata() if self.vif is not None else None,
            "scaling_enabled": self.config.scaling_enabled,
            "scaling": self.scaler.get_metadata() if self.scaler is not None else None,
            "final_output_feature_order": list(self._final_output_feature_order),
            "fitted_row_count": self._fitted_row_count,
        }

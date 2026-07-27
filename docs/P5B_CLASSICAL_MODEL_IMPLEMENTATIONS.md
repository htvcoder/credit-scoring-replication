# P5B classical model implementations

P5B implements the stable IDs `logistic_regression`, `decision_tree`, `random_forest`, and `xgboost` through the P5A registry/factory. All use explicit `random_state`, expose `predict_proba`, and preserve `y_score = P(class 1)` through class-aware mapping.

Reduced defaults are deterministic, small, CPU-safe and non-publishable (`result_scope: model_validation`); they are not paper-reference tuning grids. `paper_reference` remains a placeholder pending primary evidence. Factory validation rejects unknown parameters, invalid ranges, `n_jobs=0`, and seed conflicts. Class balancing is never enabled by default.

Decision Tree uses `sklearn.tree.DecisionTreeClassifier` with algorithm `cart`. Its metadata records `implementation`, `replication_role: approximation`, and `deviation_from_paper: c45_to_cart`; CART is not described as C4.5. Random Forest uses `RandomForestClassifier`; XGBoost retains its smoke-compatible CPU `hist` defaults.

Metadata records configured/effective parameters, library/version, seed, classes, timing, warnings, convergence status, profile and validation scope, and is JSON-safe. P2C smoke legacy fields remain available. P5C is deferred: this checkpoint does not connect real estimators to nested CV or produce scientific results.

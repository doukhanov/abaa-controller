from __future__ import annotations

import argparse
import __main__
import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class ConservativeRegressor(BaseEstimator, RegressorMixin):
    """Compatibility class for historical ABAA joblib files."""

    def __init__(self, base_estimator, scale_factor=1.1):
        self.base_estimator = base_estimator
        self.scale_factor = scale_factor

    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.base_estimator_.predict(X) * self.scale_factor

    def get_prediction_std(self, X):
        if hasattr(self.base_estimator_, "estimators_"):
            predictions = np.array([tree.predict(X) for tree in self.base_estimator_.estimators_])
            return np.std(predictions, axis=0)
        return np.ones(X.shape[0]) * 100.0

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator, "scale_factor": self.scale_factor}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ABAA joblib/pickle model.")
    parser.add_argument("model")
    args = parser.parse_args()

    # Historical files stored the wrapper as __main__.ConservativeRegressor.
    __main__.ConservativeRegressor = ConservativeRegressor
    model = joblib.load(args.model)

    print(f"Object type: {type(model).__name__}")
    if hasattr(model, "named_steps"):
        print("Pipeline steps:")
        for name, step in model.named_steps.items():
            print(f"  {name}: {type(step).__name__}")
            if hasattr(step, "get_params"):
                params = step.get_params(deep=False)
                for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "alpha", "scale_factor"):
                    if key in params:
                        print(f"    {key}: {params[key]}")
                base = params.get("base_estimator")
                if base is not None:
                    print(f"    base_estimator: {type(base).__name__}")
                    for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"):
                        if hasattr(base, key):
                            print(f"      {key}: {getattr(base, key)}")
    else:
        print(model)

    if hasattr(model, "metadata"):
        print("Metadata:")
        print(model.metadata)


if __name__ == "__main__":
    main()

import shap
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor, VotingRegressor

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def _transform_X_through_pipeline(pipe: Pipeline, X: np.ndarray) -> tuple[np.ndarray, object]:
    """Applies all pipeline steps EXCEPT the last (the model) to X.

    Returns (X_transformed, final_estimator).
    """
    if not isinstance(pipe, Pipeline):
        return X, pipe
    steps = pipe.steps
    final_name, final_est = steps[-1]
    X_t = X
    for name, step in steps[:-1]:
        X_t = step.transform(X_t)
    return np.asarray(X_t), final_est


def _explain_one(estimator, X_for_explainer: np.ndarray, X_background: np.ndarray) -> tuple[np.ndarray, float]:
    """SHAP for a single model (after extraction from the pipeline)."""
    is_tree_based = (
        hasattr(estimator, "feature_importances_")
        or hasattr(estimator, "estimators_")
        or "tree" in str(type(estimator)).lower()
        or "boost" in str(type(estimator)).lower()
        or "forest" in str(type(estimator)).lower()
    )

    if is_tree_based:
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_for_explainer)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:                     # (rows, features, classes): keep the positive class
            sv = sv[:, :, -1]
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(np.asarray(ev).flatten()[-1])
        return np.asarray(sv), float(ev)

    # Linear models
    explainer = shap.LinearExplainer(estimator, X_background)
    sv = explainer.shap_values(X_for_explainer)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(np.asarray(ev).flatten()[0])
    return np.asarray(sv), float(ev)


def compute_and_save_shap(model, X_latest: np.ndarray, feature_names: list, symbol_task: str,
                           X_background: np.ndarray | None = None):
    """Correct SHAP for Voting/Stacking/Pipeline/plain models.
    A StandardScaler inside the Pipeline is applied before the Explainer.
    """
    print(f"[SHAP] {symbol_task}: running...")
    X_latest = np.asarray(X_latest)
    X_background = np.asarray(X_background) if X_background is not None else X_latest

    try:
        if isinstance(model, StackingRegressor) and hasattr(model.final_estimator_, "coef_"):
            # linear meta-model: the SHAP values of the stack are exactly the
            # coefficient-weighted sum of the SHAP values of its members
            print(f"[SHAP] {symbol_task}: stacking regressor with a linear meta-model")
            coef = np.ravel(model.final_estimator_.coef_)
            sv_sum, base_value = np.zeros((len(X_latest), X_latest.shape[1])), float(np.ravel(model.final_estimator_.intercept_)[0])
            for c, est in zip(coef, model.estimators_):
                Xl, final_est = _transform_X_through_pipeline(est, X_latest) if isinstance(est, Pipeline) else (X_latest, est)
                Xb = _transform_X_through_pipeline(est, X_background)[0] if isinstance(est, Pipeline) else X_background
                sv, ev = _explain_one(final_est, Xl, Xb)
                sv_sum += c * np.asarray(sv).reshape(len(X_latest), -1)
                base_value += c * ev
            shap_values = sv_sum

        elif isinstance(model, (VotingClassifier, StackingClassifier, VotingRegressor)):
            print(f"[SHAP] {symbol_task}: ensemble ({type(model).__name__})")
            shap_values_list = []
            base_values = []
            estimators = (
                list(model.named_estimators_.items())
                if hasattr(model, "named_estimators_") else
                list(zip([f"e{i}" for i in range(len(model.estimators_))], model.estimators_))
            )
            w_attr = getattr(model, "weights", None)
            vote_w = [1.0] * len(estimators) if w_attr is None else [float(v) for v in w_attr]
            used_w = []
            for (name, est), w in zip(estimators, vote_w):
                if isinstance(est, Pipeline):
                    Xl, final_est = _transform_X_through_pipeline(est, X_latest)
                    Xb, _ = _transform_X_through_pipeline(est, X_background)
                else:
                    Xl, final_est = X_latest, est
                    Xb = X_background

                try:
                    sv, ev = _explain_one(final_est, Xl, Xb)
                    shap_values_list.append(sv)
                    base_values.append(ev)
                    used_w.append(float(w))
                except Exception as e:
                    print(f"[SHAP] {symbol_task}: skipped {name} ({e})")

            if not shap_values_list:
                raise RuntimeError("Could not compute SHAP for any sub-model")

            # members are combined with the voting weights; note that tree members are
            # explained on their raw (log-odds) scale, so the result is an attribution
            # of the weighted member scores rather than of the averaged probability
            arr = np.stack([np.asarray(s).reshape(len(X_latest), -1) for s in shap_values_list], axis=0)
            wts = np.asarray(used_w) / np.sum(used_w)
            shap_values = np.tensordot(wts, arr, axes=1)
            base_value = float(np.dot(wts, base_values))

        elif isinstance(model, Pipeline):
            Xl, final_est = _transform_X_through_pipeline(model, X_latest)
            Xb, _ = _transform_X_through_pipeline(model, X_background)
            shap_values, base_value = _explain_one(final_est, Xl, Xb)

        else:
            shap_values, base_value = _explain_one(model, X_latest, X_background)

        sv_arr = np.asarray(shap_values)
        if sv_arr.ndim == 1:
            sv_arr = sv_arr.reshape(1, -1)

        first_row = sv_arr[0].tolist()

        result = {
            "symbol_task": symbol_task,
            "shap_values": first_row,
            "feature_names": list(feature_names),
            "base_value": float(base_value),
            "generated_at": pd.Timestamp.now("UTC").isoformat()
        }

        filepath = ARTIFACTS / f"shap_{symbol_task}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[SHAP] {symbol_task}: saved -> {filepath}")
        return result

    except Exception as e:
        print(f"[SHAP ERROR] {symbol_task}: {e}")
        return None

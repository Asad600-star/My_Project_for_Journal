"""SHAP attributions of the soft-voting ensemble (regression test for the shape bug)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import core.explain.shap_explainer as se


def test_voting_ensemble_shap_has_one_value_per_feature(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "ARTIFACTS", tmp_path)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6))
    y = (X[:, 0] - X[:, 1] + 0.5 * rng.standard_normal(400) > 0).astype(int)
    model = VotingClassifier(
        estimators=[("lr", Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())])),
                    ("hgb", HistGradientBoostingClassifier(max_iter=50)),
                    ("rf", RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0))],
        voting="soft", weights=[1, 2, 1]).fit(X, y)
    out = se.compute_and_save_shap(model, X[-1:], [f"f{i}" for i in range(6)], "TEST_direction", X[:100])
    assert out is not None
    assert len(out["shap_values"]) == 6
    # the two informative features dominate the attribution
    top2 = set(np.argsort(np.abs(out["shap_values"]))[-2:])
    assert top2 == {0, 1}
    assert (tmp_path / "shap_TEST_direction.json").exists()


def test_stacking_regressor_shap_is_exact(tmp_path, monkeypatch):
    """For a linear meta-model the attributions add up to the stacked prediction."""
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
    from sklearn.linear_model import Ridge
    monkeypatch.setattr(se, "ARTIFACTS", tmp_path)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5))
    y = np.abs(X[:, 0]) + 0.5 * X[:, 2] + 0.1 * rng.standard_normal(300)
    model = StackingRegressor([("et", ExtraTreesRegressor(n_estimators=50, random_state=0)),
                               ("hgb", HistGradientBoostingRegressor(max_iter=50))],
                              final_estimator=Ridge(alpha=1.0), cv=3).fit(X, y)
    out = se.compute_and_save_shap(model, X[-1:], [f"f{i}" for i in range(5)], "TEST_volatility", X[:100])
    assert out is not None
    assert np.isclose(sum(out["shap_values"]) + out["base_value"], model.predict(X[-1:])[0], atol=1e-6)

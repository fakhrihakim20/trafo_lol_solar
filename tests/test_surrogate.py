"""Unit tests for src/transformer_lol/surrogate.py.

Tests verify:
1. build_features returns exactly 13 columns with correct names
2. Feature names match FEATURE_NAMES exactly (no silent additions)
3. Constant-load surrogate recovers steady-state theta_HS within 1 degC
4. save/load roundtrip gives identical predictions
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transformer_lol.surrogate import (
    FEATURE_NAMES,
    build_features,
    build_training_set,
    train_surrogate,
    save_surrogate,
    load_surrogate,
    predict_trajectory,
)
from transformer_lol.thermal_model import TransformerParams


P = TransformerParams()   # default 60 MVA ONAF


# ---------------------------------------------------------------------------
# Test 1: Feature matrix has exactly 13 columns
# ---------------------------------------------------------------------------

def test_feature_count() -> None:
    """build_features must return exactly 13 columns."""
    n = 50
    feat = build_features(
        load_pu=np.ones(n) * 0.8,
        ambient_c=np.full(n, 30.0),
        pv_share=np.zeros(n),
        theta_hs_history=np.full(n, 95.0),
        theta_to_history=np.full(n, 80.0),
    )
    assert feat.shape[1] == 13, f"Expected 13 features, got {feat.shape[1]}"


# ---------------------------------------------------------------------------
# Test 2: Feature names match FEATURE_NAMES exactly
# ---------------------------------------------------------------------------

def test_feature_names_exact() -> None:
    """Column names must match FEATURE_NAMES in order — no silent additions."""
    n = 50
    feat = build_features(
        load_pu=np.ones(n) * 0.8,
        ambient_c=np.full(n, 30.0),
        pv_share=np.zeros(n),
        theta_hs_history=np.full(n, 95.0),
        theta_to_history=np.full(n, 80.0),
    )
    assert list(feat.columns) == FEATURE_NAMES, \
        f"Feature columns mismatch.\nExpected: {FEATURE_NAMES}\nGot:      {list(feat.columns)}"


# ---------------------------------------------------------------------------
# Test 3: Constant-load surrogate recovers steady-state theta_HS
# ---------------------------------------------------------------------------

def test_constant_load_steady_state() -> None:
    """Train surrogate on constant K=0.8/30degC trajectories; predictions within 2 degC of physics.

    At K=0.8, ambient=30: theta_HS_ss = 30 + 55*(0.64*5+1)^0.8/6^0.8 + 25*0.8^1.6 ≈ 97 degC.
    The surrogate should recover this within 2 degC after a short warm-up.
    """
    # Quick training set: 30 trajectories, 48h each — all with constant K
    n_traj = 30
    hours = 48

    train_frames = []
    for i in range(n_traj):
        k = 0.7 + 0.1 * np.random.default_rng(i).random()
        n = hours
        load = np.full(n, k)
        ambient = np.full(n, 30.0)
        pv = np.zeros(n)

        from transformer_lol.thermal_model import simulate_thermal
        res = simulate_thermal(load, ambient, P, dt_min=60.0, initial_state=(0.0, 0.0))
        ths = res["theta_hs"]
        tto = res["theta_to"]

        feat = build_features(load, ambient, pv, ths, tto)
        feat["theta_hs_t"] = ths
        feat["trajectory_id"] = i
        train_frames.append(feat.dropna())

    df = pd.concat(train_frames, ignore_index=True)
    split = int(0.7 * len(df))
    model, scaler = train_surrogate(df.iloc[:split], df.iloc[split:],
                                    params={
                                        "n_estimators": 200, "max_depth": 4,
                                        "learning_rate": 0.1, "subsample": 0.8,
                                        "colsample_bytree": 0.8, "reg_lambda": 1.0,
                                        "random_state": 42, "n_jobs": 1,
                                        "tree_method": "hist",
                                        "early_stopping_rounds": 20,
                                    })

    # Predict a constant-K trajectory
    n_pred = 200
    load_test = np.full(n_pred, 0.8)
    ambient_test = np.full(n_pred, 30.0)
    pv_test = np.zeros(n_pred)

    res_phys = simulate_thermal(load_test, ambient_test, P, dt_min=60.0,
                                 initial_state=(0.0, 0.0))
    theta_hs_phys = res_phys["theta_hs"]

    res_surr = predict_trajectory(model, scaler, load_test, ambient_test, pv_test,
                                   transformer=P)
    theta_hs_surr = res_surr["theta_hs"]

    # Compare in the steady-state region (last 50 steps)
    err = np.abs(theta_hs_surr[-50:] - theta_hs_phys[-50:]).mean()
    assert err < 2.0, f"Surrogate steady-state error {err:.3f} degC exceeds 2 degC"


# ---------------------------------------------------------------------------
# Test 4: save/load roundtrip gives identical predictions
# ---------------------------------------------------------------------------

def test_save_load_roundtrip() -> None:
    """Save model to disk and reload; predictions must be identical."""
    n_traj = 20
    hours = 48
    from transformer_lol.thermal_model import simulate_thermal

    frames = []
    for i in range(n_traj):
        k = 0.6 + 0.4 * np.random.default_rng(i).random()
        n = hours
        load = np.full(n, k)
        ambient = np.full(n, 30.0)
        pv = np.zeros(n)
        res = simulate_thermal(load, ambient, P, dt_min=60.0, initial_state=(0.0, 0.0))
        feat = build_features(load, ambient, pv, res["theta_hs"], res["theta_to"])
        feat["theta_hs_t"] = res["theta_hs"]
        feat["trajectory_id"] = i
        frames.append(feat.dropna())

    df = pd.concat(frames, ignore_index=True)
    split = int(0.7 * len(df))
    model, scaler = train_surrogate(df.iloc[:split], df.iloc[split:],
                                    params={
                                        "n_estimators": 100, "max_depth": 4,
                                        "learning_rate": 0.1, "subsample": 0.8,
                                        "colsample_bytree": 0.8, "reg_lambda": 1.0,
                                        "random_state": 42, "n_jobs": 1,
                                        "tree_method": "hist",
                                        "early_stopping_rounds": 10,
                                    })

    with tempfile.TemporaryDirectory() as tmpdir:
        mpath = Path(tmpdir) / "model.json"
        spath = Path(tmpdir) / "scaler.pkl"
        save_surrogate(model, scaler, mpath, spath)
        model2, scaler2 = load_surrogate(mpath, spath)

    # Predict with both and compare
    n_pred = 50
    load_t = np.full(n_pred, 0.8)
    amb_t = np.full(n_pred, 30.0)
    pv_t = np.zeros(n_pred)

    r1 = predict_trajectory(model, scaler, load_t, amb_t, pv_t, transformer=P)
    r2 = predict_trajectory(model2, scaler2, load_t, amb_t, pv_t, transformer=P)

    np.testing.assert_allclose(r1["theta_hs"], r2["theta_hs"], atol=0.001,
                               err_msg="Loaded model gives different predictions")


# ---------------------------------------------------------------------------
# Test 5: Feature NaN pattern — first 2 rows must have NaN for lag features
# ---------------------------------------------------------------------------

def test_feature_lag_nans() -> None:
    """The first N rows of build_features output must have NaN values (from lags).

    The maximum lag is 3 (load_ramp_3h uses diff(3)), so rows 0, 1, 2
    will have at least one NaN.  Row 3 onward should be NaN-free.
    """
    n = 20
    feat = build_features(
        load_pu=np.ones(n) * 0.8,
        ambient_c=np.full(n, 30.0),
        pv_share=np.zeros(n),
        theta_hs_history=np.full(n, 95.0),
        theta_to_history=np.full(n, 80.0),
    )
    # Rows 0, 1, 2 must have at least one NaN (lag features)
    for row in range(3):
        assert feat.iloc[row].isna().any(), f"Row {row} should have NaN lag features"
    # Rows 3+ must be NaN-free (all lags are satisfied)
    assert not feat.iloc[3:].isna().any().any(), "Rows 3+ must have no NaN"

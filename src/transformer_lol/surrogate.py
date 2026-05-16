"""XGBoost autoregressive surrogate for transformer hot-spot temperature (theta_HS).

Architecture
------------
Single-step regression: predict theta_HS(t) from lagged features at t and t-1, t-2.
13 features per timestep (fixed set — no silent additions):

  load_pu_t    load_pu_t1   load_pu_t2   (current loading + 2 lags)
  ambient_c_t  ambient_c_t1             (current + 1 lag)
  pv_share_t                            (net PV contribution, signed)
  hour_sin     hour_cos                 (cyclic hour encoding)
  theta_hs_t1  theta_hs_t2             (autoregressive lags — KEY contribution)
  theta_to_t1                           (top-oil lag)
  load_ramp_1h load_ramp_3h            (duck-curve ramp signal)

Target: theta_hs_t

Training: teacher forcing (true theta_HS lags from physics simulation).
Inference: autoregressive rollout (predicted theta_HS fed back as next lag).

Validation gate (enforced in scripts/03_train_surrogate.py):
  RMSE < 1.5 °C on held-out physics trajectories
  R²   > 0.99
  Annual LoL MAPE < 5% vs physics

References
----------
Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
    KDD '16. ACM.  (XGBoost algorithm)
IEEE C57.91-2011, Clause 7 (thermal model whose output trains this surrogate).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .aging import loss_of_life_percent
from .thermal_model import TransformerParams, simulate_thermal
from .data_jember import (
    load_jember_ambient,
    load_jember_load,
    load_jember_pv,
)


# ---------------------------------------------------------------------------
# Feature list (fixed — do not change without updating all callers)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "load_pu_t",
    "load_pu_t1",
    "load_pu_t2",
    "ambient_c_t",
    "ambient_c_t1",
    "pv_share_t",
    "hour_sin",
    "hour_cos",
    "theta_hs_t1",
    "theta_hs_t2",
    "theta_to_t1",
    "load_ramp_1h",
    "load_ramp_3h",
]

assert len(FEATURE_NAMES) == 13, "Feature list must have exactly 13 entries."

# ---------------------------------------------------------------------------
# Default XGBoost hyperparameters (do not grid-search; calibrated defaults)
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(
    load_pu: np.ndarray,
    ambient_c: np.ndarray,
    pv_share: np.ndarray,
    theta_hs_history: np.ndarray,
    theta_to_history: np.ndarray,
    dt_hours: float = 1.0,
) -> pd.DataFrame:
    """Build the 13-column feature matrix for the surrogate model.

    Parameters
    ----------
    load_pu : np.ndarray, shape (T,)
        Per-unit loading at each timestep.
    ambient_c : np.ndarray, shape (T,)
        Ambient temperature [°C].
    pv_share : np.ndarray, shape (T,)
        PV generation in per-unit of transformer rated MVA (same scale as load_pu).
    theta_hs_history : np.ndarray, shape (T,)
        Hot-spot temperature [°C].  At training time, use true physics values.
        At inference, replace with model predictions step by step.
    theta_to_history : np.ndarray, shape (T,)
        Top-oil temperature [°C].
    dt_hours : float
        Timestep size [hours].  Used for ramp computation.

    Returns
    -------
    pd.DataFrame, shape (T, 13)
        Feature matrix.  Rows 0, 1 have NaN lag features and must be dropped
        before model fitting.  Column order matches FEATURE_NAMES exactly.

    Notes
    -----
    The first max_lag=2 rows will contain NaN values.  Drop these with
    `df.dropna()` before passing to XGBoost.
    """
    T = len(load_pu)
    df = pd.DataFrame(index=range(T))

    # Lagged load features
    series_load = pd.Series(load_pu)
    df["load_pu_t"] = series_load.values
    df["load_pu_t1"] = series_load.shift(1).values
    df["load_pu_t2"] = series_load.shift(2).values

    # Lagged ambient features
    series_amb = pd.Series(ambient_c)
    df["ambient_c_t"] = series_amb.values
    df["ambient_c_t1"] = series_amb.shift(1).values

    # PV share (current; no lag — exogenous driver)
    df["pv_share_t"] = pv_share

    # Cyclic hour-of-day encoding (hour index = 0…23, repeating)
    hour_idx = np.arange(T) % 24
    df["hour_sin"] = np.sin(2.0 * np.pi * hour_idx / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour_idx / 24.0)

    # Autoregressive lags (theta_HS, theta_TO)
    series_hs = pd.Series(theta_hs_history)
    series_to = pd.Series(theta_to_history)
    df["theta_hs_t1"] = series_hs.shift(1).values
    df["theta_hs_t2"] = series_hs.shift(2).values
    df["theta_to_t1"] = series_to.shift(1).values

    # Ramp features: load(t) - load(t-n) measures duck-curve steepness
    steps_1h = max(1, int(round(1.0 / dt_hours)))    # steps in 1 hour
    steps_3h = max(1, int(round(3.0 / dt_hours)))    # steps in 3 hours
    df["load_ramp_1h"] = series_load.diff(steps_1h).values
    df["load_ramp_3h"] = series_load.diff(steps_3h).values

    return df[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def build_training_set(
    n_trajectories: int = 200,
    hours_per_trajectory: int = 168,    # 1 week
    transformer: TransformerParams | None = None,
    dt_hours: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate diverse (feature, target) trajectories using the physics ODE.

    Samples scenario parameters (PV penetration, ambient offset, load growth)
    uniformly over their plausible ranges to give the surrogate diverse thermal
    dynamics across all operating conditions expected in the Monte Carlo sweep.

    Parameters
    ----------
    n_trajectories : int
        Number of independent week-long trajectories.
    hours_per_trajectory : int
        Length of each trajectory in hours.
    transformer : TransformerParams, optional
        Transformer parameters.  Defaults to ONAF 60 MVA parameters.
    dt_hours : float
        Timestep size in hours.  1.0 = hourly.
    seed : int
        Master seed.  Trajectory i uses seed = seed + i.

    Returns
    -------
    pd.DataFrame
        Feature matrix + target column 'theta_hs_t'.
        Rows with NaN (first 2 of each trajectory) are dropped.
    """
    if transformer is None:
        transformer = TransformerParams()

    rng = np.random.default_rng(seed)
    all_frames: list[pd.DataFrame] = []

    for i in range(n_trajectories):
        traj_seed = seed + i

        # Sample scenario parameters uniformly
        pv_pen = float(rng.uniform(0.0, 0.80))
        amb_delta = float(rng.uniform(0.0, 2.0))
        load_growth = float(rng.uniform(0.0, 4.5))
        load_scale = 1.0 + load_growth / 100.0   # convert %/yr to multiplier

        # Generate one trajectory's worth of profiles
        n_steps = hours_per_trajectory
        # Phase 4C: training data uses Jember loaders so the surrogate sees
        # the same statistical distribution as production MC inference.
        load_base = load_jember_load(n_steps, peak_pu=0.85, seed=traj_seed)
        load_pu = np.clip(load_base * load_scale, 0.0, 1.5)
        ambient_c = load_jember_ambient(
            n_steps, delta_c=amb_delta, seed=traj_seed
        )
        pv_gen = load_jember_pv(
            n_steps, penetration_pu=pv_pen, seed=traj_seed
        )
        net_load = np.clip(load_pu - pv_gen, 0.0, 1.5)

        # Physics simulation
        result = simulate_thermal(net_load, ambient_c, transformer,
                                  dt_min=dt_hours * 60.0)
        theta_hs = result["theta_hs"]
        theta_to = result["theta_to"]

        # Build feature frame
        feat = build_features(
            load_pu=net_load,
            ambient_c=ambient_c,
            pv_share=pv_gen,
            theta_hs_history=theta_hs,
            theta_to_history=theta_to,
            dt_hours=dt_hours,
        )
        feat["theta_hs_t"] = theta_hs   # target
        feat["trajectory_id"] = i
        feat = feat.dropna()
        all_frames.append(feat)

    if not all_frames:
        raise RuntimeError("No training data generated — check inputs.")
    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------

def train_surrogate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> tuple[xgb.XGBRegressor, StandardScaler]:
    """Train the XGBoost surrogate on the provided feature DataFrames.

    Uses chronological (not shuffled) train/val split.
    StandardScaler is fitted on training data only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training feature rows.  Must contain FEATURE_NAMES + 'theta_hs_t'.
    val_df : pd.DataFrame
        Validation feature rows.  Same schema.
    params : dict, optional
        XGBoost hyperparameters.  Defaults to DEFAULT_XGB_PARAMS.

    Returns
    -------
    (xgb.XGBRegressor, StandardScaler)
        Trained model and fitted scaler.

    Raises
    ------
    ValueError
        If RMSE >= 1.5 °C or R² <= 0.99 on the validation set.
    """
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()

    X_train = train_df[FEATURE_NAMES].values.astype(float)
    y_train = train_df["theta_hs_t"].values.astype(float)
    X_val = val_df[FEATURE_NAMES].values.astype(float)
    y_val = val_df["theta_hs_t"].values.astype(float)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Convert to DMatrix with feature names for XGBoost 2.x compatibility
    X_train_df = pd.DataFrame(X_train_s, columns=FEATURE_NAMES)
    X_val_df = pd.DataFrame(X_val_s, columns=FEATURE_NAMES)

    # Train with early stopping
    xgb_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early_stop = params.get("early_stopping_rounds", 50)

    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=early_stop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train_df, y_train,
            eval_set=[(X_val_df, y_val)],
            verbose=False,
        )

    # Validation metrics
    y_pred = model.predict(X_val_df)
    rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    ss_res = np.sum((y_pred - y_val) ** 2)
    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    if rmse >= 1.5 or r2 <= 0.99:
        raise ValueError(
            f"Surrogate validation gate FAILED: RMSE={rmse:.4f} °C "
            f"(threshold < 1.5), R²={r2:.6f} (threshold > 0.99).\n"
            "Increase n_trajectories in build_training_set() or check feature engineering."
        )

    return model, scaler


# ---------------------------------------------------------------------------
# Autoregressive inference
# ---------------------------------------------------------------------------

def predict_trajectory(
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    load_pu: np.ndarray,
    ambient_c: np.ndarray,
    pv_share: np.ndarray,
    initial_theta_hs: float = 80.0,
    initial_theta_to: float = 60.0,
    dt_hours: float = 1.0,
    warmup_steps: int = 12,
    transformer: TransformerParams | None = None,
    net_load_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Autoregressive surrogate rollout for one trajectory.

    Uses ODE simulation for the first `warmup_steps` timesteps to avoid
    cold-start error accumulation, then hands off to the surrogate.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained surrogate model.
    scaler : StandardScaler
        Feature scaler fitted during training.
    load_pu, ambient_c, pv_share : np.ndarray, shape (T,)
        Input profiles (same convention as build_features).
    initial_theta_hs : float
        Initial hot-spot temperature [°C].
    initial_theta_to : float
        Initial top-oil temperature [°C].
    dt_hours : float
        Timestep size [hours].
    warmup_steps : int
        Number of initial steps to simulate with ODE before handing to surrogate.
    transformer : TransformerParams, optional
        Required for ODE warm-up.  Defaults to standard 60 MVA ONAF params.
    net_load_override : np.ndarray, optional
        Pre-computed net load (e.g. after battery dispatch).  When provided,
        replaces the internal ``clip(load_pu - pv_share, 0, 1.5)`` calculation
        so that peak-shaving effects reach the thermal model.

    Returns
    -------
    dict with keys 'theta_hs', 'theta_to'
        Arrays of shape (T,).
    """
    if transformer is None:
        transformer = TransformerParams()

    T = len(load_pu)
    net_load = (net_load_override
                if net_load_override is not None
                else np.clip(load_pu - pv_share, 0.0, 1.5))

    # ODE warm-up for the first warmup_steps
    warmup = min(warmup_steps, T)
    delta_to0 = initial_theta_to - ambient_c[0]
    delta_hs0 = initial_theta_hs - initial_theta_to

    ode_result = simulate_thermal(
        net_load[:warmup],
        ambient_c[:warmup],
        transformer,
        dt_min=dt_hours * 60.0,
        initial_state=(delta_to0, delta_hs0),
    )
    theta_hs_arr = np.empty(T)
    theta_to_arr = np.empty(T)
    theta_hs_arr[:warmup] = ode_result["theta_hs"]
    theta_to_arr[:warmup] = ode_result["theta_to"]

    if warmup >= T:
        return {"theta_hs": theta_hs_arr, "theta_to": theta_to_arr}

    # Autoregressive rollout from step warmup onward
    # We maintain running arrays and predict one step at a time
    for t in range(warmup, T):
        # Build features for step t using known history
        feat = _build_single_step_features(
            t=t,
            load_pu=net_load,
            ambient_c=ambient_c,
            pv_share=pv_share,
            theta_hs_arr=theta_hs_arr,
            theta_to_arr=theta_to_arr,
            dt_hours=dt_hours,
        )
        feat_s = scaler.transform(feat)
        feat_df = pd.DataFrame(feat_s, columns=FEATURE_NAMES)
        theta_hs_arr[t] = float(model.predict(feat_df)[0])
        # Approximate theta_to: use previous value + small correction
        # (full ODE warm-up provides accurate theta_to for the lags)
        # For surrogate speed, theta_to is not re-predicted — use last ODE value
        # with exponential decay toward steady state.
        if t > 0:
            theta_to_arr[t] = theta_to_arr[t - 1] + 0.1 * (
                theta_hs_arr[t] - theta_to_arr[t - 1] - 25.0
            )
        else:
            theta_to_arr[t] = theta_to_arr[warmup - 1]

    return {"theta_hs": theta_hs_arr, "theta_to": theta_to_arr}


def predict_trajectories_batch(
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    load_arr: np.ndarray,
    ambient_arr: np.ndarray,
    pv_arr: np.ndarray,
    transformer: TransformerParams | None = None,
    dt_hours: float = 1.0,
    net_load_arr: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Autoregressive surrogate rollout for n_runs trajectories simultaneously.

    At each time step a single model.predict() call covers all n_runs samples,
    amortising the Python/XGBoost call overhead over the batch.  Roughly 100x
    faster than calling predict_trajectory() in a Python loop.

    Initial conditions use the analytical IEEE C57.91 steady-state at K=load[0],
    which is accurate enough for annual MC metrics (first 12 h contribute <0.15%
    of annual LoL).

    Parameters
    ----------
    model : xgb.XGBRegressor
    scaler : StandardScaler
    load_arr : np.ndarray, shape (n_runs, T)
        Per-unit loading (before PV subtraction).
    ambient_arr : np.ndarray, shape (n_runs, T)
        Ambient temperature [°C].
    pv_arr : np.ndarray, shape (n_runs, T)
        PV generation [pu of transformer MVA].
    transformer : TransformerParams, optional
    dt_hours : float
    net_load_arr : np.ndarray, shape (n_runs, T), optional
        Pre-computed net load after battery dispatch.  When provided, replaces
        the internal ``clip(load_arr - pv_arr, 0, 1.5)`` calculation so that
        battery peak-shaving effects reach the thermal surrogate.

    Returns
    -------
    dict with keys 'theta_hs', 'theta_to', each shape (n_runs, T).
    """
    if transformer is None:
        transformer = TransformerParams()

    n_runs, T = load_arr.shape
    net_load = (net_load_arr
                if net_load_arr is not None
                else np.clip(load_arr - pv_arr, 0.0, 1.5))

    # Analytical steady-state initial conditions (vectorised over n_runs)
    K0 = net_load[:, 0]  # (n_runs,) — initial loading
    dTO_ss = (transformer.delta_theta_to_r
              * ((K0 ** 2 * transformer.R + 1.0) / (transformer.R + 1.0))
              ** transformer.n)
    dHS_ss = transformer.delta_theta_hs_r * K0 ** (2.0 * transformer.m)
    theta_to_init = ambient_arr[:, 0] + dTO_ss   # (n_runs,)
    theta_hs_init = theta_to_init + dHS_ss        # (n_runs,)

    theta_hs_all = np.empty((n_runs, T), dtype=float)
    theta_to_all = np.empty((n_runs, T), dtype=float)
    theta_hs_all[:, 0] = theta_hs_init
    theta_to_all[:, 0] = theta_to_init

    # Pre-compute non-autoregressive features for all T steps (vectorised)
    t_idx = np.arange(T)
    hours = (t_idx % 24).astype(float)
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)   # (T,)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)   # (T,)

    # Lag arrays: pad left edge with the first value
    nl_t1 = np.concatenate([net_load[:, :1], net_load[:, :-1]], axis=1)  # (n_runs, T)
    nl_t2 = np.concatenate([net_load[:, :1], net_load[:, :1], net_load[:, :-2]], axis=1)
    amb_t1 = np.concatenate([ambient_arr[:, :1], ambient_arr[:, :-1]], axis=1)
    nl_t3 = np.concatenate([net_load[:, :1], net_load[:, :1], net_load[:, :1],
                             net_load[:, :-3]], axis=1)
    ramp_1h = net_load - nl_t1   # (n_runs, T)
    ramp_3h = net_load - nl_t3   # (n_runs, T)

    # Pre-allocate feature buffer (n_runs, 13) — reused each step
    X_buf = np.empty((n_runs, 13), dtype=float)

    for t in range(1, T):
        # Columns 0-7,11-12: non-AR (pre-computed above)
        X_buf[:, 0] = net_load[:, t]
        X_buf[:, 1] = nl_t1[:, t]
        X_buf[:, 2] = nl_t2[:, t]
        X_buf[:, 3] = ambient_arr[:, t]
        X_buf[:, 4] = amb_t1[:, t]
        X_buf[:, 5] = pv_arr[:, t]
        X_buf[:, 6] = hour_sin[t]
        X_buf[:, 7] = hour_cos[t]
        # Columns 8-10: autoregressive lags
        X_buf[:, 8] = theta_hs_all[:, t - 1]
        X_buf[:, 9] = theta_hs_all[:, t - 2] if t >= 2 else theta_hs_all[:, t - 1]
        X_buf[:, 10] = theta_to_all[:, t - 1]
        X_buf[:, 11] = ramp_1h[:, t]
        X_buf[:, 12] = ramp_3h[:, t]

        X_scaled = scaler.transform(X_buf)          # (n_runs, 13)
        # Pass numpy array directly; suppress feature-name warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict(X_scaled)          # (n_runs,)

        theta_hs_all[:, t] = preds
        # Approximate theta_to update (matches predict_trajectory logic)
        theta_to_all[:, t] = (theta_to_all[:, t - 1]
                               + 0.1 * (preds - theta_to_all[:, t - 1] - 25.0))

    return {"theta_hs": theta_hs_all, "theta_to": theta_to_all}


def _build_single_step_features(
    t: int,
    load_pu: np.ndarray,
    ambient_c: np.ndarray,
    pv_share: np.ndarray,
    theta_hs_arr: np.ndarray,
    theta_to_arr: np.ndarray,
    dt_hours: float,
) -> np.ndarray:
    """Build a (1, 13) feature array for step t using history from arrays.

    This is the inner loop of predict_trajectory and is performance-critical.
    Returns a 2D numpy array of shape (1, 13) ready for scaler.transform().
    """
    steps_1h = max(1, int(round(1.0 / dt_hours)))
    steps_3h = max(1, int(round(3.0 / dt_hours)))

    lpu_t = load_pu[t]
    lpu_t1 = load_pu[t - 1] if t >= 1 else load_pu[0]
    lpu_t2 = load_pu[t - 2] if t >= 2 else load_pu[0]
    amb_t = ambient_c[t]
    amb_t1 = ambient_c[t - 1] if t >= 1 else ambient_c[0]
    pv_t = pv_share[t]
    hour_idx = t % 24
    hs_t1 = theta_hs_arr[t - 1] if t >= 1 else 80.0
    hs_t2 = theta_hs_arr[t - 2] if t >= 2 else 80.0
    to_t1 = theta_to_arr[t - 1] if t >= 1 else 60.0
    ramp_1h = load_pu[t] - (load_pu[t - steps_1h] if t >= steps_1h else load_pu[0])
    ramp_3h = load_pu[t] - (load_pu[t - steps_3h] if t >= steps_3h else load_pu[0])

    row = np.array([[
        lpu_t, lpu_t1, lpu_t2,
        amb_t, amb_t1,
        pv_t,
        np.sin(2.0 * np.pi * hour_idx / 24.0),
        np.cos(2.0 * np.pi * hour_idx / 24.0),
        hs_t1, hs_t2,
        to_t1,
        ramp_1h, ramp_3h,
    ]], dtype=float)
    return row


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def save_surrogate(
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    model_path: str | Path,
    scaler_path: str | Path,
) -> None:
    """Save model and scaler to disk.

    Parameters
    ----------
    model_path : str or Path
        Path for the XGBoost JSON model file.
    scaler_path : str or Path
        Path for the pickled StandardScaler.
    """
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)


def load_surrogate(
    model_path: str | Path,
    scaler_path: str | Path,
) -> tuple[xgb.XGBRegressor, StandardScaler]:
    """Load a previously saved surrogate model and scaler.

    Parameters
    ----------
    model_path : str or Path
        Path to the XGBoost JSON model file.
    scaler_path : str or Path
        Path to the pickled StandardScaler.

    Returns
    -------
    (xgb.XGBRegressor, StandardScaler)

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    """
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    for p in (model_path, scaler_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Surrogate file not found: {p.resolve()}\n"
                "Run scripts/03_train_surrogate.py first."
            )
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    return model, scaler

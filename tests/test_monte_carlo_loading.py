"""Tests for Monte Carlo thermal loading conventions."""

from __future__ import annotations

import numpy as np
import pytest

from transformer_lol.monte_carlo import effective_thermal_loading
from transformer_lol.scenarios import Scenario


def test_effective_thermal_loading_clips_reverse_flow_by_default() -> None:
    load = np.array([0.40, 0.80, 0.20])
    pv = np.array([0.60, 0.30, 0.20])

    thermal_load = effective_thermal_loading(load, pv)

    np.testing.assert_allclose(thermal_load, np.array([0.0, 0.50, 0.0]))


def test_effective_thermal_loading_reverse_flow_sensitivity() -> None:
    load = np.array([0.40, 0.80])
    pv = np.array([0.60, 0.30])

    thermal_load = effective_thermal_loading(
        load,
        pv,
        reverse_flow_loss_uplift=0.025,
    )

    expected_reverse = np.sqrt(0.025) * 0.20
    np.testing.assert_allclose(thermal_load, np.array([expected_reverse, 0.50]))


def test_effective_thermal_loading_uses_battery_power_sign_convention() -> None:
    load = np.array([0.70, 0.70])
    pv = np.array([0.20, 0.90])
    battery_p = np.array([0.10, -0.05])  # + discharge, - charge

    thermal_load = effective_thermal_loading(load, pv, battery_p_pu=battery_p)

    np.testing.assert_allclose(thermal_load, np.array([0.40, 0.0]))


def test_scenario_rejects_negative_reverse_flow_uplift() -> None:
    with pytest.raises(ValueError, match="reverse_flow_loss_uplift"):
        Scenario(
            name="bad_reverse_flow",
            pv_penetration=0.5,
            reverse_flow_loss_uplift=-0.01,
        )

"""
Task definitions for MicrogridGym environment.

Generates tasks across 4 scenario types with parametric variations
for train/test splits. Each task defines a specific microgrid
configuration and disturbance scenario.
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """Pydantic model for task specification."""
    id: str
    scenario: Literal[
        "voltage_control", "current_control",
        "droop_control", "load_following"
    ]
    # LC filter parameters
    filter_L: float         # H
    filter_C: float         # F
    filter_Rf: float        # ohm
    # Load parameters
    load_R: float           # ohm
    load_L: float           # H
    # Inverter parameters
    inverter_V_DC: float    # V
    inverter_V_nom: float   # V
    inverter_f_nom: float   # Hz
    inverter_i_nom: float = 20.0  # A (nominal current)
    inverter_i_lim: float   # A
    # Episode parameters
    max_steps: int
    dt_control: float       # s
    dt_ode: float           # s
    # Disturbance
    disturbance_step: int | None = None
    load_R_after: float | None = None
    load_L_after: float | None = None
    # Load profile (for load_following)
    load_profile: list[float] | None = None
    # Two-inverter parameters
    num_inverters: int = 1
    filter_L_2: float | None = None
    filter_C_2: float | None = None
    filter_Rf_2: float | None = None
    inverter_V_DC_2: float | None = None
    inverter_i_lim_2: float | None = None


# --------------------------------------------------------------------------- #
#  Base parameters (OMG defaults)
# --------------------------------------------------------------------------- #

BASE = {
    "filter_L": 2.3e-3,
    "filter_C": 10e-6,
    "filter_Rf": 0.4,
    "load_R": 14.0,
    "load_L": 5.3e-3,
    "inverter_V_DC": 600.0,
    "inverter_V_nom": 230.0 * np.sqrt(2),  # 325.27 V peak
    "inverter_f_nom": 50.0,
    "inverter_i_nom": 20.0,
    "inverter_i_lim": 30.0,
    "dt_control": 5e-3,
    "dt_ode": 1e-4,
}


# --------------------------------------------------------------------------- #
#  Task generation functions
# --------------------------------------------------------------------------- #

def _voltage_control_tasks() -> tuple[list[dict], list[dict]]:
    """Generate voltage control tasks.

    Agent tunes cascaded PI gains to track sinusoidal voltage reference.
    Load step disturbance occurs mid-episode.
    """
    train, test = [], []

    # Train: vary L, C, R_load across a grid
    L_values_train = [1.5e-3, 2.0e-3, 2.3e-3, 2.8e-3, 3.5e-3]
    C_values_train = [8e-6, 10e-6, 15e-6]
    R_values_train = [10.0, 14.0, 20.0, 30.0]

    idx = 0
    for L in L_values_train:
        for C in C_values_train:
            for R in R_values_train:
                task = {
                    **BASE,
                    "id": f"voltage_control_train_{idx:03d}",
                    "scenario": "voltage_control",
                    "filter_L": L,
                    "filter_C": C,
                    "load_R": R,
                    "max_steps": 10,
                    "disturbance_step": 5,
                    "load_R_after": R * 0.5,  # Load doubles (R halves)
                }
                train.append(task)
                idx += 1

    # Test: different parameter combinations not in train
    L_values_test = [1.8e-3, 2.5e-3, 3.0e-3]
    R_values_test = [12.0, 18.0, 25.0]

    idx = 0
    for L in L_values_test:
        for R in R_values_test:
            task = {
                **BASE,
                "id": f"voltage_control_test_{idx:03d}",
                "scenario": "voltage_control",
                "filter_L": L,
                "load_R": R,
                "max_steps": 10,
                "disturbance_step": 5,
                "load_R_after": R * 0.5,
            }
            test.append(task)
            idx += 1

    return train, test


def _current_control_tasks() -> tuple[list[dict], list[dict]]:
    """Generate current control tasks.

    Agent tunes PI gains to track current reference through
    the filter inductor. Same cascaded structure but evaluated
    on current tracking rather than voltage.
    """
    train, test = [], []

    L_values_train = [1.5e-3, 2.3e-3, 3.0e-3]
    R_values_train = [10.0, 14.0, 20.0, 30.0, 50.0]

    idx = 0
    for L in L_values_train:
        for R in R_values_train:
            task = {
                **BASE,
                "id": f"current_control_train_{idx:03d}",
                "scenario": "current_control",
                "filter_L": L,
                "load_R": R,
                "max_steps": 8,
                "disturbance_step": 4,
                "load_R_after": R * 0.6,
            }
            train.append(task)
            idx += 1

    L_values_test = [1.8e-3, 2.5e-3]
    R_values_test = [12.0, 25.0, 40.0]

    idx = 0
    for L in L_values_test:
        for R in R_values_test:
            task = {
                **BASE,
                "id": f"current_control_test_{idx:03d}",
                "scenario": "current_control",
                "filter_L": L,
                "load_R": R,
                "max_steps": 8,
                "disturbance_step": 4,
                "load_R_after": R * 0.6,
            }
            test.append(task)
            idx += 1

    return train, test


def _droop_control_tasks() -> tuple[list[dict], list[dict]]:
    """Generate droop control tasks.

    Agent tunes droop coefficients for two parallel inverters
    sharing a common load. Goal is proportional power sharing
    and voltage quality.
    """
    train, test = [], []

    R_values_train = [10.0, 14.0, 20.0, 30.0]
    # Different filter parameters for inverter 2 to create asymmetry
    L2_offsets_train = [0.0, 0.5e-3, -0.3e-3]

    idx = 0
    for R in R_values_train:
        for L2_off in L2_offsets_train:
            L2 = BASE["filter_L"] + L2_off
            task = {
                **BASE,
                "id": f"droop_control_train_{idx:03d}",
                "scenario": "droop_control",
                "load_R": R,
                "max_steps": 12,
                "disturbance_step": 6,
                "load_R_after": R * 0.5,
                "num_inverters": 2,
                "filter_L_2": L2,
                "filter_C_2": BASE["filter_C"],
                "filter_Rf_2": BASE["filter_Rf"],
                "inverter_V_DC_2": BASE["inverter_V_DC"],
                "inverter_i_lim_2": BASE["inverter_i_lim"],
            }
            train.append(task)
            idx += 1

    R_values_test = [12.0, 18.0, 25.0]
    L2_offsets_test = [0.2e-3, -0.5e-3]

    idx = 0
    for R in R_values_test:
        for L2_off in L2_offsets_test:
            L2 = BASE["filter_L"] + L2_off
            task = {
                **BASE,
                "id": f"droop_control_test_{idx:03d}",
                "scenario": "droop_control",
                "load_R": R,
                "max_steps": 12,
                "disturbance_step": 6,
                "load_R_after": R * 0.5,
                "num_inverters": 2,
                "filter_L_2": L2,
                "filter_C_2": BASE["filter_C"],
                "filter_Rf_2": BASE["filter_Rf"],
                "inverter_V_DC_2": BASE["inverter_V_DC"],
                "inverter_i_lim_2": BASE["inverter_i_lim"],
            }
            test.append(task)
            idx += 1

    return train, test


def _load_following_tasks() -> tuple[list[dict], list[dict]]:
    """Generate load following tasks.

    Agent tunes PI gains for a time-varying load. Load resistance
    changes at each step following a predefined profile. Agent may
    re-tune gains each step.
    """
    train, test = [], []

    max_steps = 12

    # Different load profiles
    def make_profile(base_R: float, pattern: str) -> list[float]:
        """Generate a load resistance profile."""
        if pattern == "ramp_down":
            return [base_R * (1 - 0.04 * i) for i in range(max_steps)]
        elif pattern == "ramp_up":
            return [base_R * (0.5 + 0.04 * i) for i in range(max_steps)]
        elif pattern == "step_sequence":
            return [base_R if i < 4 else base_R * 0.5 if i < 8 else base_R * 0.7
                    for i in range(max_steps)]
        elif pattern == "oscillating":
            return [base_R * (1 + 0.3 * np.sin(2 * np.pi * i / 6))
                    for i in range(max_steps)]
        else:
            return [base_R] * max_steps

    L_values_train = [2.0e-3, 2.3e-3, 3.0e-3]
    patterns_train = ["ramp_down", "ramp_up", "step_sequence", "oscillating"]

    idx = 0
    for L in L_values_train:
        for pattern in patterns_train:
            base_R = 14.0
            profile = make_profile(base_R, pattern)
            task = {
                **BASE,
                "id": f"load_following_train_{idx:03d}",
                "scenario": "load_following",
                "filter_L": L,
                "load_R": profile[0],
                "max_steps": max_steps,
                "load_profile": [round(r, 3) for r in profile],
            }
            train.append(task)
            idx += 1

    L_values_test = [1.8e-3, 2.5e-3]
    patterns_test = ["ramp_down", "step_sequence", "oscillating"]

    idx = 0
    for L in L_values_test:
        for pattern in patterns_test:
            base_R = 14.0
            profile = make_profile(base_R, pattern)
            task = {
                **BASE,
                "id": f"load_following_test_{idx:03d}",
                "scenario": "load_following",
                "filter_L": L,
                "load_R": profile[0],
                "max_steps": max_steps,
                "load_profile": [round(r, 3) for r in profile],
            }
            test.append(task)
            idx += 1

    return train, test


# --------------------------------------------------------------------------- #
#  Build and cache all tasks
# --------------------------------------------------------------------------- #

def _build_all_tasks() -> dict[str, list[dict]]:
    """Generate all tasks, keyed by split."""
    tasks: dict[str, list[dict]] = {"train": [], "test": []}

    for gen_fn in [
        _voltage_control_tasks,
        _current_control_tasks,
        _droop_control_tasks,
        _load_following_tasks,
    ]:
        train, test = gen_fn()
        tasks["train"].extend(train)
        tasks["test"].extend(test)

    return tasks


ALL_TASKS = _build_all_tasks()

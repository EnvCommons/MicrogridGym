"""
Pure Python microgrid simulation engine.

Models three-phase inverter(s) with LC output filters, RL loads,
cascaded PI controllers (voltage → current), and droop control.
Based on the physics from the OpenModelica Microgrid Gym (OMG) toolbox.

Circuit (single inverter):

    V_inv ----[L, R_f]----+---- [R_load, L_load] ---- GND
                           |
                          [C]
                           |
                          GND

Control architecture (cascaded PI):
    V_ref → [Voltage PI] → i_ref → [Current PI] → duty → [Inverter]

Three-phase state equations (per phase k):
    L * di_Lk/dt = V_inv_k - V_Ck - R_f * i_Lk
    C * dV_Ck/dt = i_Lk - i_load_k
    L_load * di_load_k/dt = V_Ck - R_load * i_load_k
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Parameter data classes
# --------------------------------------------------------------------------- #

@dataclass
class LCFilterParams:
    """Three-phase LC filter parameters."""
    L: float = 2.3e-3     # Inductance per phase (H) - OMG default
    C: float = 10e-6      # Capacitance per phase (F) - OMG default
    R_f: float = 0.4      # Filter parasitic resistance (ohm) - OMG default


@dataclass
class RLLoadParams:
    """Three-phase RL load parameters."""
    R: float = 14.0       # Resistance per phase (ohm) - OMG default
    L: float = 5.3e-3     # Inductance per phase (H) - OMG default


@dataclass
class InverterParams:
    """Inverter parameters."""
    V_DC: float = 600.0        # DC link voltage (V) - OMG default
    V_nom: float = 325.27      # Peak nominal voltage, 230*sqrt(2) (V)
    f_nom: float = 50.0        # Nominal frequency (Hz)
    i_nom: float = 20.0        # Nominal current (A)
    i_lim: float = 30.0        # Current limit (A)


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    dt_ode: float = 1e-4        # ODE / PI update timestep (s), matches OMG
    dt_control: float = 5e-3    # Agent control interval (s)
    n_rk4_substeps: int = 4     # RK4 substeps per PI update
    phase_offsets: Tuple[float, float, float] = (
        0.0, -2 * np.pi / 3, 2 * np.pi / 3
    )


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #

def voltage_reference(t: float, V_nom: float, f: float,
                      phase_offsets: Tuple[float, float, float]) -> np.ndarray:
    """Generate 3-phase sinusoidal voltage reference."""
    omega = 2.0 * np.pi * f
    return np.array([V_nom * np.sin(omega * t + phi) for phi in phase_offsets])


def pi_controller(error: np.ndarray, integral: np.ndarray,
                  kP: float, kI: float, dt: float,
                  output_min: float = -1e6,
                  output_max: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
    """Three-phase PI controller with anti-windup clamping.

    Returns (output, updated_integral).
    """
    new_integral = integral + error * dt
    output = kP * error + kI * new_integral

    clamped = np.clip(output, output_min, output_max)
    # Anti-windup: if output was clamped, freeze integral
    for k in range(len(error)):
        if output[k] != clamped[k]:
            new_integral[k] = integral[k]

    return clamped, new_integral


def compute_power(v: np.ndarray, i: np.ndarray) -> Tuple[float, float]:
    """Compute instantaneous active and reactive power from 3-phase signals.

    P = sum(v*i) and Q via Clarke transform.
    """
    P = float(np.sum(v * i))
    v_alpha = v[0] - 0.5 * v[1] - 0.5 * v[2]
    v_beta = (np.sqrt(3) / 2.0) * (v[1] - v[2])
    i_alpha = i[0] - 0.5 * i[1] - 0.5 * i[2]
    i_beta = (np.sqrt(3) / 2.0) * (i[1] - i[2])
    Q = float(v_beta * i_alpha - v_alpha * i_beta)
    return P, Q


# --------------------------------------------------------------------------- #
#  Single-inverter simulation
# --------------------------------------------------------------------------- #

class MicrogridSimulation:
    """Single-inverter microgrid simulation with LC filter and RL load.

    Uses cascaded PI control matching OMG architecture:
      Outer loop: voltage PI → current reference
      Inner loop: current PI → duty cycle
    """

    def __init__(self, lc: LCFilterParams, load: RLLoadParams,
                 inv: InverterParams, config: SimulationConfig):
        self.lc = lc
        self.load = load
        self.inv = inv
        self.config = config

        # State: [i_L0, V_C0, i_load0, i_L1, V_C1, i_load1, i_L2, V_C2, i_load2]
        self.x = np.zeros(9)
        # Cascaded PI integrators
        self.pi_v_integral = np.zeros(3)   # Voltage loop
        self.pi_i_integral = np.zeros(3)   # Current loop
        self.t = 0.0
        self._step_count = 0

    def reset(self) -> dict:
        """Reset simulation to zero initial conditions."""
        self.x = np.zeros(9)
        self.pi_v_integral = np.zeros(3)
        self.pi_i_integral = np.zeros(3)
        self.t = 0.0
        self._step_count = 0
        return self._observe()

    def step(self, kP_v: float, kI_v: float,
             kP_i: float = 0.012, kI_i: float = 90.0) -> dict:
        """Advance simulation by one agent control interval.

        Args:
            kP_v: Voltage loop proportional gain (OMG default: 0.025)
            kI_v: Voltage loop integral gain (OMG default: 60)
            kP_i: Current loop proportional gain (OMG default: 0.012)
            kI_i: Current loop integral gain (OMG default: 90)

        Uses a two-level loop:
        - PI controllers update at dt_ode rate
        - RK4 integrates ODE between PI updates
        """
        n_pi = max(1, int(round(self.config.dt_control / self.config.dt_ode)))
        dt = self.config.dt_ode
        dt_rk4 = dt / self.config.n_rk4_substeps

        v_errors = []
        i_peaks = []
        # Per-phase tracking data for OMG-style reward
        v_abs_errors_per_phase = []  # |V_ref - V_C| per phase per substep
        i_abs_per_phase = []         # |i_L| per phase per substep
        i_ref_per_phase = []         # |i_ref| per phase per substep
        i_actual_per_phase = []      # |i_L| per phase per substep

        for _ in range(n_pi):
            # Voltage reference
            V_ref = voltage_reference(
                self.t, self.inv.V_nom, self.inv.f_nom,
                self.config.phase_offsets
            )

            # Read state
            V_C = np.array([self.x[3 * k + 1] for k in range(3)])
            i_L = np.array([self.x[3 * k] for k in range(3)])

            # Outer loop: voltage PI → current reference
            v_error = V_ref - V_C
            i_ref, self.pi_v_integral = pi_controller(
                v_error, self.pi_v_integral, kP_v, kI_v, dt,
                output_min=-self.inv.i_lim, output_max=self.inv.i_lim
            )

            # Inner loop: current PI → duty cycle
            i_error = i_ref - i_L
            duty, self.pi_i_integral = pi_controller(
                i_error, self.pi_i_integral, kP_i, kI_i, dt,
                output_min=-1.0, output_max=1.0
            )

            # RK4 integration with constant duty
            for _ in range(self.config.n_rk4_substeps):
                self.x = self._rk4_step(self.x, duty, dt_rk4)
            self.t += dt

            # Metrics
            v_errors.append(float(np.sqrt(np.mean(v_error ** 2))))
            i_peaks.append(float(np.max(np.abs(
                [self.x[3 * k] for k in range(3)]
            ))))
            # Per-phase data for OMG reward
            v_abs_errors_per_phase.append(np.abs(v_error).tolist())
            i_abs_per_phase.append(np.abs(i_L).tolist())
            i_ref_per_phase.append(i_ref.tolist())
            i_actual_per_phase.append(i_L.tolist())

        self._step_count += 1
        obs = self._observe(v_errors, i_peaks)
        # Attach raw per-phase data for reward computation
        obs["_v_abs_errors_per_phase"] = v_abs_errors_per_phase
        obs["_i_abs_per_phase"] = i_abs_per_phase
        obs["_i_ref_per_phase"] = i_ref_per_phase
        obs["_i_actual_per_phase"] = i_actual_per_phase
        return obs

    def _rk4_step(self, x: np.ndarray, duty: np.ndarray,
                  dt: float) -> np.ndarray:
        """Single RK4 integration step with constant duty."""
        k1 = self._rhs(x, duty)
        k2 = self._rhs(x + 0.5 * dt * k1, duty)
        k3 = self._rhs(x + 0.5 * dt * k2, duty)
        k4 = self._rhs(x + dt * k3, duty)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rhs(self, x: np.ndarray, duty: np.ndarray) -> np.ndarray:
        """ODE right-hand side for single inverter + LC filter + RL load."""
        dx = np.zeros(9)
        for k in range(3):
            i_L = x[3 * k]
            V_C = x[3 * k + 1]
            i_load = x[3 * k + 2]

            V_inv = self.inv.V_DC * np.clip(duty[k], -1.0, 1.0)

            dx[3 * k] = (V_inv - V_C - self.lc.R_f * i_L) / self.lc.L
            dx[3 * k + 1] = (i_L - i_load) / self.lc.C
            dx[3 * k + 2] = (V_C - self.load.R * i_load) / self.load.L

        return dx

    def check_stability(self) -> bool:
        """Check if simulation state has diverged."""
        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            return False
        for k in range(3):
            if abs(self.x[3 * k + 1]) > 2 * self.inv.V_DC:
                return False
            if abs(self.x[3 * k]) > 10 * self.inv.i_lim:
                return False
        return True

    def _observe(self, v_errors=None, i_peaks=None) -> dict:
        """Build observation dictionary."""
        V_C = [self.x[3 * k + 1] for k in range(3)]
        i_L = [self.x[3 * k] for k in range(3)]
        i_load = [self.x[3 * k + 2] for k in range(3)]

        V_ref = voltage_reference(
            self.t, self.inv.V_nom, self.inv.f_nom,
            self.config.phase_offsets
        )

        # Compute active/reactive power
        P, Q = compute_power(np.array(V_C), np.array(i_load))

        obs = {
            "time": round(self.t, 6),
            "step": self._step_count,
            "capacitor_voltages_V": [round(v, 2) for v in V_C],
            "inductor_currents_A": [round(i, 3) for i in i_L],
            "load_currents_A": [round(i, 3) for i in i_load],
            "voltage_reference_V": [round(v, 2) for v in V_ref.tolist()],
            "voltage_rms_V": round(float(np.sqrt(np.mean(np.array(V_C) ** 2))), 2),
            "current_peak_A": round(float(max(abs(i) for i in i_L)), 3),
            "active_power_W": round(P, 2),
            "reactive_power_VAR": round(Q, 2),
        }

        if v_errors is not None:
            obs["mean_voltage_error_V"] = round(float(np.mean(v_errors)), 4)
            obs["max_voltage_error_V"] = round(float(np.max(v_errors)), 4)
        if i_peaks is not None:
            obs["max_inductor_current_A"] = round(float(np.max(i_peaks)), 3)
            obs["current_limit_A"] = self.inv.i_lim

        return obs


# --------------------------------------------------------------------------- #
#  Two-inverter droop simulation
# --------------------------------------------------------------------------- #

@dataclass
class DroopState:
    """State for droop-controlled inverter."""
    theta: float = 0.0
    P_filtered: float = 0.0
    Q_filtered: float = 0.0


class TwoInverterSimulation:
    """Two parallel inverters with droop control sharing a common RL load.

    Each inverter has its own LC filter and cascaded PI controller.
    Both use droop equations for frequency and voltage amplitude.

    State vector: 18 elements (9 per inverter)
    Per inverter per phase: [i_L, V_C, i_out]
    """

    def __init__(self, lc1: LCFilterParams, lc2: LCFilterParams,
                 load: RLLoadParams,
                 inv1: InverterParams, inv2: InverterParams,
                 config: SimulationConfig):
        self.lc1 = lc1
        self.lc2 = lc2
        self.load = load
        self.inv1 = inv1
        self.inv2 = inv2
        self.config = config

        self.x = np.zeros(18)
        # Cascaded PI for each inverter
        self.pi_v_integral_1 = np.zeros(3)
        self.pi_i_integral_1 = np.zeros(3)
        self.pi_v_integral_2 = np.zeros(3)
        self.pi_i_integral_2 = np.zeros(3)
        self.droop1 = DroopState()
        self.droop2 = DroopState()
        self.t = 0.0
        self._step_count = 0

        self.tau_P = 0.01   # Power low-pass filter time constant
        self.tau_Q = 0.01

    def reset(self) -> dict:
        """Reset to zero initial conditions."""
        self.x = np.zeros(18)
        self.pi_v_integral_1 = np.zeros(3)
        self.pi_i_integral_1 = np.zeros(3)
        self.pi_v_integral_2 = np.zeros(3)
        self.pi_i_integral_2 = np.zeros(3)
        self.droop1 = DroopState()
        self.droop2 = DroopState()
        self.t = 0.0
        self._step_count = 0
        return self._observe()

    def step(self, kP_droop_1: float, kQ_droop_1: float,
             kP_droop_2: float, kQ_droop_2: float,
             kP_v: float = 0.025, kI_v: float = 60.0,
             kP_i: float = 0.012, kI_i: float = 90.0) -> dict:
        """Advance one control interval with droop parameters."""
        n_pi = max(1, int(round(self.config.dt_control / self.config.dt_ode)))
        dt = self.config.dt_ode
        dt_rk4 = dt / self.config.n_rk4_substeps

        v_errors = []
        i_peaks = []
        # Per-phase tracking data for OMG-style reward
        v_abs_errors_per_phase = []
        i_abs_per_phase = []
        i_ref_per_phase = []
        i_actual_per_phase = []

        for _ in range(n_pi):
            # Measure voltages and currents
            V_C1 = np.array([self.x[3 * k + 1] for k in range(3)])
            i_L1 = np.array([self.x[3 * k] for k in range(3)])
            i_out1 = np.array([self.x[3 * k + 2] for k in range(3)])
            V_C2 = np.array([self.x[9 + 3 * k + 1] for k in range(3)])
            i_L2 = np.array([self.x[9 + 3 * k] for k in range(3)])
            i_out2 = np.array([self.x[9 + 3 * k + 2] for k in range(3)])

            P1, Q1 = compute_power(V_C1, i_out1)
            P2, Q2 = compute_power(V_C2, i_out2)

            # Droop control
            f1, V1_amp, self.droop1 = self._droop_step(
                P1, Q1, self.droop1,
                kP_droop_1, kQ_droop_1,
                self.inv1.f_nom, self.inv1.V_nom, dt
            )
            f2, V2_amp, self.droop2 = self._droop_step(
                P2, Q2, self.droop2,
                kP_droop_2, kQ_droop_2,
                self.inv2.f_nom, self.inv2.V_nom, dt
            )

            # Voltage references from droop
            V_ref1 = np.array([
                V1_amp * np.sin(self.droop1.theta + phi)
                for phi in self.config.phase_offsets
            ])
            V_ref2 = np.array([
                V2_amp * np.sin(self.droop2.theta + phi)
                for phi in self.config.phase_offsets
            ])

            # Cascaded PI - Inverter 1
            v_err1 = V_ref1 - V_C1
            i_ref1, self.pi_v_integral_1 = pi_controller(
                v_err1, self.pi_v_integral_1, kP_v, kI_v, dt,
                output_min=-self.inv1.i_lim, output_max=self.inv1.i_lim
            )
            i_err1 = i_ref1 - i_L1
            duty1, self.pi_i_integral_1 = pi_controller(
                i_err1, self.pi_i_integral_1, kP_i, kI_i, dt,
                output_min=-1.0, output_max=1.0
            )

            # Cascaded PI - Inverter 2
            v_err2 = V_ref2 - V_C2
            i_ref2, self.pi_v_integral_2 = pi_controller(
                v_err2, self.pi_v_integral_2, kP_v, kI_v, dt,
                output_min=-self.inv2.i_lim, output_max=self.inv2.i_lim
            )
            i_err2 = i_ref2 - i_L2
            duty2, self.pi_i_integral_2 = pi_controller(
                i_err2, self.pi_i_integral_2, kP_i, kI_i, dt,
                output_min=-1.0, output_max=1.0
            )

            # RK4 integration
            for _ in range(self.config.n_rk4_substeps):
                k1 = self._rhs(self.x, duty1, duty2)
                k2 = self._rhs(self.x + 0.5 * dt_rk4 * k1, duty1, duty2)
                k3 = self._rhs(self.x + 0.5 * dt_rk4 * k2, duty1, duty2)
                k4 = self._rhs(self.x + dt_rk4 * k3, duty1, duty2)
                self.x += (dt_rk4 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.t += dt

            # Error tracking (against nominal reference for consistency)
            V_ref_nom = voltage_reference(
                self.t, self.inv1.V_nom, self.inv1.f_nom,
                self.config.phase_offsets
            )
            V_bus = (V_C1 + V_C2) / 2.0
            v_errors.append(float(np.sqrt(np.mean((V_ref_nom - V_bus) ** 2))))
            i_all = [abs(self.x[j]) for j in
                     [3 * k for k in range(3)] +
                     [9 + 3 * k for k in range(3)]]
            i_peaks.append(max(i_all))

            # Per-phase data (average across both inverters for bus voltage)
            v_abs_errors_per_phase.append(np.abs(V_ref_nom - V_bus).tolist())
            # Use max current across both inverters per phase
            i_both = np.maximum(np.abs(i_L1), np.abs(i_L2))
            i_abs_per_phase.append(i_both.tolist())
            i_ref_per_phase.append(((i_ref1 + i_ref2) / 2.0).tolist())
            i_actual_per_phase.append(((i_L1 + i_L2) / 2.0).tolist())

        self._step_count += 1
        obs = self._observe(v_errors, i_peaks)
        # Attach raw per-phase data for reward computation
        obs["_v_abs_errors_per_phase"] = v_abs_errors_per_phase
        obs["_i_abs_per_phase"] = i_abs_per_phase
        obs["_i_ref_per_phase"] = i_ref_per_phase
        obs["_i_actual_per_phase"] = i_actual_per_phase
        return obs

    def _droop_step(self, P: float, Q: float, state: DroopState,
                    kP_droop: float, kQ_droop: float,
                    f_nom: float, V_nom: float,
                    dt: float) -> Tuple[float, float, DroopState]:
        """One step of droop control with power low-pass filtering."""
        alpha_P = dt / (self.tau_P + dt)
        alpha_Q = dt / (self.tau_Q + dt)

        P_filt = state.P_filtered + alpha_P * (P - state.P_filtered)
        Q_filt = state.Q_filtered + alpha_Q * (Q - state.Q_filtered)

        f = f_nom - kP_droop * P_filt
        V = V_nom - kQ_droop * Q_filt

        V = max(0.5 * V_nom, min(1.5 * V_nom, V))
        f = max(0.8 * f_nom, min(1.2 * f_nom, f))

        theta = state.theta + 2.0 * np.pi * f * dt
        new_state = DroopState(theta=theta, P_filtered=P_filt, Q_filtered=Q_filt)
        return f, V, new_state

    def _rhs(self, x: np.ndarray, duty1: np.ndarray,
             duty2: np.ndarray) -> np.ndarray:
        """ODE RHS for two inverters sharing a common RL load."""
        dx = np.zeros(18)

        for k in range(3):
            i_L1 = x[3 * k]
            V_C1 = x[3 * k + 1]
            i_out1 = x[3 * k + 2]

            i_L2 = x[9 + 3 * k]
            V_C2 = x[9 + 3 * k + 1]
            i_out2 = x[9 + 3 * k + 2]

            V_inv1 = self.inv1.V_DC * np.clip(duty1[k], -1.0, 1.0)
            V_inv2 = self.inv2.V_DC * np.clip(duty2[k], -1.0, 1.0)

            # Inverter 1 dynamics
            dx[3 * k] = (V_inv1 - V_C1 - self.lc1.R_f * i_L1) / self.lc1.L
            dx[3 * k + 1] = (i_L1 - i_out1) / self.lc1.C
            dx[3 * k + 2] = (V_C1 - self.load.R * i_out1) / self.load.L

            # Inverter 2 dynamics
            dx[9 + 3 * k] = (V_inv2 - V_C2 - self.lc2.R_f * i_L2) / self.lc2.L
            dx[9 + 3 * k + 1] = (i_L2 - i_out2) / self.lc2.C
            dx[9 + 3 * k + 2] = (V_C2 - self.load.R * i_out2) / self.load.L

        return dx

    def check_stability(self) -> bool:
        """Check if simulation state has diverged."""
        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            return False
        for inv_offset in [0, 9]:
            for k in range(3):
                if abs(self.x[inv_offset + 3 * k + 1]) > 2 * self.inv1.V_DC:
                    return False
                if abs(self.x[inv_offset + 3 * k]) > 10 * self.inv1.i_lim:
                    return False
        return True

    def _observe(self, v_errors=None, i_peaks=None) -> dict:
        """Build observation dictionary for two-inverter system."""
        V_C1 = [self.x[3 * k + 1] for k in range(3)]
        i_L1 = [self.x[3 * k] for k in range(3)]
        V_C2 = [self.x[9 + 3 * k + 1] for k in range(3)]
        i_L2 = [self.x[9 + 3 * k] for k in range(3)]

        V_ref1 = voltage_reference(
            self.t, self.inv1.V_nom, self.inv1.f_nom,
            self.config.phase_offsets
        )

        i_out1 = np.array([self.x[3 * k + 2] for k in range(3)])
        i_out2 = np.array([self.x[9 + 3 * k + 2] for k in range(3)])
        P1, Q1 = compute_power(np.array(V_C1), i_out1)
        P2, Q2 = compute_power(np.array(V_C2), i_out2)

        obs = {
            "time": round(self.t, 6),
            "step": self._step_count,
            "inv1_capacitor_voltages_V": [round(v, 2) for v in V_C1],
            "inv1_inductor_currents_A": [round(i, 3) for i in i_L1],
            "inv2_capacitor_voltages_V": [round(v, 2) for v in V_C2],
            "inv2_inductor_currents_A": [round(i, 3) for i in i_L2],
            "voltage_reference_V": [round(v, 2) for v in V_ref1.tolist()],
            "inv1_voltage_rms_V": round(
                float(np.sqrt(np.mean(np.array(V_C1) ** 2))), 2),
            "inv2_voltage_rms_V": round(
                float(np.sqrt(np.mean(np.array(V_C2) ** 2))), 2),
            "inv1_active_power_W": round(P1, 2),
            "inv2_active_power_W": round(P2, 2),
            "inv1_reactive_power_VAR": round(Q1, 2),
            "inv2_reactive_power_VAR": round(Q2, 2),
            "power_sharing_ratio": (
                round(P1 / max(abs(P2), 1e-6), 3) if abs(P2) > 1e-6
                else 0.0
            ),
        }

        if v_errors is not None:
            obs["mean_voltage_error_V"] = round(float(np.mean(v_errors)), 4)
            obs["max_voltage_error_V"] = round(float(np.max(v_errors)), 4)
        if i_peaks is not None:
            obs["max_inductor_current_A"] = round(float(np.max(i_peaks)), 3)
            obs["current_limit_A"] = self.inv1.i_lim

        return obs

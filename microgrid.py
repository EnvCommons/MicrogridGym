"""
MicrogridGym: OpenReward environment for microgrid inverter control.

Based on the OpenModelica Microgrid Gym (OMG) toolbox. The agent tunes
cascaded PI controller gains and/or droop coefficients for three-phase
inverter(s) with LC filters supplying RL loads.
"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from openreward.environments import (
    Environment,
    JSONObject,
    TextBlock,
    ToolOutput,
    tool,
)

from simulation import (
    InverterParams,
    LCFilterParams,
    MicrogridSimulation,
    RLLoadParams,
    SimulationConfig,
    TwoInverterSimulation,
)
from tasks import ALL_TASKS, TaskSpec


# --------------------------------------------------------------------------- #
#  Pydantic models for tool parameters
# --------------------------------------------------------------------------- #

class SetControllerParams(BaseModel, extra="forbid"):
    """Parameters for set_controller tool.

    For voltage_control / current_control / load_following scenarios:
        Required: kP_v, kI_v
        Optional: kP_i, kI_i (defaults to OMG current loop defaults)

    For droop_control scenario:
        Required: kP_droop_1, kQ_droop_1, kP_droop_2, kQ_droop_2
        Optional: kP_v, kI_v, kP_i, kI_i (inner loop gains)
    """
    kP_v: Optional[float] = Field(
        None,
        description="Voltage loop proportional gain. Typical range: 0.01-0.2. "
                    "OMG default: 0.025"
    )
    kI_v: Optional[float] = Field(
        None,
        description="Voltage loop integral gain. Typical range: 20-500. "
                    "OMG default: 60"
    )
    kP_i: Optional[float] = Field(
        None,
        description="Current loop proportional gain. Typical range: 0.005-0.05. "
                    "OMG default: 0.012"
    )
    kI_i: Optional[float] = Field(
        None,
        description="Current loop integral gain. Typical range: 30-300. "
                    "OMG default: 90"
    )
    kP_droop_1: Optional[float] = Field(
        None,
        description="Active power droop coefficient for inverter 1 (Hz/W). "
                    "Typical range: 1e-6 to 1e-3"
    )
    kQ_droop_1: Optional[float] = Field(
        None,
        description="Reactive power droop coefficient for inverter 1 (V/VAR). "
                    "Typical range: 1e-4 to 1e-1"
    )
    kP_droop_2: Optional[float] = Field(
        None,
        description="Active power droop coefficient for inverter 2 (Hz/W). "
                    "Typical range: 1e-6 to 1e-3"
    )
    kQ_droop_2: Optional[float] = Field(
        None,
        description="Reactive power droop coefficient for inverter 2 (V/VAR). "
                    "Typical range: 1e-4 to 1e-1"
    )


class InfoParams(BaseModel, extra="forbid"):
    """No parameters needed for the info tool."""
    pass


# --------------------------------------------------------------------------- #
#  Environment class
# --------------------------------------------------------------------------- #

class MicrogridGym(Environment):
    """Microgrid inverter control environment.

    The agent tunes cascaded PI controller gains for three-phase
    power electronic inverters. At each step the agent sets controller
    parameters, the simulation advances one control interval, and
    the agent receives updated measurements and a reward based on
    voltage tracking accuracy.
    """

    def __init__(self, task_spec: JSONObject,
                 secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.sim = None
        self.current_obs: dict = {}
        self.step_rewards: list[float] = []
        self.episode_done = False

    async def setup(self):
        """Initialize simulation from task spec."""
        lc = LCFilterParams(
            L=self.config.filter_L,
            C=self.config.filter_C,
            R_f=self.config.filter_Rf,
        )
        load = RLLoadParams(
            R=self.config.load_R,
            L=self.config.load_L,
        )
        inv = InverterParams(
            V_DC=self.config.inverter_V_DC,
            V_nom=self.config.inverter_V_nom,
            f_nom=self.config.inverter_f_nom,
            i_nom=self.config.inverter_i_nom,
            i_lim=self.config.inverter_i_lim,
        )
        sim_config = SimulationConfig(
            dt_ode=self.config.dt_ode,
            dt_control=self.config.dt_control,
        )

        if self.config.scenario == "droop_control":
            lc2 = LCFilterParams(
                L=self.config.filter_L_2 or self.config.filter_L,
                C=self.config.filter_C_2 or self.config.filter_C,
                R_f=self.config.filter_Rf_2 or self.config.filter_Rf,
            )
            inv2 = InverterParams(
                V_DC=self.config.inverter_V_DC_2 or self.config.inverter_V_DC,
                V_nom=self.config.inverter_V_nom,
                f_nom=self.config.inverter_f_nom,
                i_lim=self.config.inverter_i_lim_2 or self.config.inverter_i_lim,
            )
            self.sim = TwoInverterSimulation(lc, lc2, load, inv, inv2, sim_config)
        else:
            self.sim = MicrogridSimulation(lc, load, inv, sim_config)

        self.current_obs = self.sim.reset()
        self.step_rewards = []
        self.episode_done = False

    async def teardown(self):
        self.sim = None

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ALL_TASKS:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'.")
        return ALL_TASKS[split]

    # ------------------------------------------------------------------ #
    #  Prompt
    # ------------------------------------------------------------------ #

    async def get_prompt(self) -> List[TextBlock]:
        """Return scenario-specific prompt with initial observations."""
        c = self.config

        scenario_desc = {
            "voltage_control": (
                "VOLTAGE CONTROL: You are tuning a cascaded PI controller for "
                "a three-phase inverter with an LC output filter supplying an "
                "RL load. The outer voltage loop generates a current reference, "
                "and the inner current loop generates the PWM duty cycle. Your "
                "goal is to maintain the capacitor voltage as close as possible "
                "to the sinusoidal reference. A load step disturbance will occur "
                f"at step {c.disturbance_step} (load resistance halves)."
            ),
            "current_control": (
                "CURRENT CONTROL: You are tuning a cascaded PI controller for "
                "a three-phase inverter. The focus is on accurate current "
                "tracking through the filter inductor. A load step disturbance "
                f"will occur at step {c.disturbance_step}."
            ),
            "droop_control": (
                "DROOP CONTROL: You are configuring droop controllers for two "
                "parallel inverters sharing a common RL load. Each inverter has "
                "its own LC filter and cascaded PI controller. Your goal is to "
                "achieve proportional active and reactive power sharing while "
                "maintaining voltage quality. A load step disturbance will occur "
                f"at step {c.disturbance_step}."
            ),
            "load_following": (
                "LOAD FOLLOWING: You are tuning a cascaded PI voltage controller "
                "for an inverter supplying a time-varying load. The load "
                "resistance changes at each step according to a predefined "
                "profile. You may re-tune your PI gains at every step to "
                "maintain voltage quality."
            ),
        }

        param_desc = {
            "voltage_control": (
                "PARAMETERS TO SET (via set_controller tool):\n"
                "  Required:\n"
                "    kP_v  - Voltage loop proportional gain (typical: 0.01-0.2)\n"
                "    kI_v  - Voltage loop integral gain (typical: 20-500)\n"
                "  Optional:\n"
                "    kP_i  - Current loop proportional gain (default: 0.012)\n"
                "    kI_i  - Current loop integral gain (default: 90)"
            ),
            "current_control": (
                "PARAMETERS TO SET (via set_controller tool):\n"
                "  Required:\n"
                "    kP_v  - Voltage loop proportional gain (typical: 0.01-0.2)\n"
                "    kI_v  - Voltage loop integral gain (typical: 20-500)\n"
                "  Optional:\n"
                "    kP_i  - Current loop proportional gain (default: 0.012)\n"
                "    kI_i  - Current loop integral gain (default: 90)"
            ),
            "droop_control": (
                "PARAMETERS TO SET (via set_controller tool):\n"
                "  Required:\n"
                "    kP_droop_1  - Active power droop, inverter 1 (Hz/W)\n"
                "    kQ_droop_1  - Reactive power droop, inverter 1 (V/VAR)\n"
                "    kP_droop_2  - Active power droop, inverter 2 (Hz/W)\n"
                "    kQ_droop_2  - Reactive power droop, inverter 2 (V/VAR)\n"
                "  Optional:\n"
                "    kP_v, kI_v  - Inner voltage loop PI gains\n"
                "    kP_i, kI_i  - Inner current loop PI gains"
            ),
            "load_following": (
                "PARAMETERS TO SET (via set_controller tool):\n"
                "  Required:\n"
                "    kP_v  - Voltage loop proportional gain (typical: 0.01-0.2)\n"
                "    kI_v  - Voltage loop integral gain (typical: 20-500)\n"
                "  Optional:\n"
                "    kP_i  - Current loop proportional gain (default: 0.012)\n"
                "    kI_i  - Current loop integral gain (default: 90)"
            ),
        }

        circuit_info = (
            f"CIRCUIT PARAMETERS:\n"
            f"  Filter inductance (L):   {c.filter_L * 1e3:.2f} mH\n"
            f"  Filter capacitance (C):  {c.filter_C * 1e6:.1f} uF\n"
            f"  Filter resistance (R_f): {c.filter_Rf:.2f} ohm\n"
            f"  Load resistance (R):     {c.load_R:.1f} ohm\n"
            f"  Load inductance (L):     {c.load_L * 1e3:.2f} mH\n"
            f"  DC bus voltage:          {c.inverter_V_DC:.0f} V\n"
            f"  Nominal voltage:         {c.inverter_V_nom:.1f} V peak "
            f"({c.inverter_V_nom / np.sqrt(2):.0f} V RMS)\n"
            f"  Nominal frequency:       {c.inverter_f_nom:.0f} Hz\n"
            f"  Current limit:           {c.inverter_i_lim:.0f} A"
        )

        if c.scenario == "droop_control":
            circuit_info += (
                f"\n\n  Inverter 2:\n"
                f"    Filter L:  {(c.filter_L_2 or c.filter_L) * 1e3:.2f} mH\n"
                f"    Filter C:  {(c.filter_C_2 or c.filter_C) * 1e6:.1f} uF\n"
                f"    Filter Rf: {(c.filter_Rf_2 or c.filter_Rf):.2f} ohm"
            )

        load_info = ""
        if c.scenario == "load_following" and c.load_profile:
            load_info = (
                "\n\nLOAD PROFILE (R in ohm per step):\n  " +
                ", ".join(f"{r:.1f}" for r in c.load_profile)
            )

        obs_text = self._format_observation(self.current_obs, None, False, 0.0)

        prompt = f"""You are a power electronics control engineer tuning controller parameters for a microgrid inverter system.

{scenario_desc[c.scenario]}

{circuit_info}{load_info}

CONTROL ARCHITECTURE (cascaded PI):
  V_ref --> [Voltage PI] --> i_ref --> [Current PI] --> duty --> [Inverter+LC] --> V_out
  Outer voltage loop: error = V_ref - V_capacitor -> current reference
  Inner current loop: error = i_ref - i_inductor -> PWM duty cycle

{param_desc[c.scenario]}

EPISODE STRUCTURE:
  Total steps: {c.max_steps}
  Each step advances the simulation by {c.dt_control * 1e3:.1f} ms
  Use the set_controller tool to set parameters and advance one step
  Use the info tool for detailed reference documentation

REWARD (based on OMG reward structure):
  Per-step reward based on voltage+current tracking and current safety:
  voltage_err = mean over time of sum_phases( sqrt(|V_ref - V_actual| / V_nom) )
  current_err = mean over time of sum_phases( sqrt(|i_ref - i_actual| / i_lim) )
  barrier = log-barrier penalty when |i| exceeds {c.inverter_i_nom:.0f} A (nom), hard limit {c.inverter_i_lim:.0f} A
  step_reward = exp(-(voltage_err + current_err + barrier) / 3)
  Episode reward = mean of all step rewards, in [0, 1]
  Goal: maximize episode reward (closer to 1.0 is better)

STRATEGY HINTS:
  - Start with moderate gains and observe the transient response
  - Higher kI_v improves steady-state tracking but may cause oscillation
  - Higher kP_v improves transient response but may cause overshoot
  - The current loop should be faster than the voltage loop
  - After a load disturbance, monitor how the controller recovers

CURRENT OBSERVATION:
{obs_text}

Begin by setting your controller parameters using set_controller."""

        return [TextBlock(text=prompt)]

    # ------------------------------------------------------------------ #
    #  Tools
    # ------------------------------------------------------------------ #

    @tool
    async def set_controller(self, params: SetControllerParams) -> ToolOutput:
        """Set controller parameters and advance the simulation by one
        control interval. Returns updated measurements and reward."""

        if self.episode_done:
            return ToolOutput(
                blocks=[TextBlock(text="Episode already finished.")],
                metadata={"error": "episode_finished"},
                reward=0.0,
                finished=True,
            )

        scenario = self.config.scenario

        # Validate required parameters
        if scenario in ("voltage_control", "current_control", "load_following"):
            if params.kP_v is None or params.kI_v is None:
                return ToolOutput(
                    blocks=[TextBlock(
                        text="Error: kP_v and kI_v are required for this "
                             "scenario. Please provide both parameters."
                    )],
                    metadata={"error": "missing_parameters"},
                    reward=0.0,
                    finished=False,
                )
            # Validate ranges
            if params.kP_v < 0 or params.kI_v < 0:
                return ToolOutput(
                    blocks=[TextBlock(text="Error: gains must be non-negative.")],
                    metadata={"error": "invalid_parameters"},
                    reward=0.0,
                    finished=False,
                )
        elif scenario == "droop_control":
            droop_params = [
                params.kP_droop_1, params.kQ_droop_1,
                params.kP_droop_2, params.kQ_droop_2,
            ]
            if any(v is None for v in droop_params):
                return ToolOutput(
                    blocks=[TextBlock(
                        text="Error: all droop parameters required "
                             "(kP_droop_1, kQ_droop_1, kP_droop_2, kQ_droop_2)."
                    )],
                    metadata={"error": "missing_parameters"},
                    reward=0.0,
                    finished=False,
                )

        # Apply load disturbance if scheduled
        current_step = self.sim._step_count
        if (self.config.disturbance_step is not None
                and current_step == self.config.disturbance_step):
            if self.config.load_R_after is not None:
                self.sim.load.R = self.config.load_R_after
            if self.config.load_L_after is not None:
                self.sim.load.L = self.config.load_L_after

        # Apply load profile for load_following scenario
        if (self.config.scenario == "load_following"
                and self.config.load_profile is not None
                and current_step < len(self.config.load_profile)):
            self.sim.load.R = self.config.load_profile[current_step]

        # Step simulation
        kP_i = params.kP_i if params.kP_i is not None else 0.012
        kI_i = params.kI_i if params.kI_i is not None else 90.0

        if scenario == "droop_control":
            kP_v = params.kP_v if params.kP_v is not None else 0.025
            kI_v = params.kI_v if params.kI_v is not None else 60.0
            obs = self.sim.step(
                kP_droop_1=params.kP_droop_1,
                kQ_droop_1=params.kQ_droop_1,
                kP_droop_2=params.kP_droop_2,
                kQ_droop_2=params.kQ_droop_2,
                kP_v=kP_v, kI_v=kI_v,
                kP_i=kP_i, kI_i=kI_i,
            )
        else:
            obs = self.sim.step(
                kP_v=params.kP_v,
                kI_v=params.kI_v,
                kP_i=kP_i,
                kI_i=kI_i,
            )

        self.current_obs = obs

        # Check stability
        if not self.sim.check_stability():
            self.episode_done = True
            self.step_rewards.append(0.0)
            episode_reward = sum(self.step_rewards) / len(self.step_rewards)

            return ToolOutput(
                blocks=[TextBlock(text=(
                    "SIMULATION UNSTABLE: The controller parameters caused "
                    "numerical divergence. Episode terminated.\n\n"
                    f"Step: {obs['step']}/{self.config.max_steps}\n"
                    f"Episode reward: {episode_reward:.4f}"
                ))],
                metadata={
                    "error": "instability",
                    "obs": obs,
                    "episode_reward": episode_reward,
                },
                reward=episode_reward,
                finished=True,
            )

        # Compute reward
        step_reward = self._compute_reward(obs)
        self.step_rewards.append(step_reward)

        # Check episode completion
        finished = obs["step"] >= self.config.max_steps
        if finished:
            self.episode_done = True

        episode_reward = sum(self.step_rewards) / len(self.step_rewards)
        obs_text = self._format_observation(
            obs, step_reward, finished, episode_reward
        )

        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={
                "obs": obs,
                "step_reward": step_reward,
                "episode_reward": episode_reward if finished else None,
                "step": obs["step"],
                "max_steps": self.config.max_steps,
            },
            reward=episode_reward if finished else step_reward,
            finished=finished,
        )

    @tool
    async def info(self, params: InfoParams) -> ToolOutput:
        """Show reference documentation about the microgrid system,
        controller architecture, and parameter guidelines."""

        reference = """MICROGRID CONTROL REFERENCE

CIRCUIT TOPOLOGY:
  V_inv ----[L, R_f]----+---- [R_load, L_load] ---- GND
                         |
                        [C]
                         |
                        GND

  V_inv = V_DC * duty  (inverter output, controlled by PWM)

CASCADED PI CONTROL:
  The controller has two loops in cascade:

  1. VOLTAGE LOOP (outer, slower):
     error_v = V_ref - V_capacitor
     i_ref = kP_v * error_v + kI_v * integral(error_v)
     Output is clamped to [-i_lim, +i_lim]

  2. CURRENT LOOP (inner, faster):
     error_i = i_ref - i_inductor
     duty = kP_i * error_i + kI_i * integral(error_i)
     Output is clamped to [-1, +1]

  Both loops include anti-windup (integral freezes when output saturates).

DROOP CONTROL (two-inverter scenarios):
  f = f_nom - kP_droop * P_filtered    (frequency droop)
  V = V_nom - kQ_droop * Q_filtered    (voltage droop)

  Power measurements are low-pass filtered (tau = 10ms).
  Equal droop gains -> equal power sharing.
  Larger droop gain -> less power share for that inverter.

TUNING GUIDELINES:

  Voltage loop (kP_v, kI_v):
    - Higher kP_v: faster transient response, but may oscillate
    - Higher kI_v: eliminates steady-state error, but slow response
    - Rule of thumb: kI_v / kP_v ~ 1000-5000 (integral time constant)

  Current loop (kP_i, kI_i):
    - Should be significantly faster than voltage loop
    - If omitted, the OMG toolbox defaults are used; they are not tuned
      for any particular circuit
    - Increase kP_i if current tracking is sluggish
    - Decrease kP_i if you see high-frequency oscillation

  Droop coefficients:
    - kP_droop: 1e-5 to 1e-3 Hz/W typical
    - kQ_droop: 1e-3 to 1e-1 V/VAR typical
    - Larger values give more frequency/voltage droop (less stiff)
    - Equal values on both inverters -> 50/50 power sharing

REWARD STRUCTURE (based on OMG):
  The reward follows the OpenModelica Microgrid Gym formulation:

  1. Voltage tracking error (per phase, root-error):
     voltage_err = sum_phases( sqrt(|V_ref - V_actual| / V_nom) )

  2. Current tracking error (per phase, root-error):
     current_err = sum_phases( sqrt(|i_ref - i_actual| / i_lim) )

  3. Log-barrier current constraint penalty:
     barrier = -sum_phases( mu * ln(1 - max(|i| - i_nom, 0) / (i_lim - i_nom)) )
     mu = 2, i_nom = 20 A, i_lim = 30 A

  Total OMG error = voltage_err + current_err + barrier
  step_reward = exp(-total_error / 3)   (mapped to [0, 1])
  episode_reward = mean(step_rewards)

  Perfect tracking, low current -> reward ≈ 1.0
  Moderate voltage error -> reward ≈ 0.5-0.7
  Large errors or current violations -> reward near 0"""

        return ToolOutput(
            blocks=[TextBlock(text=reference)],
            metadata={},
            reward=0.0,
            finished=False,
        )

    # ------------------------------------------------------------------ #
    #  Reward and observation formatting
    # ------------------------------------------------------------------ #

    def _compute_reward(self, obs: dict) -> float:
        """Compute per-step reward using OMG-style error formulation.

        Based on the reward from OpenModelica Microgrid Gym
        (single_inverter_voltage_current_control_safe_opt.py):

        OMG error components:
          voltage_error = sum_phases( |v_ref - v_actual| / v_nom )^0.5
          current_error = sum_phases( |i_ref - i_actual| / i_lim )^0.5
          barrier = -sum_phases( mu * ln(1 - max(|i| - i_nom, 0) / (i_lim - i_nom)) )

        OMG reward = -(voltage_error + current_error + barrier)

        We map to [0, 1] via: reward = exp(omg_reward / normalization)
        where normalization scales so that typical good performance -> ~0.8
        """
        V_nom = self.config.inverter_V_nom
        i_lim = self.config.inverter_i_lim
        i_nom = self.config.inverter_i_nom
        mu = 2.0  # OMG barrier parameter

        # Get per-phase data from simulation
        v_errors_per_phase = obs.get("_v_abs_errors_per_phase", [])
        i_abs_per_phase = obs.get("_i_abs_per_phase", [])
        i_ref_per_phase = obs.get("_i_ref_per_phase", [])
        i_actual_per_phase = obs.get("_i_actual_per_phase", [])

        if not v_errors_per_phase:
            return 0.0

        n_samples = len(v_errors_per_phase)

        # Voltage tracking error: mean over time of sum over phases of sqrt(|error|/v_nom)
        voltage_error = 0.0
        for t_idx in range(n_samples):
            for phase in range(3):
                voltage_error += np.sqrt(
                    abs(v_errors_per_phase[t_idx][phase]) / V_nom
                )
        voltage_error /= n_samples

        # Current tracking error: mean over time of sum over phases of sqrt(|error|/i_lim)
        current_error = 0.0
        for t_idx in range(n_samples):
            for phase in range(3):
                i_err = abs(i_ref_per_phase[t_idx][phase] - i_actual_per_phase[t_idx][phase])
                current_error += np.sqrt(i_err / i_lim)
        current_error /= n_samples

        # Log-barrier penalty for current constraint violations
        barrier = 0.0
        margin = i_lim - i_nom
        if margin > 0:
            for t_idx in range(n_samples):
                for phase in range(3):
                    i_abs = i_abs_per_phase[t_idx][phase]
                    excess = max(i_abs - i_nom, 0.0)
                    ratio = excess / margin
                    # Clamp to avoid log(0)
                    if ratio >= 0.99:
                        barrier += mu * 5.0  # Large penalty
                    elif ratio > 0.0:
                        barrier -= mu * np.log(1.0 - ratio)
        barrier /= n_samples

        # Total OMG error (positive, larger is worse)
        omg_error = voltage_error + current_error + barrier

        # Map to [0, 1]: exp(-omg_error / scale)
        # Scale chosen so that moderate error (~3) maps to ~0.3
        # and good performance (error ~0.5) maps to ~0.8
        scale = 3.0
        reward = float(np.exp(-omg_error / scale))

        return reward

    def _format_observation(self, obs: dict, step_reward: float | None,
                            finished: bool, episode_reward: float) -> str:
        """Format observation as human-readable text."""
        parts = []
        parts.append(
            f"Step {obs['step']}/{self.config.max_steps} "
            f"(t = {obs['time']:.4f} s)"
        )
        parts.append("")

        if self.config.scenario == "droop_control":
            parts.append("Inverter 1:")
            parts.append("  Capacitor Voltages (V): " +
                         ", ".join(f"{v:.1f}" for v in obs["inv1_capacitor_voltages_V"]))
            parts.append("  Inductor Currents (A):  " +
                         ", ".join(f"{i:.2f}" for i in obs["inv1_inductor_currents_A"]))
            parts.append(f"  Voltage RMS: {obs['inv1_voltage_rms_V']:.1f} V")
            parts.append(f"  Active Power:   {obs['inv1_active_power_W']:.1f} W")
            parts.append(f"  Reactive Power: {obs['inv1_reactive_power_VAR']:.1f} VAR")
            parts.append("")
            parts.append("Inverter 2:")
            parts.append("  Capacitor Voltages (V): " +
                         ", ".join(f"{v:.1f}" for v in obs["inv2_capacitor_voltages_V"]))
            parts.append("  Inductor Currents (A):  " +
                         ", ".join(f"{i:.2f}" for i in obs["inv2_inductor_currents_A"]))
            parts.append(f"  Voltage RMS: {obs['inv2_voltage_rms_V']:.1f} V")
            parts.append(f"  Active Power:   {obs['inv2_active_power_W']:.1f} W")
            parts.append(f"  Reactive Power: {obs['inv2_reactive_power_VAR']:.1f} VAR")
            parts.append("")
            parts.append(f"Power Sharing Ratio (P1/P2): {obs.get('power_sharing_ratio', 0):.3f}")
        else:
            parts.append("Capacitor Voltages (V): " +
                         ", ".join(f"{v:.1f}" for v in obs["capacitor_voltages_V"]))
            parts.append("Voltage Reference (V):  " +
                         ", ".join(f"{v:.1f}" for v in obs["voltage_reference_V"]))
            parts.append(f"Voltage RMS: {obs['voltage_rms_V']:.1f} V "
                         f"(nominal: {self.config.inverter_V_nom / np.sqrt(2):.0f} V RMS)")
            if "mean_voltage_error_V" in obs:
                err_pct = obs["mean_voltage_error_V"] / self.config.inverter_V_nom * 100
                parts.append(
                    f"Mean Voltage Error: {obs['mean_voltage_error_V']:.2f} V "
                    f"({err_pct:.1f}%)"
                )
            parts.append("")
            parts.append("Inductor Currents (A): " +
                         ", ".join(f"{i:.2f}" for i in obs["inductor_currents_A"]))
            parts.append("Load Currents (A):     " +
                         ", ".join(f"{i:.2f}" for i in obs["load_currents_A"]))
            parts.append(f"Active Power:   {obs.get('active_power_W', 0):.1f} W")
            parts.append(f"Reactive Power: {obs.get('reactive_power_VAR', 0):.1f} VAR")

        if "max_inductor_current_A" in obs:
            parts.append(
                f"Peak Current: {obs['max_inductor_current_A']:.2f} A "
                f"(nominal: {self.config.inverter_i_nom:.0f} A, "
                f"limit: {obs.get('current_limit_A', self.config.inverter_i_lim):.0f} A)"
            )

        parts.append("")

        if step_reward is not None:
            parts.append(f"Step Reward: {step_reward:.4f}")

        if finished:
            parts.append("")
            parts.append("=== EPISODE COMPLETE ===")
            parts.append(f"Episode Reward: {episode_reward:.4f}")
        else:
            remaining = self.config.max_steps - obs["step"]
            parts.append(f"Remaining steps: {remaining}")

            # Disturbance hints
            if self.config.disturbance_step is not None:
                if obs["step"] < self.config.disturbance_step:
                    steps_until = self.config.disturbance_step - obs["step"]
                    parts.append(
                        f"Note: Load disturbance in {steps_until} step(s)"
                    )
                elif obs["step"] == self.config.disturbance_step:
                    parts.append("Note: Load disturbance occurred this step!")

            # Load profile hint
            if (self.config.scenario == "load_following"
                    and self.config.load_profile is not None
                    and obs["step"] < len(self.config.load_profile)):
                next_R = self.config.load_profile[obs["step"]]
                parts.append(f"Next step load resistance: {next_R:.1f} ohm")

        return "\n".join(parts)

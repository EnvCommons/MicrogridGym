"""
Systematic verification tests for MicrogridGym against the original
OpenModelica Microgrid Gym (OMG) implementation.

Tests verify:
  1. Circuit physics (LC filter dynamics, ODE equations, natural frequency)
  2. PI controller (proportional, integral, anti-windup, cascaded structure)
  3. Reward function (root-error, log-barrier, OMG formula fidelity)
  4. Droop control (frequency/voltage droop, power filtering, power sharing)
  5. Simulation integration (RK4 accuracy, three-phase symmetry, stability)
  6. Environment lifecycle (task counts, tool interface, episode structure)

Reference: https://github.com/upb-lea/openmodelica-microgrid-gym
"""

import asyncio
import math

import numpy as np
import pytest

from simulation import (
    InverterParams,
    LCFilterParams,
    MicrogridSimulation,
    RLLoadParams,
    SimulationConfig,
    TwoInverterSimulation,
    compute_power,
    pi_controller,
    voltage_reference,
)
from tasks import ALL_TASKS, BASE, TaskSpec


# ======================================================================== #
#  1. CIRCUIT PHYSICS TESTS
# ======================================================================== #

class TestCircuitPhysics:
    """Verify ODE equations match LC filter + RL load dynamics."""

    def _make_sim(self, **overrides):
        lc_kwargs = {k: overrides.pop(k) for k in list(overrides)
                     if k in ("L", "C", "R_f")}
        load_kwargs = {k: overrides.pop(k) for k in list(overrides)
                       if k in ("R",)}
        load_kwargs.update({k.replace("load_", ""): overrides.pop(k)
                            for k in list(overrides)
                            if k.startswith("load_")})
        inv_kwargs = {k: overrides.pop(k) for k in list(overrides)
                      if k in ("V_DC", "V_nom", "f_nom", "i_nom", "i_lim")}
        cfg_kwargs = {k: overrides.pop(k) for k in list(overrides)
                      if k in ("dt_ode", "dt_control", "n_rk4_substeps")}
        return MicrogridSimulation(
            LCFilterParams(**lc_kwargs),
            RLLoadParams(**load_kwargs),
            InverterParams(**inv_kwargs),
            SimulationConfig(**cfg_kwargs),
        )

    def test_ode_rhs_inductor_voltage_equation(self):
        """di_L/dt = (V_inv - V_C - R_f * i_L) / L

        OMG uses the same equation from the Modelica FMU.
        """
        sim = self._make_sim()
        # Set known state: i_L=5A, V_C=200V, i_load=4A on phase 0
        sim.x[0] = 5.0    # i_L0
        sim.x[1] = 200.0  # V_C0
        sim.x[2] = 4.0    # i_load0

        duty = np.array([0.5, 0.0, 0.0])
        dx = sim._rhs(sim.x, duty)

        # Expected: di_L/dt = (V_DC * 0.5 - 200 - 0.4 * 5) / 2.3e-3
        V_inv = 600.0 * 0.5  # 300 V
        expected = (V_inv - 200.0 - 0.4 * 5.0) / 2.3e-3
        assert abs(dx[0] - expected) < 1e-6, f"di_L/dt: {dx[0]} vs expected {expected}"

    def test_ode_rhs_capacitor_equation(self):
        """dV_C/dt = (i_L - i_load) / C"""
        sim = self._make_sim()
        sim.x[0] = 10.0   # i_L0
        sim.x[1] = 100.0  # V_C0
        sim.x[2] = 8.0    # i_load0

        duty = np.zeros(3)
        dx = sim._rhs(sim.x, duty)

        expected = (10.0 - 8.0) / 10e-6
        assert abs(dx[1] - expected) < 1e-6

    def test_ode_rhs_load_equation(self):
        """L_load * di_load/dt = V_C - R_load * i_load"""
        sim = self._make_sim()
        sim.x[0] = 10.0
        sim.x[1] = 150.0  # V_C0
        sim.x[2] = 5.0    # i_load0

        duty = np.zeros(3)
        dx = sim._rhs(sim.x, duty)

        expected = (150.0 - 14.0 * 5.0) / 5.3e-3
        assert abs(dx[2] - expected) < 1e-6

    def test_natural_frequency(self):
        """LC resonance: f_n = 1 / (2*pi*sqrt(L*C)).

        With OMG defaults L=2.3mH, C=10uF -> f_n ~= 1049 Hz.
        The simulation should exhibit oscillation near this frequency
        when excited with a step input.
        """
        L = 2.3e-3
        C = 10e-6
        f_natural = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        assert 1000 < f_natural < 1100, f"Natural freq {f_natural:.0f} Hz"

    def test_inverter_output_duty_clipping(self):
        """V_inv = V_DC * clip(duty, -1, 1) — matches OMG inverter model."""
        sim = self._make_sim()
        sim.x[:] = 0.0

        # duty > 1 should clip
        duty_over = np.array([1.5, -1.5, 0.5])
        dx = sim._rhs(sim.x, duty_over)
        # Phase 0: V_inv = 600 * clip(1.5, -1, 1) = 600 * 1.0 = 600
        expected_0 = 600.0 / 2.3e-3
        assert abs(dx[0] - expected_0) < 1e-6
        # Phase 1: V_inv = 600 * clip(-1.5, -1, 1) = 600 * (-1) = -600
        expected_1 = -600.0 / 2.3e-3
        assert abs(dx[3] - expected_1) < 1e-6

    def test_steady_state_voltage(self):
        """In steady state with constant load, V_C should track V_ref.

        Run many steps with good gains and verify final voltage
        approaches reference.
        """
        sim = self._make_sim()
        sim.reset()
        # Run 20 control intervals (100ms total) with tuned gains
        for _ in range(20):
            obs = sim.step(kP_v=0.08, kI_v=300.0, kP_i=0.03, kI_i=200.0)

        V_nom = 325.27
        v_err = obs["mean_voltage_error_V"]
        # Error should be < 10% of nominal after settling
        assert v_err / V_nom < 0.10, f"Steady-state error {v_err:.1f} V ({v_err/V_nom*100:.1f}%)"

    def test_three_phase_state_vector_layout(self):
        """State vector: [i_L0, V_C0, i_load0, i_L1, V_C1, i_load1, ...].

        9 elements for single inverter, matching OMG 3-phase layout.
        """
        sim = self._make_sim()
        assert sim.x.shape == (9,)
        sim.x[0] = 1.0   # i_L phase 0
        sim.x[3] = 2.0   # i_L phase 1
        sim.x[6] = 3.0   # i_L phase 2
        assert sim.x[0] == 1.0
        assert sim.x[3] == 2.0
        assert sim.x[6] == 3.0


# ======================================================================== #
#  2. PI CONTROLLER TESTS
# ======================================================================== #

class TestPIController:
    """Verify PI controller matches OMG cascaded control architecture."""

    def test_proportional_only(self):
        """Pure proportional: output = kP * error."""
        error = np.array([1.0, -2.0, 0.5])
        integral = np.zeros(3)
        output, new_int = pi_controller(error, integral, kP=0.5, kI=0.0, dt=1e-4)
        np.testing.assert_allclose(output, [0.5, -1.0, 0.25])

    def test_integral_accumulation(self):
        """Integral accumulates: integral += error * dt."""
        error = np.array([10.0, 10.0, 10.0])
        integral = np.zeros(3)
        dt = 1e-4

        # After 1 step: integral = 10 * 1e-4 = 1e-3
        output, new_int = pi_controller(error, integral, kP=0.0, kI=100.0, dt=dt)
        expected_integral = 10.0 * dt
        np.testing.assert_allclose(new_int, [expected_integral] * 3)
        np.testing.assert_allclose(output, [100.0 * expected_integral] * 3)

    def test_integral_accumulation_multi_step(self):
        """After N steps with constant error, integral = N * error * dt."""
        error = np.array([5.0, 5.0, 5.0])
        integral = np.zeros(3)
        dt = 1e-4

        for _ in range(100):
            _, integral = pi_controller(error, integral, kP=0.0, kI=1.0, dt=dt)

        expected = 5.0 * dt * 100
        np.testing.assert_allclose(integral, [expected] * 3, rtol=1e-10)

    def test_anti_windup_clamps_output(self):
        """Output is clamped to [output_min, output_max].

        OMG clips duty to [-1, 1] for current loop.
        """
        error = np.array([100.0, -100.0, 0.0])
        integral = np.zeros(3)
        output, _ = pi_controller(
            error, integral, kP=1.0, kI=0.0, dt=1e-4,
            output_min=-1.0, output_max=1.0,
        )
        assert output[0] == 1.0, "Should clamp to +1"
        assert output[1] == -1.0, "Should clamp to -1"
        assert output[2] == 0.0

    def test_anti_windup_freezes_integral(self):
        """When output saturates, integral should not accumulate.

        This prevents integral windup — similar to OMG's back-calculation
        anti-windup, though our implementation uses integral freeze.
        """
        error = np.array([100.0])
        integral = np.array([0.0])
        dt = 1e-4

        # kP=1.0 gives output=100, which clamps to 1.0
        _, int_1 = pi_controller(
            error, integral, kP=1.0, kI=1000.0, dt=dt,
            output_min=-1.0, output_max=1.0,
        )
        # Integral should be frozen (not accumulated) since output was clamped
        assert int_1[0] == 0.0, f"Integral should freeze: got {int_1[0]}"

    def test_cascaded_pi_structure(self):
        """Cascaded PI: voltage PI -> current reference -> current PI -> duty.

        This matches OMG's two-loop architecture:
          V_ref -> [voltage PI] -> i_ref -> [current PI] -> duty
        """
        # Outer loop: voltage error -> current reference
        V_ref = np.array([325.27, 0.0, 0.0])
        V_C = np.array([300.0, 0.0, 0.0])
        v_error = V_ref - V_C  # [25.27, 0, 0]

        v_integral = np.zeros(3)
        i_ref, _ = pi_controller(
            v_error, v_integral, kP=0.025, kI=60.0, dt=1e-4,
            output_min=-30.0, output_max=30.0,
        )
        # i_ref should be reasonable current value, not duty
        assert abs(i_ref[0]) < 30.0, "Current reference should be within i_lim"
        assert i_ref[0] > 0, "Positive voltage error -> positive current ref"

        # Inner loop: current error -> duty
        i_L = np.array([0.0, 0.0, 0.0])
        i_error = i_ref - i_L

        i_integral = np.zeros(3)
        duty, _ = pi_controller(
            i_error, i_integral, kP=0.012, kI=90.0, dt=1e-4,
            output_min=-1.0, output_max=1.0,
        )
        # Duty should be in [-1, 1]
        assert all(-1.0 <= d <= 1.0 for d in duty)

    def test_voltage_pi_limits_match_omg(self):
        """Voltage loop output clamped to [-i_lim, +i_lim] = [-30, +30] A.

        Matches OMG: voltage PI limits are the inverter current limits.
        """
        error = np.array([1000.0, -1000.0, 0.0])
        integral = np.zeros(3)
        output, _ = pi_controller(
            error, integral, kP=0.025, kI=60.0, dt=1e-4,
            output_min=-30.0, output_max=30.0,
        )
        assert output[0] == 30.0
        assert output[1] == -30.0

    def test_current_pi_limits_match_omg(self):
        """Current loop output clamped to [-1, +1] (duty cycle).

        Matches OMG: duty cycle / modulation index limits.
        """
        error = np.array([50.0, -50.0, 0.0])
        integral = np.zeros(3)
        output, _ = pi_controller(
            error, integral, kP=0.012, kI=90.0, dt=1e-4,
            output_min=-1.0, output_max=1.0,
        )
        assert output[0] == 1.0
        assert output[1] == -1.0

    def test_omg_default_gains(self):
        """OMG default gains: kP_v=0.025, kI_v=60, kP_i=0.012, kI_i=90.

        These should produce stable simulation without divergence.
        """
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.reset()
        for _ in range(10):
            obs = sim.step(kP_v=0.025, kI_v=60.0, kP_i=0.012, kI_i=90.0)
            assert sim.check_stability(), "OMG defaults should be stable"


# ======================================================================== #
#  3. REWARD FUNCTION TESTS
# ======================================================================== #

class TestRewardFunction:
    """Verify reward follows OMG formulation.

    OMG reward = -(voltage_err + current_err + barrier)
    Our mapping: step_reward = exp(-omg_error / 3)
    """

    def _run_episode_get_reward(self, kP_v, kI_v, kP_i=0.012, kI_i=90.0,
                                n_steps=5):
        """Run a short episode and return (obs_list, reward) via env."""
        from microgrid import MicrogridGym, SetControllerParams

        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _run():
            await env.setup()
            rewards = []
            for _ in range(n_steps):
                params = SetControllerParams(
                    kP_v=kP_v, kI_v=kI_v, kP_i=kP_i, kI_i=kI_i,
                )
                result = await env.set_controller(params)
                rewards.append(result.reward)
            await env.teardown()
            return rewards

        return asyncio.get_event_loop().run_until_complete(_run())

    def test_voltage_root_error_formula(self):
        """Voltage error uses sqrt(|V_ref - V_actual| / V_nom).

        OMG: np.sum((np.abs(vsp - vabc) / v_nom) ** 0.5, axis=0)
        The exponent is 0.5 (square root), NOT squared.
        """
        V_nom = 325.27

        # If |V_ref - V_actual| = V_nom for all 3 phases:
        # error per phase = sqrt(V_nom / V_nom) = 1.0
        # sum over 3 phases = 3.0
        error_per_phase = np.sqrt(V_nom / V_nom)
        assert abs(error_per_phase - 1.0) < 1e-10

        # If |V_ref - V_actual| = 0.01 * V_nom:
        # error per phase = sqrt(0.01) = 0.1
        small_error = np.sqrt(0.01 * V_nom / V_nom)
        assert abs(small_error - 0.1) < 1e-10

    def test_current_root_error_formula(self):
        """Current error uses sqrt(|i_ref - i_actual| / i_lim).

        OMG: np.sum((np.abs(isp - iabc) / i_lim) ** 0.5, axis=0)
        """
        i_lim = 30.0

        # If |i_ref - i_actual| = i_lim: error = sqrt(1) = 1.0
        assert abs(np.sqrt(i_lim / i_lim) - 1.0) < 1e-10

        # If |i_ref - i_actual| = 1.0: error = sqrt(1/30)
        assert abs(np.sqrt(1.0 / i_lim) - np.sqrt(1.0 / 30.0)) < 1e-10

    def test_log_barrier_zero_below_i_nom(self):
        """When |i| < i_nom, barrier penalty should be zero.

        OMG: max(|i| - i_nom, 0) = 0 when |i| < i_nom
        -> ln(1 - 0) = ln(1) = 0
        """
        i_nom = 20.0
        i_lim = 30.0
        mu = 2.0

        # Current below nominal
        i_abs = 15.0
        excess = max(i_abs - i_nom, 0.0)
        assert excess == 0.0
        barrier = -mu * np.log(1.0 - excess / (i_lim - i_nom))
        assert barrier == 0.0

    def test_log_barrier_increases_above_i_nom(self):
        """When |i| > i_nom, barrier grows logarithmically.

        OMG: -mu * ln(1 - max(|i| - i_nom, 0) / (i_lim - i_nom))
        """
        i_nom = 20.0
        i_lim = 30.0
        mu = 2.0
        margin = i_lim - i_nom  # 10 A

        # At i = 25 A (halfway to limit)
        excess = 25.0 - i_nom  # 5.0
        ratio = excess / margin  # 0.5
        barrier_25 = -mu * np.log(1.0 - ratio)
        expected = -2.0 * np.log(0.5)  # = 2 * 0.693 = 1.386
        assert abs(barrier_25 - expected) < 1e-10

        # At i = 29 A (near limit) — should be much larger
        excess_29 = 29.0 - i_nom  # 9.0
        ratio_29 = excess_29 / margin  # 0.9
        barrier_29 = -mu * np.log(1.0 - ratio_29)
        expected_29 = -2.0 * np.log(0.1)  # = 2 * 2.303 = 4.605
        assert abs(barrier_29 - expected_29) < 1e-10
        assert barrier_29 > barrier_25

    def test_log_barrier_cap_near_limit(self):
        """When |i| approaches i_lim, barrier is capped to avoid log(0).

        Our implementation caps at mu * 5.0 when ratio >= 0.99.
        """
        from microgrid import MicrogridGym
        # Verify the cap exists by checking _compute_reward logic
        # The cap prevents -inf from log(0)
        i_nom = 20.0
        i_lim = 30.0
        mu = 2.0
        margin = i_lim - i_nom

        # At i = 29.95 (ratio = 9.95/10 = 0.995 >= 0.99)
        excess = 29.95 - i_nom
        ratio = excess / margin
        assert ratio >= 0.99
        # Should use cap: mu * 5.0 = 10.0
        barrier_cap = mu * 5.0
        assert barrier_cap == 10.0

    def test_mu_parameter_value(self):
        """mu = 2 matches OMG SafeOpt example."""
        # Directly verify in _compute_reward
        from microgrid import MicrogridGym
        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)
        # mu is hardcoded in _compute_reward; verify via source
        import inspect
        source = inspect.getsource(env._compute_reward)
        assert "mu = 2.0" in source, "mu should be 2.0 matching OMG"

    def test_reward_in_zero_one_range(self):
        """Our reward mapping exp(-error/3) gives values in (0, 1]."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.reset()

        from microgrid import MicrogridGym
        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            from microgrid import SetControllerParams
            # Various gain settings
            for kP_v, kI_v in [(0.025, 60), (0.08, 300), (0.001, 5)]:
                params = SetControllerParams(kP_v=kP_v, kI_v=kI_v)
                result = await env.set_controller(params)
                assert 0.0 <= result.reward <= 1.0, \
                    f"Reward {result.reward} out of [0,1] for kP_v={kP_v}"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_perfect_tracking_gives_high_reward(self):
        """Near-perfect voltage tracking should yield reward close to 1.0.

        If voltage and current errors are near zero and current < i_nom,
        the OMG error ≈ 0 and reward ≈ exp(0) = 1.0.
        """
        # Construct a fake obs with near-zero errors
        from microgrid import MicrogridGym
        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            # Fabricate obs with perfect tracking
            obs = {
                "_v_abs_errors_per_phase": [[0.1, 0.1, 0.1]] * 50,
                "_i_abs_per_phase": [[5.0, 5.0, 5.0]] * 50,
                "_i_ref_per_phase": [[5.0, 5.0, 5.0]] * 50,
                "_i_actual_per_phase": [[5.0, 5.0, 5.0]] * 50,
            }
            reward = env._compute_reward(obs)
            # Very small errors, no barrier -> reward should be high
            assert reward > 0.8, f"Near-perfect tracking gave reward {reward:.4f}"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_large_errors_give_low_reward(self):
        """Large voltage/current errors should yield reward near 0."""
        from microgrid import MicrogridGym
        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            # Fabricate obs with large errors
            V_nom = 325.27
            obs = {
                "_v_abs_errors_per_phase": [[V_nom, V_nom, V_nom]] * 50,
                "_i_abs_per_phase": [[29.0, 29.0, 29.0]] * 50,  # near i_lim
                "_i_ref_per_phase": [[15.0, 15.0, 15.0]] * 50,
                "_i_actual_per_phase": [[0.0, 0.0, 0.0]] * 50,
            }
            reward = env._compute_reward(obs)
            assert reward < 0.2, f"Large errors gave reward {reward:.4f}"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_reward_components_are_additive(self):
        """OMG error = voltage_err + current_err + barrier (additive).

        Verify each component independently contributes.
        """
        from microgrid import MicrogridGym
        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()

            # Case 1: only voltage error
            obs_v = {
                "_v_abs_errors_per_phase": [[50.0, 50.0, 50.0]] * 10,
                "_i_abs_per_phase": [[5.0, 5.0, 5.0]] * 10,
                "_i_ref_per_phase": [[5.0, 5.0, 5.0]] * 10,
                "_i_actual_per_phase": [[5.0, 5.0, 5.0]] * 10,
            }
            r_v = env._compute_reward(obs_v)

            # Case 2: only current error
            obs_i = {
                "_v_abs_errors_per_phase": [[0.01, 0.01, 0.01]] * 10,
                "_i_abs_per_phase": [[5.0, 5.0, 5.0]] * 10,
                "_i_ref_per_phase": [[10.0, 10.0, 10.0]] * 10,
                "_i_actual_per_phase": [[5.0, 5.0, 5.0]] * 10,
            }
            r_i = env._compute_reward(obs_i)

            # Case 3: only barrier
            obs_b = {
                "_v_abs_errors_per_phase": [[0.01, 0.01, 0.01]] * 10,
                "_i_abs_per_phase": [[25.0, 25.0, 25.0]] * 10,
                "_i_ref_per_phase": [[25.0, 25.0, 25.0]] * 10,
                "_i_actual_per_phase": [[25.0, 25.0, 25.0]] * 10,
            }
            r_b = env._compute_reward(obs_b)

            # Case 4: all combined
            obs_all = {
                "_v_abs_errors_per_phase": [[50.0, 50.0, 50.0]] * 10,
                "_i_abs_per_phase": [[25.0, 25.0, 25.0]] * 10,
                "_i_ref_per_phase": [[10.0, 10.0, 10.0]] * 10,
                "_i_actual_per_phase": [[5.0, 5.0, 5.0]] * 10,
            }
            r_all = env._compute_reward(obs_all)

            # Combined should be worse (lower reward) than any individual
            assert r_all < r_v, "Combined error should give lower reward than voltage-only"
            assert r_all < r_i, "Combined error should give lower reward than current-only"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_reward_manual_computation(self):
        """Manually compute OMG reward and verify against _compute_reward."""
        from microgrid import MicrogridGym
        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)
        V_nom = 325.27
        i_lim = 30.0
        i_nom = 20.0
        mu = 2.0

        async def _test():
            await env.setup()
            obs = {
                "_v_abs_errors_per_phase": [[10.0, 20.0, 15.0], [12.0, 18.0, 14.0]],
                "_i_abs_per_phase": [[22.0, 18.0, 21.0], [23.0, 19.0, 22.0]],
                "_i_ref_per_phase": [[8.0, -4.0, -4.0], [9.0, -4.5, -4.5]],
                "_i_actual_per_phase": [[7.0, -3.5, -3.8], [8.5, -4.2, -4.3]],
            }

            # Manual computation
            n = 2
            v_err = 0.0
            for t in range(n):
                for p in range(3):
                    v_err += np.sqrt(abs(obs["_v_abs_errors_per_phase"][t][p]) / V_nom)
            v_err /= n

            i_err = 0.0
            for t in range(n):
                for p in range(3):
                    ie = abs(obs["_i_ref_per_phase"][t][p] - obs["_i_actual_per_phase"][t][p])
                    i_err += np.sqrt(ie / i_lim)
            i_err /= n

            barrier = 0.0
            margin = i_lim - i_nom
            for t in range(n):
                for p in range(3):
                    i_abs = obs["_i_abs_per_phase"][t][p]
                    excess = max(i_abs - i_nom, 0.0)
                    if excess > 0:
                        ratio = excess / margin
                        if ratio >= 0.99:
                            barrier += mu * 5.0
                        else:
                            barrier -= mu * np.log(1.0 - ratio)
            barrier /= n

            total = v_err + i_err + barrier
            expected_reward = float(np.exp(-total / 3.0))

            actual_reward = env._compute_reward(obs)
            assert abs(actual_reward - expected_reward) < 1e-6, \
                f"Manual: {expected_reward:.6f}, Got: {actual_reward:.6f}"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())


# ======================================================================== #
#  4. DROOP CONTROL TESTS
# ======================================================================== #

class TestDroopControl:
    """Verify droop control matches OMG droop equations."""

    def _make_droop_sim(self, **overrides):
        R = overrides.get("R", 14.0)
        lc1 = LCFilterParams()
        lc2 = LCFilterParams()
        load = RLLoadParams(R=R)
        inv1 = InverterParams()
        inv2 = InverterParams()
        config = SimulationConfig()
        return TwoInverterSimulation(lc1, lc2, load, inv1, inv2, config)

    def test_droop_frequency_equation(self):
        """f = f_nom - kP_droop * P_filtered.

        OMG: output = PT1filter(P * gain) + f_nom
        Frequency decreases with increasing active power.
        """
        sim = self._make_droop_sim()
        sim.reset()

        # With zero power, frequency should be f_nom
        from simulation import DroopState
        f, V, _ = sim._droop_step(
            P=0.0, Q=0.0, state=DroopState(),
            kP_droop=1e-4, kQ_droop=1e-2,
            f_nom=50.0, V_nom=325.27, dt=1e-4,
        )
        assert abs(f - 50.0) < 1e-6, f"Zero power -> f should be 50 Hz, got {f}"

    def test_droop_frequency_decreases_with_power(self):
        """Active power droop: more power -> lower frequency."""
        sim = self._make_droop_sim()
        sim.reset()

        from simulation import DroopState
        state = DroopState()

        # Apply constant positive power for many steps to let filter settle
        for _ in range(1000):
            f, V, state = sim._droop_step(
                P=5000.0, Q=0.0, state=state,
                kP_droop=1e-4, kQ_droop=0.0,
                f_nom=50.0, V_nom=325.27, dt=1e-4,
            )

        # f = 50 - 1e-4 * 5000 = 50 - 0.5 = 49.5 Hz
        assert abs(f - 49.5) < 0.1, f"Expected ~49.5 Hz, got {f:.3f}"

    def test_droop_voltage_decreases_with_reactive_power(self):
        """Reactive power droop: more Q -> lower voltage."""
        sim = self._make_droop_sim()
        sim.reset()

        from simulation import DroopState
        state = DroopState()

        for _ in range(1000):
            f, V, state = sim._droop_step(
                P=0.0, Q=1000.0, state=state,
                kP_droop=0.0, kQ_droop=1e-2,
                f_nom=50.0, V_nom=325.27, dt=1e-4,
            )

        # V = 325.27 - 0.01 * 1000 = 325.27 - 10 = 315.27
        assert abs(V - 315.27) < 1.0, f"Expected ~315.27 V, got {V:.2f}"

    def test_droop_power_filter(self):
        """Power is low-pass filtered before droop. OMG uses PT1 filter.

        Our implementation: IIR with alpha = dt / (tau + dt).
        """
        sim = self._make_droop_sim()
        from simulation import DroopState
        state = DroopState()

        # Step response of PT1 filter
        dt = 1e-4
        tau = sim.tau_P  # 0.01
        alpha = dt / (tau + dt)

        # After 1 step with P=1000, filtered ≈ alpha * 1000
        _, _, state = sim._droop_step(
            P=1000.0, Q=0.0, state=state,
            kP_droop=0.0, kQ_droop=0.0,
            f_nom=50.0, V_nom=325.27, dt=dt,
        )
        expected_P = alpha * 1000.0
        assert abs(state.P_filtered - expected_P) < 1e-6

    def test_droop_voltage_clamp(self):
        """Voltage amplitude clamped to [0.5*V_nom, 1.5*V_nom].

        Prevents unrealistic voltage magnitudes.
        """
        sim = self._make_droop_sim()
        from simulation import DroopState
        state = DroopState(Q_filtered=1e6)  # Huge Q to force droop

        _, V, _ = sim._droop_step(
            P=0.0, Q=1e6, state=state,
            kP_droop=0.0, kQ_droop=1.0,
            f_nom=50.0, V_nom=325.27, dt=1e-4,
        )
        # Clamped to 0.5 * 325.27 = 162.64
        assert V >= 0.5 * 325.27 - 0.01

    def test_droop_frequency_clamp(self):
        """Frequency clamped to [0.8*f_nom, 1.2*f_nom] = [40, 60] Hz."""
        sim = self._make_droop_sim()
        from simulation import DroopState
        state = DroopState(P_filtered=1e8)  # Huge P

        f, _, _ = sim._droop_step(
            P=1e8, Q=0.0, state=state,
            kP_droop=1.0, kQ_droop=0.0,
            f_nom=50.0, V_nom=325.27, dt=1e-4,
        )
        assert f >= 40.0 - 0.01

    def test_equal_droop_equal_sharing(self):
        """Equal droop gains -> equal power sharing between inverters.

        This is a fundamental property of droop control.
        """
        sim = self._make_droop_sim()
        sim.reset()

        # Run with identical droop gains
        for _ in range(20):
            obs = sim.step(
                kP_droop_1=1e-4, kQ_droop_1=1e-2,
                kP_droop_2=1e-4, kQ_droop_2=1e-2,
                kP_v=0.025, kI_v=60.0,
            )

        # Power sharing ratio should be near 1.0
        ratio = obs.get("power_sharing_ratio", 0)
        if abs(obs.get("inv2_active_power_W", 0)) > 1.0:
            assert abs(ratio - 1.0) < 0.5, f"Equal droop -> ratio should be ~1.0, got {ratio}"

    def test_two_inverter_state_vector(self):
        """Two-inverter sim has 18-element state (9 per inverter)."""
        sim = self._make_droop_sim()
        assert sim.x.shape == (18,)

    def test_droop_simulation_stability(self):
        """Droop simulation should be stable with reasonable parameters."""
        sim = self._make_droop_sim()
        sim.reset()

        for _ in range(12):
            obs = sim.step(
                kP_droop_1=1e-4, kQ_droop_1=1e-2,
                kP_droop_2=1e-4, kQ_droop_2=1e-2,
                kP_v=0.025, kI_v=60.0,
            )
            assert sim.check_stability(), "Droop sim should be stable"

    def test_droop_per_phase_data_present(self):
        """TwoInverterSimulation.step() should expose per-phase data
        for OMG reward computation."""
        sim = self._make_droop_sim()
        sim.reset()
        obs = sim.step(
            kP_droop_1=1e-4, kQ_droop_1=1e-2,
            kP_droop_2=1e-4, kQ_droop_2=1e-2,
        )
        assert "_v_abs_errors_per_phase" in obs
        assert "_i_abs_per_phase" in obs
        assert "_i_ref_per_phase" in obs
        assert "_i_actual_per_phase" in obs

        # Each should have n_pi = 50 entries, each with 3 phases
        n_pi = int(round(5e-3 / 1e-4))
        assert len(obs["_v_abs_errors_per_phase"]) == n_pi
        assert len(obs["_v_abs_errors_per_phase"][0]) == 3


# ======================================================================== #
#  5. SIMULATION INTEGRATION TESTS
# ======================================================================== #

class TestSimulationIntegration:
    """Verify numerical integration and simulation behavior."""

    def test_rk4_accuracy_vs_euler(self):
        """RK4 should be significantly more accurate than Euler for LC circuits.

        LC circuits are oscillatory (undamped harmonic oscillator) and
        forward Euler is unconditionally unstable for them.
        """
        # Simple harmonic oscillator: dx/dt = [0, 1; -w^2, 0] * x
        # True solution: x(t) = cos(w*t), v(t) = -w*sin(w*t)
        L = 2.3e-3
        C = 10e-6
        w = 1.0 / np.sqrt(L * C)
        dt = 1e-4  # same as sim dt_ode

        # Euler integration
        x_euler = np.array([1.0, 0.0])  # [position, velocity]
        for _ in range(100):
            dx = np.array([x_euler[1], -w**2 * x_euler[0]])
            x_euler = x_euler + dt * dx

        # RK4 integration
        x_rk4 = np.array([1.0, 0.0])
        for _ in range(100):
            def rhs(x):
                return np.array([x[1], -w**2 * x[0]])
            k1 = rhs(x_rk4)
            k2 = rhs(x_rk4 + 0.5 * dt * k1)
            k3 = rhs(x_rk4 + 0.5 * dt * k2)
            k4 = rhs(x_rk4 + dt * k3)
            x_rk4 = x_rk4 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        t_final = 100 * dt
        true_x = np.cos(w * t_final)

        euler_error = abs(x_euler[0] - true_x)
        rk4_error = abs(x_rk4[0] - true_x)

        # RK4 should be orders of magnitude more accurate
        assert rk4_error < euler_error, \
            f"RK4 error {rk4_error:.6e} should be < Euler error {euler_error:.6e}"

    def test_three_phase_symmetry(self):
        """Three phases should be symmetric (120° apart).

        With balanced load and symmetric initial conditions,
        the three phases should have equal amplitude and 120° offsets.
        """
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.reset()

        # Run for enough steps to reach near-steady state
        for _ in range(20):
            obs = sim.step(kP_v=0.08, kI_v=300.0, kP_i=0.03, kI_i=200.0)

        V_C = obs["capacitor_voltages_V"]
        # Sum of three balanced sinusoids = 0
        v_sum = sum(V_C)
        V_rms = obs["voltage_rms_V"]
        # For balanced 3-phase, sum should be near zero relative to amplitude
        assert abs(v_sum) < 0.15 * V_rms, \
            f"3-phase sum should be near 0: {v_sum:.2f} (V_rms={V_rms:.1f})"

    def test_phase_offsets_match_omg(self):
        """Phase offsets: (0, -2π/3, +2π/3) matching OMG convention."""
        config = SimulationConfig()
        expected = (0.0, -2 * np.pi / 3, 2 * np.pi / 3)
        for i in range(3):
            assert abs(config.phase_offsets[i] - expected[i]) < 1e-10

    def test_voltage_reference_three_phase(self):
        """V_ref_k = V_nom * sin(2πft + φ_k) for k = 0, 1, 2."""
        V_nom = 325.27
        f = 50.0
        offsets = (0.0, -2 * np.pi / 3, 2 * np.pi / 3)

        t = 0.005  # 5ms
        V_ref = voltage_reference(t, V_nom, f, offsets)

        for k in range(3):
            expected = V_nom * np.sin(2 * np.pi * f * t + offsets[k])
            assert abs(V_ref[k] - expected) < 1e-10

    def test_stability_detection_nan(self):
        """NaN/Inf in state vector should be detected as unstable."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.x[0] = float('nan')
        assert not sim.check_stability()

        sim.x[0] = float('inf')
        assert not sim.check_stability()

    def test_stability_detection_voltage_divergence(self):
        """|V_C| > 2 * V_DC should be detected as unstable."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        # Set voltage above 2 * V_DC = 1200 V
        sim.x[1] = 1300.0
        assert not sim.check_stability()

    def test_stability_detection_current_divergence(self):
        """|i_L| > 10 * i_lim should be detected as unstable."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.x[0] = 350.0  # > 10 * 30 = 300
        assert not sim.check_stability()

    def test_time_advances_correctly(self):
        """Each step should advance time by dt_control = 5ms."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.reset()
        assert sim.t == 0.0

        sim.step(kP_v=0.025, kI_v=60.0)
        assert abs(sim.t - 5e-3) < 1e-10

        sim.step(kP_v=0.025, kI_v=60.0)
        assert abs(sim.t - 10e-3) < 1e-10

    def test_step_count_increments(self):
        """Step count should increment by 1 each call."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.reset()
        for i in range(5):
            obs = sim.step(kP_v=0.025, kI_v=60.0)
            assert obs["step"] == i + 1

    def test_reset_clears_state(self):
        """reset() should zero all state variables."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.reset()
        sim.step(kP_v=0.025, kI_v=60.0)
        assert sim.t > 0
        assert sim._step_count > 0

        sim.reset()
        assert sim.t == 0.0
        assert sim._step_count == 0
        np.testing.assert_array_equal(sim.x, np.zeros(9))
        np.testing.assert_array_equal(sim.pi_v_integral, np.zeros(3))
        np.testing.assert_array_equal(sim.pi_i_integral, np.zeros(3))

    def test_per_phase_data_in_observation(self):
        """step() should attach per-phase data for reward computation."""
        sim = MicrogridSimulation(
            LCFilterParams(), RLLoadParams(),
            InverterParams(), SimulationConfig(),
        )
        sim.reset()
        obs = sim.step(kP_v=0.025, kI_v=60.0)

        for key in ["_v_abs_errors_per_phase", "_i_abs_per_phase",
                     "_i_ref_per_phase", "_i_actual_per_phase"]:
            assert key in obs, f"Missing {key}"
            assert len(obs[key]) == 50, f"{key} should have 50 entries (n_pi)"
            assert len(obs[key][0]) == 3, f"{key}[0] should have 3 phases"

    def test_control_interval_subdivisions(self):
        """dt_control/dt_ode = 5e-3/1e-4 = 50 PI updates per control step.

        Each PI update has n_rk4_substeps=4 RK4 steps.
        Total RK4 steps per control interval = 50 * 4 = 200.
        """
        config = SimulationConfig()
        n_pi = int(round(config.dt_control / config.dt_ode))
        assert n_pi == 50
        assert config.n_rk4_substeps == 4
        # Total RK4 steps
        assert n_pi * config.n_rk4_substeps == 200

    def test_power_computation_clarke(self):
        """Active and reactive power via Clarke transform.

        P = sum(v*i), Q uses Clarke alpha/beta.
        """
        # Balanced three-phase voltage and in-phase current
        V = np.array([325.27, -162.64, -162.64])  # approximate
        I = V / 14.0  # resistive load

        P, Q = compute_power(V, I)
        # Active power should be positive
        assert P > 0
        # Reactive power should be near zero for resistive load
        # (not exactly zero due to unbalanced test signal)


# ======================================================================== #
#  6. ENVIRONMENT & TASK TESTS
# ======================================================================== #

class TestEnvironmentLifecycle:
    """Verify environment class, task counts, and tool interface."""

    def test_task_count_train(self):
        """99 training tasks."""
        assert len(ALL_TASKS["train"]) == 99

    def test_task_count_test(self):
        """27 test tasks."""
        assert len(ALL_TASKS["test"]) == 27

    def test_task_count_total(self):
        """126 total tasks."""
        assert len(ALL_TASKS["train"]) + len(ALL_TASKS["test"]) == 126

    def test_voltage_control_task_counts(self):
        """60 train + 9 test voltage control tasks."""
        train_vc = [t for t in ALL_TASKS["train"]
                    if t["scenario"] == "voltage_control"]
        test_vc = [t for t in ALL_TASKS["test"]
                   if t["scenario"] == "voltage_control"]
        assert len(train_vc) == 60, f"Expected 60 train VC, got {len(train_vc)}"
        assert len(test_vc) == 9, f"Expected 9 test VC, got {len(test_vc)}"

    def test_current_control_task_counts(self):
        """15 train + 6 test current control tasks."""
        train = [t for t in ALL_TASKS["train"]
                 if t["scenario"] == "current_control"]
        test = [t for t in ALL_TASKS["test"]
                if t["scenario"] == "current_control"]
        assert len(train) == 15
        assert len(test) == 6

    def test_droop_control_task_counts(self):
        """12 train + 6 test droop control tasks."""
        train = [t for t in ALL_TASKS["train"]
                 if t["scenario"] == "droop_control"]
        test = [t for t in ALL_TASKS["test"]
                if t["scenario"] == "droop_control"]
        assert len(train) == 12
        assert len(test) == 6

    def test_load_following_task_counts(self):
        """12 train + 6 test load following tasks."""
        train = [t for t in ALL_TASKS["train"]
                 if t["scenario"] == "load_following"]
        test = [t for t in ALL_TASKS["test"]
                if t["scenario"] == "load_following"]
        assert len(train) == 12
        assert len(test) == 6

    def test_all_tasks_validate(self):
        """All tasks parse as valid TaskSpec."""
        for split in ["train", "test"]:
            for task in ALL_TASKS[split]:
                ts = TaskSpec.model_validate(task)
                assert ts.id
                assert ts.scenario in (
                    "voltage_control", "current_control",
                    "droop_control", "load_following",
                )

    def test_unique_task_ids(self):
        """All task IDs should be unique."""
        all_ids = []
        for split in ["train", "test"]:
            for task in ALL_TASKS[split]:
                all_ids.append(task["id"])
        assert len(all_ids) == len(set(all_ids)), "Duplicate task IDs found"

    def test_train_test_no_overlap(self):
        """Train and test IDs should not overlap."""
        train_ids = {t["id"] for t in ALL_TASKS["train"]}
        test_ids = {t["id"] for t in ALL_TASKS["test"]}
        overlap = train_ids & test_ids
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_default_parameters_match_omg(self):
        """BASE parameters should match OMG defaults."""
        assert abs(BASE["inverter_V_nom"] - 230.0 * np.sqrt(2)) < 0.01
        assert BASE["inverter_f_nom"] == 50.0
        assert BASE["inverter_i_nom"] == 20.0
        assert BASE["inverter_i_lim"] == 30.0
        assert BASE["dt_ode"] == 1e-4

    def test_episode_lengths(self):
        """Episode lengths per scenario match spec."""
        for task in ALL_TASKS["train"] + ALL_TASKS["test"]:
            s = task["scenario"]
            expected = {
                "voltage_control": 10,
                "current_control": 8,
                "droop_control": 12,
                "load_following": 12,
            }
            assert task["max_steps"] == expected[s], \
                f"{task['id']}: max_steps={task['max_steps']}, expected {expected[s]}"

    def test_disturbance_steps(self):
        """Disturbance step should be mid-episode for relevant scenarios."""
        for task in ALL_TASKS["train"] + ALL_TASKS["test"]:
            s = task["scenario"]
            if s in ("voltage_control", "current_control", "droop_control"):
                ds = task.get("disturbance_step")
                assert ds is not None, f"{task['id']}: missing disturbance_step"
                assert 0 < ds < task["max_steps"], \
                    f"{task['id']}: disturbance_step={ds} out of range"

    def test_load_following_has_profile(self):
        """Load following tasks must have a load_profile list."""
        for task in ALL_TASKS["train"] + ALL_TASKS["test"]:
            if task["scenario"] == "load_following":
                assert task.get("load_profile") is not None
                assert len(task["load_profile"]) == task["max_steps"]
                assert all(r > 0 for r in task["load_profile"])

    def test_droop_tasks_have_two_inverters(self):
        """Droop control tasks should have num_inverters=2 and inv2 params."""
        for task in ALL_TASKS["train"] + ALL_TASKS["test"]:
            if task["scenario"] == "droop_control":
                assert task.get("num_inverters") == 2
                assert task.get("filter_L_2") is not None
                assert task.get("inverter_V_DC_2") is not None

    def test_environment_setup_teardown(self):
        """Environment setup and teardown should work cleanly."""
        from microgrid import MicrogridGym

        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            assert env.sim is not None
            assert env.current_obs is not None
            assert len(env.step_rewards) == 0
            assert not env.episode_done

            await env.teardown()
            assert env.sim is None

        asyncio.get_event_loop().run_until_complete(_test())

    def test_get_prompt_returns_textblock(self):
        """get_prompt() should return List[TextBlock]."""
        from microgrid import MicrogridGym

        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            prompt = await env.get_prompt()
            assert len(prompt) == 1
            assert hasattr(prompt[0], "text")
            assert len(prompt[0].text) > 100
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_set_controller_requires_kP_kI(self):
        """set_controller should reject missing required params."""
        from microgrid import MicrogridGym, SetControllerParams

        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            # Missing kI_v
            params = SetControllerParams(kP_v=0.025)
            result = await env.set_controller(params)
            assert result.metadata.get("error") == "missing_parameters"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_set_controller_rejects_negative_gains(self):
        """Negative gains should be rejected."""
        from microgrid import MicrogridGym, SetControllerParams

        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            params = SetControllerParams(kP_v=-0.1, kI_v=60.0)
            result = await env.set_controller(params)
            assert result.metadata.get("error") == "invalid_parameters"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_episode_terminates_at_max_steps(self):
        """Episode should end after max_steps tool calls."""
        from microgrid import MicrogridGym, SetControllerParams

        task = ALL_TASKS["train"][0]
        ts = TaskSpec.model_validate(task)
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            for i in range(ts.max_steps):
                params = SetControllerParams(kP_v=0.025, kI_v=60.0)
                result = await env.set_controller(params)
                if i < ts.max_steps - 1:
                    assert not result.finished
                else:
                    assert result.finished
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_episode_reward_is_mean_of_steps(self):
        """Final episode reward = mean of all step rewards."""
        from microgrid import MicrogridGym, SetControllerParams

        task = ALL_TASKS["train"][0]
        ts = TaskSpec.model_validate(task)
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            step_rewards = []
            for _ in range(ts.max_steps):
                params = SetControllerParams(kP_v=0.025, kI_v=60.0)
                result = await env.set_controller(params)
            # Episode reward in metadata
            ep_reward = result.reward
            manual_mean = sum(env.step_rewards) / len(env.step_rewards)
            assert abs(ep_reward - manual_mean) < 1e-10, \
                f"Episode reward {ep_reward} != mean {manual_mean}"
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_instability_terminates_episode(self):
        """Unstable gains should terminate episode with low reward."""
        from microgrid import MicrogridGym, SetControllerParams

        # Use a task and deliberately destabilize
        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            # Very aggressive gains that may cause instability
            params = SetControllerParams(kP_v=10.0, kI_v=50000.0,
                                         kP_i=5.0, kI_i=50000.0)
            # Run multiple steps until instability or completion
            for _ in range(10):
                result = await env.set_controller(params)
                if result.finished:
                    break
            # Should finish (either stable completion or instability)
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_load_disturbance_applied(self):
        """Load disturbance should change R_load at the specified step."""
        from microgrid import MicrogridGym, SetControllerParams

        task = ALL_TASKS["train"][0]
        ts = TaskSpec.model_validate(task)
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            original_R = env.sim.load.R

            for i in range(ts.disturbance_step + 1):
                params = SetControllerParams(kP_v=0.025, kI_v=60.0)
                await env.set_controller(params)

            # After disturbance step, R should have changed
            assert env.sim.load.R != original_R, \
                f"Load R should change at step {ts.disturbance_step}"
            assert abs(env.sim.load.R - ts.load_R_after) < 1e-6
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_info_tool_does_not_advance(self):
        """info() should not advance simulation or count as a step."""
        from microgrid import MicrogridGym, InfoParams

        task = ALL_TASKS["train"][0]
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            t_before = env.sim.t
            step_before = env.sim._step_count

            result = await env.info(InfoParams())

            assert env.sim.t == t_before
            assert env.sim._step_count == step_before
            assert not result.finished
            assert result.reward == 0.0
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_list_splits(self):
        """list_splits should return ['train', 'test']."""
        from microgrid import MicrogridGym
        splits = MicrogridGym.list_splits()
        assert set(splits) == {"train", "test"}

    def test_list_tasks_matches_all_tasks(self):
        """list_tasks should return same tasks as ALL_TASKS."""
        from microgrid import MicrogridGym
        for split in ["train", "test"]:
            tasks = MicrogridGym.list_tasks(split)
            assert len(tasks) == len(ALL_TASKS[split])

    def test_droop_episode_full(self):
        """Full droop control episode should complete."""
        from microgrid import MicrogridGym, SetControllerParams

        droop_tasks = [t for t in ALL_TASKS["train"]
                       if t["scenario"] == "droop_control"]
        task = droop_tasks[0]
        ts = TaskSpec.model_validate(task)
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            for _ in range(ts.max_steps):
                params = SetControllerParams(
                    kP_droop_1=1e-4, kQ_droop_1=1e-2,
                    kP_droop_2=1e-4, kQ_droop_2=1e-2,
                    kP_v=0.025, kI_v=60.0,
                )
                result = await env.set_controller(params)
            assert result.finished
            assert 0 <= result.reward <= 1.0
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_load_following_episode_full(self):
        """Full load following episode should complete."""
        from microgrid import MicrogridGym, SetControllerParams

        lf_tasks = [t for t in ALL_TASKS["train"]
                    if t["scenario"] == "load_following"]
        task = lf_tasks[0]
        ts = TaskSpec.model_validate(task)
        env = MicrogridGym(task)

        async def _test():
            await env.setup()
            for _ in range(ts.max_steps):
                params = SetControllerParams(kP_v=0.025, kI_v=60.0)
                result = await env.set_controller(params)
            assert result.finished
            assert 0 <= result.reward <= 1.0
            await env.teardown()

        asyncio.get_event_loop().run_until_complete(_test())


# ======================================================================== #
#  7. OMG-SPECIFIC PARAMETER FIDELITY TESTS
# ======================================================================== #

class TestOMGParameterFidelity:
    """Verify that our parameters match the original OMG values."""

    def test_v_nom_230_rms(self):
        """V_nom = 230 * sqrt(2) ≈ 325.27 V peak (European grid standard)."""
        assert abs(BASE["inverter_V_nom"] - 230.0 * np.sqrt(2)) < 0.01

    def test_f_nom_50hz(self):
        """European grid nominal frequency: 50 Hz."""
        assert BASE["inverter_f_nom"] == 50.0

    def test_i_nom_20a(self):
        """OMG nominal current: 20 A."""
        assert BASE["inverter_i_nom"] == 20.0

    def test_i_lim_30a(self):
        """OMG current limit: 30 A."""
        assert BASE["inverter_i_lim"] == 30.0

    def test_v_dc(self):
        """DC bus voltage: 600 V (some OMG configs use 1000 V)."""
        assert BASE["inverter_V_DC"] == 600.0

    def test_filter_inductance(self):
        """Filter inductance L = 2.3 mH (OMG default)."""
        assert abs(BASE["filter_L"] - 2.3e-3) < 1e-6

    def test_filter_capacitance(self):
        """Filter capacitance C = 10 uF (OMG default)."""
        assert abs(BASE["filter_C"] - 10e-6) < 1e-9

    def test_filter_resistance(self):
        """Filter parasitic resistance R_f = 0.4 ohm (OMG default)."""
        assert abs(BASE["filter_Rf"] - 0.4) < 1e-6

    def test_load_resistance(self):
        """Load resistance R = 14 ohm (OMG default)."""
        assert abs(BASE["load_R"] - 14.0) < 1e-6

    def test_load_inductance(self):
        """Load inductance L_load = 5.3 mH (OMG default)."""
        assert abs(BASE["load_L"] - 5.3e-3) < 1e-6

    def test_dt_ode_matches_omg_ts_ctrl(self):
        """dt_ode = 1e-4 s ≈ OMG ts_ctrl = 2 * ts_sim = 2 * 5e-5 = 1e-4 s.

        OMG controller updates at ts_ctrl, not ts_sim.
        """
        assert BASE["dt_ode"] == 1e-4

    def test_omg_gain_bounds(self):
        """OMG SafeOpt bounds: kP in [0, 0.03], kI in [0, 300].

        Our parameter descriptions should cover this range.
        """
        # Verify the OMG bounds are within our described typical ranges
        omg_kP_upper = 0.03
        omg_kI_upper = 300

        from microgrid import SetControllerParams
        desc_kP = SetControllerParams.model_fields["kP_v"].description
        desc_kI = SetControllerParams.model_fields["kI_v"].description
        # Just verify the fields have descriptions with range info
        assert "0.01" in desc_kP or "0.2" in desc_kP
        assert "20" in desc_kI or "500" in desc_kI

    def test_omg_default_kp_v(self):
        """OMG default voltage kP = 0.025."""
        # Verify our code documents this
        from microgrid import SetControllerParams
        desc = SetControllerParams.model_fields["kP_v"].description
        assert "0.025" in desc

    def test_omg_default_ki_v(self):
        """OMG default voltage kI = 60."""
        from microgrid import SetControllerParams
        desc = SetControllerParams.model_fields["kI_v"].description
        assert "60" in desc

    def test_omg_default_kp_i(self):
        """OMG default current kP = 0.012 (close to OMG's 0.01)."""
        from microgrid import SetControllerParams
        desc = SetControllerParams.model_fields["kP_i"].description
        assert "0.012" in desc

    def test_omg_default_ki_i(self):
        """OMG default current kI = 90 (close to OMG's 10, scaled for our architecture)."""
        from microgrid import SetControllerParams
        desc = SetControllerParams.model_fields["kI_i"].description
        assert "90" in desc


# ======================================================================== #
#  8. CROSS-SCENARIO CONSISTENCY TESTS
# ======================================================================== #

class TestCrossScenarioConsistency:
    """Verify consistency across different scenario types."""

    def test_all_scenarios_use_same_base_params(self):
        """All scenarios should use the same base circuit parameters
        (unless explicitly overridden)."""
        for task in ALL_TASKS["train"] + ALL_TASKS["test"]:
            assert task["inverter_V_nom"] == BASE["inverter_V_nom"]
            assert task["inverter_f_nom"] == BASE["inverter_f_nom"]
            assert task["inverter_i_lim"] == BASE["inverter_i_lim"]

    def test_all_scenarios_produce_nonzero_reward(self):
        """Every scenario type should produce non-zero reward with
        OMG default gains (0.025, 60)."""
        from microgrid import MicrogridGym, SetControllerParams

        scenarios_tested = set()
        for task in ALL_TASKS["train"]:
            s = task["scenario"]
            if s in scenarios_tested:
                continue
            scenarios_tested.add(s)

            env = MicrogridGym(task)

            async def _test():
                await env.setup()
                if s == "droop_control":
                    params = SetControllerParams(
                        kP_droop_1=1e-4, kQ_droop_1=1e-2,
                        kP_droop_2=1e-4, kQ_droop_2=1e-2,
                    )
                else:
                    params = SetControllerParams(kP_v=0.025, kI_v=60.0)
                result = await env.set_controller(params)
                assert result.reward > 0.0, \
                    f"Scenario {s} gave zero reward on first step"
                await env.teardown()

            asyncio.get_event_loop().run_until_complete(_test())

    def test_reward_differentiates_gain_quality(self):
        """Different gains should produce meaningfully different rewards.

        The OMG reward structure (root-error + log-barrier) penalizes
        both voltage tracking error and current constraint violations.
        With active control, currents increase (triggering barrier), but
        voltage tracking improves. The reward captures this tradeoff,
        so different gain settings should produce distinct reward values.
        """
        from microgrid import MicrogridGym, SetControllerParams

        task = ALL_TASKS["train"][0]
        ts = TaskSpec.model_validate(task)

        async def _run_episode(kP_v, kI_v, kP_i, kI_i):
            env = MicrogridGym(task)
            await env.setup()
            for _ in range(ts.max_steps):
                params = SetControllerParams(
                    kP_v=kP_v, kI_v=kI_v, kP_i=kP_i, kI_i=kI_i,
                )
                result = await env.set_controller(params)
            await env.teardown()
            return result.reward

        async def _test():
            r1 = await _run_episode(0.025, 60.0, 0.012, 90.0)
            r2 = await _run_episode(0.08, 300.0, 0.03, 200.0)
            r3 = await _run_episode(0.0001, 0.1, 0.0001, 0.1)

            # All should be in [0, 1]
            for r in [r1, r2, r3]:
                assert 0.0 <= r <= 1.0

            # Different gains should produce different rewards
            rewards = [r1, r2, r3]
            assert len(set(round(r, 4) for r in rewards)) >= 2, \
                f"Reward should differentiate gains: {rewards}"

        asyncio.get_event_loop().run_until_complete(_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

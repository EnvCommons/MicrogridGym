# MicrogridGym

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/MicrogridGym)

## Description

MicrogridGym is an environment for tuning cascaded PI controllers and droop coefficients for three-phase power electronic inverters in microgrid configurations. Based on the physics from the [OpenModelica Microgrid Gym (OMG)](https://github.com/upb-lea/openmodelica-microgrid-gym) toolbox, it implements a pure Python simulation of inverters with LC output filters supplying RL loads.

## Capabilities

- Understanding three-phase AC power electronics and LC filter dynamics
- Tuning cascaded PI controller gains (voltage and current loops)
- Configuring droop control for multi-inverter power sharing
- Adapting controller parameters to load disturbances
- Reasoning about control stability and tracking performance tradeoffs

## Compute Requirements

MicrogridGym runs a lightweight numerical simulation (RK4 ODE integration) and requires minimal compute resources.

## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) (matching the original OMG toolbox license).

## Tasks

There are 126 tasks across 4 scenario types and 2 splits:

| Scenario | Description | Agent Action | Steps | Train | Test |
|----------|-------------|-------------|-------|-------|------|
| Voltage Control | Single inverter, maintain 3-phase output voltage with load step disturbance | kP_v, kI_v, kP_i, kI_i | 10 | 60 | 9 |
| Current Control | Single inverter, track current reference with load step disturbance | kP_v, kI_v, kP_i, kI_i | 8 | 15 | 6 |
| Droop Control | Two parallel inverters, proportional power sharing | Droop coefficients + PI gains | 12 | 12 | 6 |
| Load Following | Single inverter, time-varying load profile | kP_v, kI_v, kP_i, kI_i (per step) | 12 | 12 | 6 |

**Total: 99 train + 27 test = 126 tasks**

Each task defines a specific circuit configuration (filter inductance, capacitance, load resistance) and disturbance scenario. Train and test splits use different parameter values.

## Reward Structure

This is a dense, verifiable reward environment. Rewards are computed algorithmically at each step, following the [OMG reward formulation](https://github.com/upb-lea/openmodelica-microgrid-gym):

**Voltage tracking error** (per-phase root-error):

$$\text{voltage\_err} = \frac{1}{N}\sum_{t}\sum_{k=1}^{3} \sqrt{\frac{|V_{\text{ref},k} - V_{\text{actual},k}|}{V_{\text{nom}}}}$$

**Current tracking error** (per-phase root-error):

$$\text{current\_err} = \frac{1}{N}\sum_{t}\sum_{k=1}^{3} \sqrt{\frac{|i_{\text{ref},k} - i_{\text{actual},k}|}{i_{\text{lim}}}}$$

**Log-barrier current constraint penalty**:

$$\text{barrier} = -\frac{1}{N}\sum_{t}\sum_{k=1}^{3} \mu \cdot \ln\left(1 - \frac{\max(|i_k| - i_{\text{nom}},\, 0)}{i_{\text{lim}} - i_{\text{nom}}}\right)$$

where $\mu = 2$, $i_{\text{nom}} = 20\text{A}$, $i_{\text{lim}} = 30\text{A}$.

**Step reward** (mapped to [0, 1]):

$$\text{step\_reward} = \exp\left(-\frac{\text{voltage\_err} + \text{current\_err} + \text{barrier}}{3}\right)$$

The episode reward is the mean of all step rewards.

No LLM graders are used.

## Data

No external data files are required. All circuit parameters and task definitions are generated programmatically from physically meaningful parameter ranges based on the OMG toolbox defaults.

## Tools

Agents are given two tools:

- **`set_controller`**: Set controller parameters (PI gains and/or droop coefficients) and advance the simulation by one control interval (5ms). Returns voltage/current measurements, tracking error, and step reward.
- **`info`**: Reference documentation about the circuit topology, control architecture, parameter ranges, and tuning guidelines.

## Time Horizon

Each episode consists of 8-12 tool calls depending on the scenario type. The agent sets controller parameters at each step, observes the resulting voltage/current waveforms, and can adjust gains for the next step.

## Other Environment Requirements

There are no further environment requirements; MicrogridGym works out of the box with the OpenReward platform without any secrets.

## Safety

Agents in MicrogridGym interact with a numerical simulation of power electronics. The environment does not connect to real hardware or external systems. Unstable controller parameters cause clean simulation termination with zero reward, not real-world damage.

## Citations

```bibtex
@article{Heid2020OMG,
  author  = {Stefan Heid and Daniel Weber and Henrik Bode and Eyke H{\"u}llermeier and Oliver Wallscheid},
  title   = {{OMG}: A Scalable and Flexible Simulation and Testing Environment Toolbox for Intelligent Microgrid Control},
  journal = {Journal of Open Source Software},
  volume  = {5},
  number  = {54},
  pages   = {2435},
  year    = {2020},
  doi     = {10.21105/joss.02435}
}
```

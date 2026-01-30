"""Reactor models and solver helpers.

This module provides tools for simulating Continuous Stirred-Tank Reactors (CSTR).
It includes configuration data structures, right-hand side (RHS) function builders
for ODE solvers, and solver wrappers.

The CSTR model accounts for:
- Mass balance with inflow, outflow, and reaction.
- Energy balance with inflow, heat of reaction, and jacket cooling/heating.
- Jacket energy balance.

Note: The current implementation assumes a simple A -> B reaction stoichiometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from simpreactor.kinetics import PowerLawKinetics


@dataclass(frozen=True)
class CSTRConfiguration:
    """Physical configuration and properties of a CSTR.

    Attributes:
        volume: Reactor volume (m³).
        density: Reaction mixture density (kg/m³).
        heat_capacity: Reaction mixture specific heat capacity (J/(kg·K)).
        ua: Overall heat transfer coefficient * Area (W/K).
        jacket_mass: Mass of the cooling/heating jacket fluid (kg).
        jacket_heat_capacity: Jacket fluid specific heat capacity (J/(kg·K)).
    """

    volume: float
    density: float
    heat_capacity: float
    ua: float
    jacket_mass: float
    jacket_heat_capacity: float


def build_cstr_rhs(
    inflow_concentrations: Mapping[str, float],
    inflow_temperature: float,
    inflow_rate: float,
    kinetics: PowerLawKinetics,
    species_order: Sequence[str],
    reaction_enthalpy: float,
    configuration: CSTRConfiguration,
    jacket_inlet_temperature: float,
    jacket_heat_input: float,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build the Right-Hand Side (RHS) function for the CSTR ODE system.

    This function constructs the system of differential equations describing the
    dynamic behavior of a CSTR.

    Equations:
        Species Mass Balance:
            dC_i/dt = (q / V) * (C_in,i - C_i) + r_i

        Reactor Energy Balance:
            dT/dt = (q / V) * (T_in - T)
                  + (-dH_rxn * rate * V) / (rho * Cp * V)
                  - (UA * (T - Tj)) / (rho * Cp * V)

        Jacket Energy Balance:
            dTj/dt = (Q_in_jacket - UA * (Tj - T)) / (m_j * Cp_j)

    Where:
        q = inflow_rate
        V = volume
        rho = density
        Cp = heat_capacity
        UA = ua
        m_j = jacket_mass
        Cp_j = jacket_heat_capacity

    **Important Note on Stoichiometry:**
    This implementation currently assumes a stoichiometry where the first species
    in `species_order` is the reactant (consumed) and the second species (if present)
    is the product (generated).
        r_0 = -rate
        r_1 = +rate

    Args:
        inflow_concentrations: Map of species names to inflow concentrations (mol/m³).
        inflow_temperature: Temperature of the inflow stream (K).
        inflow_rate: Volumetric flow rate (m³/s).
        kinetics: Kinetic model to calculate reaction rate.
        species_order: Order of species in the state vector. index 0 is reactant.
        reaction_enthalpy: Enthalpy of reaction (J/mol). Exothermic < 0.
        configuration: Physical configuration of the reactor.
        jacket_inlet_temperature: Temperature of the jacket inlet (K). Currently unused in the model (batch jacket assumption).
        jacket_heat_input: External heat input to the jacket (W).

    Returns:
        A function `rhs(t, y)` compatible with `scipy.integrate.solve_ivp`.
        The state vector `y` contains: `[...concentrations, T, Tj]`.
    """
    volume = configuration.volume
    density = configuration.density
    heat_capacity = configuration.heat_capacity
    ua = configuration.ua
    jacket_mass = configuration.jacket_mass
    jacket_heat_capacity = configuration.jacket_heat_capacity

    inflow = np.array([inflow_concentrations.get(name, 0.0) for name in species_order])

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        concentrations = state[:-2]
        temperature = state[-2]
        jacket_temperature = state[-1]

        conc_map = {name: value for name, value in zip(species_order, concentrations, strict=False)}
        rate = kinetics.rate(conc_map, temperature)

        reaction_term = np.zeros_like(concentrations)
        reaction_term[0] = -rate
        if len(concentrations) > 1:
            reaction_term[1] = rate

        dcdt = (inflow_rate * (inflow - concentrations)) / volume + reaction_term

        heat_release = -reaction_enthalpy * rate
        dtdt = (
            inflow_rate * heat_capacity * (inflow_temperature - temperature)
            + heat_release * volume
            - ua * (temperature - jacket_temperature)
        ) / (density * heat_capacity * volume)

        d_tj_dt = (
            jacket_heat_input - ua * (jacket_temperature - temperature)
        ) / (jacket_mass * jacket_heat_capacity)

        return np.concatenate([dcdt, [dtdt, d_tj_dt]])

    return rhs


def solve_cstr(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    time_span: tuple[float, float],
    evaluation_times: np.ndarray,
) -> solve_ivp:
    """Solve the CSTR system of differential equations.

    Args:
        rhs: The right-hand side function generated by `build_cstr_rhs`.
        initial_state: Initial state vector `[...concentrations, T, Tj]`.
        time_span: Tuple (t_start, t_end) for the integration.
        evaluation_times: Array of time points at which to store the solution.

    Returns:
        Result object from `scipy.integrate.solve_ivp`.
    """
    return solve_ivp(
        rhs,
        time_span,
        initial_state,
        t_eval=evaluation_times,
        method="BDF",
    )

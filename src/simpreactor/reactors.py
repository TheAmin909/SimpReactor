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

from simpreactor.constants import R_GAS
from simpreactor.kinetics import KineticsModel, PowerLawKinetics
from simpreactor.thermo import ThermoInterface


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


@dataclass(frozen=True)
class PFRConfiguration:
    length: float
    diameter: float
    overall_heat_transfer_coefficient: float = 0.0  # U [W/m2/K]
    perimeter_temperature: float = 298.15  # T_jacket [K]

    @property
    def cross_sectional_area(self) -> float:
        return np.pi * (self.diameter / 2.0) ** 2

    @property
    def perimeter(self) -> float:
        return np.pi * self.diameter


def build_cstr_rhs(
    inflow_concentrations: Mapping[str, float],
    inflow_temperature: float,
    inflow_rate: float,
    kinetics: KineticsModel,
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

        # TODO: Generalize stoichiometry. Currently hardcoded A->B logic if 2 species
        reaction_term = np.zeros_like(concentrations)
        if len(concentrations) >= 1:
            reaction_term[0] = -rate # A consumes
        if len(concentrations) >= 2:
            reaction_term[1] = rate  # B produces

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


def build_pfr_rhs(
    kinetics: KineticsModel,
    thermo: ThermoInterface,
    species_order: Sequence[str],
    stoichiometry: Sequence[float],
    reaction_enthalpy: float,
    configuration: PFRConfiguration,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build the RHS for a steady-state PFR (d/dz).

    State vector: [F_1, F_2, ..., F_N, T, P]
    Independent variable: z (axial distance)
    """
    area = configuration.cross_sectional_area
    perimeter = configuration.perimeter
    U = configuration.overall_heat_transfer_coefficient
    Tj = configuration.perimeter_temperature
    stoich_arr = np.array(stoichiometry)

    def rhs(z: float, state: np.ndarray) -> np.ndarray:
        flows = state[:-2] # mol/s
        T = state[-2]
        P = state[-1]

        total_flow = np.sum(flows)
        # Calculate concentrations: Ci = (Fi / F_total) * (P / RT)
        # Volume flow v = F_total * R * T / P
        if total_flow <= 1e-12:
             # Avoid division by zero if flow stops (shouldn't happen in steady PFR usually)
             concentration_factor = 0.0
        else:
             concentration_factor = P / (R_GAS * T * total_flow)

        concentrations = flows * concentration_factor
        conc_map = {name: val for name, val in zip(species_order, concentrations, strict=False)}

        # 1. Reaction Rate
        r = kinetics.rate(conc_map, T) # mol/m3/s

        # dFi/dz = nu_i * r * Ac
        dFdz = stoich_arr * r * area

        # 2. Energy Balance
        # dT/dz = ( - sum(dH * r) * Ac - U * perim * (T - Tj) ) / sum(Fi * Cpi)

        # Heat of reaction term
        # If reaction_enthalpy is DeltaH_rxn [J/mol_rxn]
        heat_gen = -reaction_enthalpy * r * area # Watts/m

        heat_loss = U * perimeter * (T - Tj) # Watts/m

        # Mixture Heat Capacity
        # sum(Fi * Cpi)
        # We need individual Cp or mixture Cp?
        # sum(Fi * Cpi) = F_total * Cp_mix_molar

        # Get Cp_mix [J/mol/K] from thermo
        # Thermo expects mole fractions
        mole_fractions = flows / total_flow if total_flow > 0 else np.zeros_like(flows)
        mf_map = {name: val for name, val in zip(species_order, mole_fractions, strict=False)}

        cp_mix = thermo.heat_capacity(T, mf_map) # J/mol/K
        total_heat_capacity_flow = total_flow * cp_mix # (mol/s) * (J/mol/K) = W/K

        if total_heat_capacity_flow > 1e-12:
            dTdz = (heat_gen - heat_loss) / total_heat_capacity_flow
        else:
            dTdz = 0.0

        # 3. Pressure Drop (Ergun) - Ignored for MVP M1
        dPdz = 0.0

        return np.concatenate([dFdz, [dTdz, dPdz]])

    return rhs


def solve_pfr(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray, # [F0..., T0, P0]
    length: float,
    evaluation_points: int = 100,
) -> solve_ivp:
    z_span = (0.0, length)
    z_eval = np.linspace(0.0, length, evaluation_points)

    return solve_ivp(
        rhs,
        z_span,
        initial_state,
        t_eval=z_eval,
        method="BDF", # PFRs can be stiff
    )

"""Reactor models and solver helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from simpreactor.kinetics import PowerLawKinetics


@dataclass(frozen=True)
class CSTRConfiguration:
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
    return solve_ivp(
        rhs,
        time_span,
        initial_state,
        t_eval=evaluation_times,
        method="BDF",
    )

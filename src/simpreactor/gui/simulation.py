"""Simulation helpers for the GUI layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simpreactor.kinetics import ArrheniusKinetics, PowerLawKinetics
from simpreactor.reactors import CSTRConfiguration, build_cstr_rhs, solve_cstr


@dataclass(frozen=True)
class CSTRInputs:
    inflow_a: float
    inflow_b: float
    inflow_temperature: float
    inflow_rate: float
    volume: float
    density: float
    heat_capacity: float
    ua: float
    jacket_mass: float
    jacket_heat_capacity: float
    reaction_enthalpy: float
    jacket_inlet_temperature: float
    jacket_heat_input: float
    pre_exponential: float
    activation_energy: float
    exponent_a: float
    duration: float
    points: int


@dataclass(frozen=True)
class CSTRResult:
    time: np.ndarray
    a: np.ndarray
    b: np.ndarray
    temperature: np.ndarray
    jacket_temperature: np.ndarray


def run_cstr_simulation(inputs: CSTRInputs) -> CSTRResult:
    kinetics = PowerLawKinetics(
        arrhenius=ArrheniusKinetics(
            pre_exponential=inputs.pre_exponential,
            activation_energy=inputs.activation_energy,
        ),
        exponents={"A": inputs.exponent_a},
    )

    config = CSTRConfiguration(
        volume=inputs.volume,
        density=inputs.density,
        heat_capacity=inputs.heat_capacity,
        ua=inputs.ua,
        jacket_mass=inputs.jacket_mass,
        jacket_heat_capacity=inputs.jacket_heat_capacity,
    )

    rhs = build_cstr_rhs(
        inflow_concentrations={"A": inputs.inflow_a, "B": inputs.inflow_b},
        inflow_temperature=inputs.inflow_temperature,
        inflow_rate=inputs.inflow_rate,
        kinetics=kinetics,
        species_order=["A", "B"],
        reaction_enthalpy=inputs.reaction_enthalpy,
        configuration=config,
        jacket_inlet_temperature=inputs.jacket_inlet_temperature,
        jacket_heat_input=inputs.jacket_heat_input,
    )

    initial_state = np.array(
        [
            inputs.inflow_a,
            inputs.inflow_b,
            inputs.inflow_temperature,
            inputs.jacket_inlet_temperature,
        ]
    )
    time_span = (0.0, inputs.duration)
    evaluation_times = np.linspace(time_span[0], time_span[1], inputs.points)

    result = solve_cstr(rhs, initial_state, time_span, evaluation_times)

    return CSTRResult(
        time=result.t,
        a=result.y[0],
        b=result.y[1],
        temperature=result.y[2],
        jacket_temperature=result.y[3],
    )

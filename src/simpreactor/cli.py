"""Command-line entrypoints for SimpReactor."""

from __future__ import annotations

import json
from typing import Annotated

import numpy as np
import typer

from simpreactor.kinetics import ArrheniusKinetics, PowerLawKinetics
from simpreactor.reactors import CSTRConfiguration, build_cstr_rhs, solve_cstr

app = typer.Typer(add_completion=False)


@app.command()
def cstr_demo(
    duration: Annotated[float, typer.Option(help="Simulation duration (s).")]=200.0,
    points: Annotated[int, typer.Option(help="Number of output points.")]=200,
) -> None:
    """Run a simple CSTR demo with A -> B conversion."""
    kinetics = PowerLawKinetics(
        arrhenius=ArrheniusKinetics(pre_exponential=2.4e3, activation_energy=85000.0),
        exponents={"A": 1.0},
    )
    config = CSTRConfiguration(
        volume=1.0,
        density=1000.0,
        heat_capacity=4200.0,
        ua=1800.0,
        jacket_mass=45.0,
        jacket_heat_capacity=4200.0,
    )

    rhs = build_cstr_rhs(
        inflow_concentrations={"A": 1.5, "B": 0.0},
        inflow_temperature=350.0,
        inflow_rate=0.15,
        kinetics=kinetics,
        species_order=["A", "B"],
        reaction_enthalpy=-75000.0,
        configuration=config,
        jacket_inlet_temperature=320.0,
        jacket_heat_input=0.0,
    )

    initial_state = np.array([1.5, 0.0, 350.0, 320.0])
    time_span = (0.0, duration)
    evaluation_times = np.linspace(time_span[0], time_span[1], points)

    result = solve_cstr(rhs, initial_state, time_span, evaluation_times)
    final_state = result.y[:, -1]

    payload = {
        "time": result.t.tolist(),
        "A": result.y[0].tolist(),
        "B": result.y[1].tolist(),
        "T": result.y[2].tolist(),
        "Tj": result.y[3].tolist(),
        "final": {
            "A": float(final_state[0]),
            "B": float(final_state[1]),
            "T": float(final_state[2]),
            "Tj": float(final_state[3]),
        },
    }
    typer.echo(json.dumps(payload, indent=2))

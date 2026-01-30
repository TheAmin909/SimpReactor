"""Command-line entrypoints for SimpReactor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Dict

import numpy as np
import typer

from simpreactor.kinetics import (
    ArrheniusKinetics,
    LHHWKinetics,
    PowerLawKinetics,
)
from simpreactor.thermo import IdealGasThermo, SpeciesProperties
from simpreactor.persistence import sqlite_store
from simpreactor.reactors import (
    CSTRConfiguration,
    PFRConfiguration,
    build_cstr_rhs,
    build_pfr_rhs,
    solve_cstr,
    solve_pfr,
)

app = typer.Typer(add_completion=False)


def _parse_kinetics(data: Dict[str, Any]) -> Any:
    k_type = data.get("type", "power_law").lower()
    arr_data = data["arrhenius"]
    arrhenius = ArrheniusKinetics(
        pre_exponential=float(arr_data["A"]),
        activation_energy=float(arr_data["Ea"]),
    )

    if k_type == "power_law":
        return PowerLawKinetics(
            arrhenius=arrhenius, exponents=data.get("exponents", {})
        )
    elif k_type == "lhhw":
        ads_consts = {}
        for sp, p in data.get("adsorption_constants", {}).items():
            ads_consts[sp] = ArrheniusKinetics(
                pre_exponential=float(p["A"]), activation_energy=float(p["Ea"])
            )
        return LHHWKinetics(
            arrhenius=arrhenius,
            numerator_exponents=data.get("numerator_exponents", {}),
            adsorption_constants=ads_consts,
            denominator_exponent=float(data.get("denominator_exponent", 1.0)),
        )
    else:
        raise ValueError(f"Unknown kinetics type: {k_type}")


def _parse_thermo(data: Dict[str, Any]) -> IdealGasThermo:
    props = {}
    for name, p in data.get("species", {}).items():
        props[name] = SpeciesProperties(
            molecular_weight=float(p["mw"]),
            heat_capacity=float(p["cp"]),
            heat_of_formation=float(p["h_form"]),
        )
    return IdealGasThermo(props)


@app.command()
def run(
    config_file: Annotated[
        Path, typer.Argument(help="Path to JSON configuration file.")
    ],
    output: Annotated[
        Path | None, typer.Option(help="Path to save output JSON.")
    ] = None,
) -> None:
    """Run a reactor simulation from a config file."""
    with open(config_file, "r") as f:
        config = json.load(f)

    reactor_type = config.get("reactor_type", "PFR").upper()

    # Common parsing
    kinetics = _parse_kinetics(config["kinetics"])
    thermo = _parse_thermo(config["thermo"])
    stoich_map = config["stoichiometry"]
    species_order = list(config["thermo"]["species"].keys()) # Ensure consistency
    stoichiometry = [stoich_map.get(s, 0.0) for s in species_order]

    # Calculate reaction enthalpy at reference T (simplified)
    # DeltaH = sum(nu_i * h_f_i)
    reaction_enthalpy = 0.0
    for s, coeff in stoich_map.items():
        # Check if species exists in thermo
        if s in thermo.properties:
            reaction_enthalpy += coeff * thermo.properties[s].heat_of_formation

    if reactor_type == "PFR":
        r_conf = config["reactor_config"]
        pfr_config = PFRConfiguration(
            length=float(r_conf["length"]),
            diameter=float(r_conf["diameter"]),
            overall_heat_transfer_coefficient=float(r_conf.get("U", 0.0)),
            perimeter_temperature=float(r_conf.get("Tw", 298.15)),
        )

        inlet = config["inlet"]
        T0 = float(inlet["T"])
        P0 = float(inlet["P"])
        flow_rate = float(inlet["flow_rate"]) # mol/s
        comp = inlet["composition"] # mole fractions

        # Initial State: [F1, ..., FN, T, P]
        flows = []
        for s in species_order:
            flows.append(flow_rate * comp.get(s, 0.0))

        initial_state = np.array(flows + [T0, P0])

        rhs = build_pfr_rhs(
            kinetics=kinetics,
            thermo=thermo,
            species_order=species_order,
            stoichiometry=stoichiometry,
            reaction_enthalpy=reaction_enthalpy,
            configuration=pfr_config,
        )

        points = config.get("solver", {}).get("points", 100)
        result = solve_pfr(rhs, initial_state, pfr_config.length, points)

        # Prepare output
        data = {
            "z": result.t.tolist(),
            "species": {},
            "T": result.y[-2].tolist(),
            "P": result.y[-1].tolist()
        }
        for i, s in enumerate(species_order):
            data["species"][s] = result.y[i].tolist() # Molar flows

        json_output = json.dumps(data, indent=2)
        typer.echo(json_output)

        if output:
            with open(output, "w") as f:
                f.write(json_output)

    else:
        typer.echo(f"Reactor type {reactor_type} not supported in generic run command yet.")


@app.command()
def cstr_demo(
    duration: Annotated[float, typer.Option(help="Simulation duration (s).")]=200.0,
    points: Annotated[int, typer.Option(help="Number of output points.")]=200,
    project_file: Annotated[
        Path | None,
        typer.Option(help="Optional .crdproj file to persist results."),
    ] = None,
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

    if project_file is not None:
        connection = sqlite_store.connect(project_file)
        sqlite_store.ensure_schema(connection)
        project_id = sqlite_store.create_project(
            connection,
            name="CSTR demo",
            notes="Autogenerated from SimpReactor CLI demo.",
        )
        run_id = sqlite_store.save_run(
            connection,
            project_id=project_id,
            model={
                "reactor": "CSTR",
                "species_order": ["A", "B"],
                "inflow_concentrations": {"A": 1.5, "B": 0.0},
                "inflow_temperature": 350.0,
                "inflow_rate": 0.15,
                "reaction_enthalpy": -75000.0,
                "configuration": config.__dict__,
                "kinetics": {
                    "type": "power_law",
                    "arrhenius": kinetics.arrhenius.__dict__,
                    "exponents": kinetics.exponents,
                },
            },
            solver={"method": "BDF", "points": points, "duration_s": duration},
            manifest={"final_state": payload["final"]},
            duration_ms=int(duration * 1000.0),
        )
        sqlite_store.save_profile(
            connection,
            run_id=run_id,
            x_values=payload["time"],
            series={
                "A": payload["A"],
                "B": payload["B"],
                "T": payload["T"],
                "Tj": payload["Tj"],
            },
            units={"A": "mol/L", "B": "mol/L", "T": "K", "Tj": "K"},
        )
        connection.close()

    typer.echo(json.dumps(payload, indent=2))

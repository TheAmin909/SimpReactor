"""Ideal gas thermodynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from simpreactor.constants import R_GAS
from simpreactor.thermo.base import ThermoInterface


@dataclass(frozen=True)
class SpeciesProperties:
    molecular_weight: float  # kg/mol
    heat_capacity: float  # J/mol/K (constant for MVP)
    heat_of_formation: float  # J/mol


class IdealGasThermo(ThermoInterface):
    """Ideal gas thermodynamics with constant Cp and ideal mixing."""

    def __init__(self, properties: Mapping[str, SpeciesProperties]):
        self.properties = properties

    def heat_capacity(self, temperature: float, composition: Mapping[str, float]) -> float:
        """Calculate molar heat capacity of the mixture."""
        # composition is mole fractions or concentrations?
        # Typically thermo uses mole fractions.
        # But simulation might track concentrations [mol/m3].
        # For ideal gas, Cp is independent of P.
        # Assuming composition is mole fractions for intensive properties.
        # If input is concentrations, we might need to normalize.
        # Let's assume composition is normalized mole fractions for now.

        cp_mix = 0.0
        for species, fraction in composition.items():
            props = self.properties.get(species)
            if props:
                cp_mix += fraction * props.heat_capacity
        return cp_mix

    def enthalpy(self, temperature: float, composition: Mapping[str, float]) -> float:
        """Calculate molar enthalpy of the mixture."""
        h_mix = 0.0
        for species, fraction in composition.items():
            props = self.properties.get(species)
            if props:
                # H(T) = H_form + Cp * (T - T_ref)
                # Assuming T_ref for H_form is 298.15 K
                h_i = props.heat_of_formation + props.heat_capacity * (temperature - 298.15)
                h_mix += fraction * h_i
        return h_mix

    def density(self, temperature: float, pressure: float, composition: Mapping[str, float]) -> float:
        """Calculate density using Ideal Gas Law."""
        # rho = P / (R * T) * MW_avg
        # returns kg/m^3
        mw_avg = 0.0
        for species, fraction in composition.items():
            props = self.properties.get(species)
            if props:
                mw_avg += fraction * props.molecular_weight

        molar_density = pressure / (R_GAS * temperature) # mol/m^3
        return molar_density * mw_avg

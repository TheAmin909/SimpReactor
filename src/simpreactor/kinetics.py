"""Kinetics helpers and rate expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from simpreactor.constants import R_GAS


class KineticsModel(Protocol):
    def rate(self, concentrations: Mapping[str, float], temperature: float) -> float:
        """Calculate reaction rate given concentrations and temperature."""
        ...


@dataclass(frozen=True)
class ArrheniusKinetics:
    pre_exponential: float
    activation_energy: float

    def rate_constant(self, temperature: float) -> float:
        return self.pre_exponential * np.exp(-self.activation_energy / (R_GAS * temperature))


@dataclass(frozen=True)
class PowerLawKinetics:
    arrhenius: ArrheniusKinetics
    exponents: Mapping[str, float]

    def rate(self, concentrations: Mapping[str, float], temperature: float) -> float:
        k = self.arrhenius.rate_constant(temperature)
        rate = k
        for species, exponent in self.exponents.items():
            rate *= concentrations.get(species, 0.0) ** exponent
        return rate


@dataclass(frozen=True)
class LHHWKinetics:
    """Langmuir-Hinshelwood / Eley-Rideal kinetics.

    Rate = (k * product(C_i^alpha_i)) / (1 + sum(K_j * C_j))^m
    """
    arrhenius: ArrheniusKinetics
    numerator_exponents: Mapping[str, float]
    adsorption_constants: Mapping[str, ArrheniusKinetics]
    denominator_exponent: float = 1.0

    def rate(self, concentrations: Mapping[str, float], temperature: float) -> float:
        # Numerator
        k = self.arrhenius.rate_constant(temperature)
        numerator = k
        for species, exponent in self.numerator_exponents.items():
            numerator *= concentrations.get(species, 0.0) ** exponent

        # Denominator
        denominator_sum = 1.0
        for species, ads_params in self.adsorption_constants.items():
            k_ads = ads_params.rate_constant(temperature)
            denominator_sum += k_ads * concentrations.get(species, 0.0)

        return numerator / (denominator_sum ** self.denominator_exponent)

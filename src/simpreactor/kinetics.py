"""Kinetics helpers and rate expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

R_GAS = 8.31446261815324


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

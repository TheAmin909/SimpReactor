"""Base interface for thermodynamic models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping


class ThermoInterface(ABC):
    """Abstract base class for thermodynamic property packages."""

    @abstractmethod
    def heat_capacity(self, temperature: float, composition: Mapping[str, float]) -> float:
        """Calculate mixture heat capacity (J/mol/K or J/kg/K depending on basis)."""
        pass

    @abstractmethod
    def enthalpy(self, temperature: float, composition: Mapping[str, float]) -> float:
        """Calculate mixture enthalpy (J/mol or J/kg)."""
        pass

    @abstractmethod
    def density(self, temperature: float, pressure: float, composition: Mapping[str, float]) -> float:
        """Calculate mixture density (kg/m^3 or mol/m^3)."""
        pass

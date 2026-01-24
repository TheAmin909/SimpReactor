"""Data structures for species and reactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Species:
    name: str
    formula: str
    phase: str = "gas"


@dataclass(frozen=True)
class Reaction:
    name: str
    stoichiometry: Mapping[str, float]
    reversible: bool = False

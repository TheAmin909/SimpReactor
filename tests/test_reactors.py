import unittest
import numpy as np
from simpreactor.reactors import PFRConfiguration, build_pfr_rhs, solve_pfr
from simpreactor.kinetics import PowerLawKinetics, ArrheniusKinetics
from simpreactor.thermo import IdealGasThermo, SpeciesProperties

class TestPFR(unittest.TestCase):
    def setUp(self):
        self.props = {
            "A": SpeciesProperties(0.050, 50.0, 0.0),
            "B": SpeciesProperties(0.050, 50.0, -10000.0),
        }
        self.thermo = IdealGasThermo(self.props)
        self.kinetics = PowerLawKinetics(
            arrhenius=ArrheniusKinetics(1.0, 0.0), # k=1
            exponents={"A": 1.0}
        )
        self.config = PFRConfiguration(
            length=1.0,
            diameter=0.1,
            overall_heat_transfer_coefficient=0.0,
            perimeter_temperature=300.0
        )
        self.species_order = ["A", "B"]
        self.stoichiometry = [-1.0, 1.0]

    def test_pfr_isothermal_analytical(self):
        # A -> B, k=1
        # PFR Analytical: FA = FA0 * exp(-k * tau)
        T0 = 300.0
        P0 = 101325.0
        u = 0.5
        area = self.config.cross_sectional_area
        Q = u * area
        CA0 = P0 / (8.31446 * T0)
        FA0 = CA0 * Q

        initial_state = np.array([FA0, 0.0, T0, P0])

        rhs = build_pfr_rhs(
            kinetics=self.kinetics,
            thermo=self.thermo,
            species_order=self.species_order,
            stoichiometry=self.stoichiometry,
            reaction_enthalpy=0.0,
            configuration=self.config
        )

        result = solve_pfr(rhs, initial_state, self.config.length)

        FA_final = result.y[0, -1]
        tau = self.config.length / u # 1.0 / 0.5 = 2.0 s
        FA_analytical = FA0 * np.exp(-1.0 * tau)

        self.assertAlmostEqual(FA_final, FA_analytical, delta=FA0 * 0.01) # 1% error tolerance

    def test_pfr_zero_length(self):
        # Limit test: L -> 0 means conversion -> 0
        config = PFRConfiguration(length=1e-6, diameter=0.1)
        rhs = build_pfr_rhs(
            kinetics=self.kinetics,
            thermo=self.thermo,
            species_order=self.species_order,
            stoichiometry=self.stoichiometry,
            reaction_enthalpy=0.0,
            configuration=config
        )
        initial_state = np.array([1.0, 0.0, 300.0, 101325.0])
        result = solve_pfr(rhs, initial_state, config.length)

        FA_final = result.y[0, -1]
        self.assertAlmostEqual(FA_final, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()

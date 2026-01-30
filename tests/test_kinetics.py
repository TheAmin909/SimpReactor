import unittest
import numpy as np
from simpreactor.kinetics import PowerLawKinetics, LHHWKinetics, ArrheniusKinetics

class TestKinetics(unittest.TestCase):
    def test_power_law(self):
        # r = k * C_A^1
        arr = ArrheniusKinetics(pre_exponential=10.0, activation_energy=0.0)
        kin = PowerLawKinetics(arrhenius=arr, exponents={"A": 1.0})

        rate = kin.rate({"A": 2.0}, 300.0)
        self.assertAlmostEqual(rate, 20.0)

    def test_lhhw_inhibition(self):
        # r = k*CA / (1 + K*CA)^2
        arr = ArrheniusKinetics(10.0, 0.0)
        k_ads = ArrheniusKinetics(1.0, 0.0) # K = 1

        kin = LHHWKinetics(
            arrhenius=arr,
            numerator_exponents={"A": 1.0},
            adsorption_constants={"A": k_ads},
            denominator_exponent=2.0
        )

        # At low CA (0.01), K*CA << 1, rate ~ k*CA
        r_low = kin.rate({"A": 0.01}, 300.0)
        # 10 * 0.01 / (1 + 0.01)^2 ~ 0.1
        self.assertAlmostEqual(r_low, 0.1 / (1.01**2), places=3)

        # At high CA (100), K*CA >> 1, rate ~ k*CA / (K*CA)^2 = k/(K^2 * CA) -> decreasing
        r_high = kin.rate({"A": 100.0}, 300.0)
        # 10 * 100 / (1 + 100)^2 = 1000 / 10201 ~ 0.098

        self.assertTrue(r_high < 1.0) # Definitely inhibited compared to linear 1000
        self.assertTrue(r_high > 0.0)

if __name__ == '__main__':
    unittest.main()

import unittest
import tracemalloc
from analytics.continuum_modelling.microkinetic_modelling import MicroKineticModel

class TestMicrokinetic(unittest.TestCase):

    top_model = MicroKineticModel(pH=7, rotation_rate=1600, bulk_concentration=0.008, temperature=298)

    def test_top_model_attributes(self):
        self.assertTrue(self.top_model.pH == 7)
        self.assertTrue(self.top_model.rotation_rate == 1600)
        self.assertTrue(self.top_model.bulk_concentration == 0.008)
        self.assertTrue(self.top_model.temperature == 298)
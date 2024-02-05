import unittest
import tracemalloc
import pandas as pd
import numpy as np
from analytics.quantum_chemistry.gaussian import GaussianParser
tracemalloc.start()


class TestGaussianParser(unittest.TestCase):
    """
    Test class for the GaussianParser class
    """
    pedot_log = GaussianParser("../test_trajectories/pedot_raman/step1.log")
    bbl_log = GaussianParser("../test_trajectories/bbl/step3.log")

    def test_get_bonds_from_log(self):
        result = self.bbl_log.get_bonds_from_log()
        self.assertTrue(type(result))
        self.assertTrue(result.iloc[0, 0] == 1)
        self.assertTrue(result.iloc[10, 2] == 1.3935)
        self.assertTrue(result.iloc[30, 3] == "C")
        self.assertTrue(result.iloc[50, 1] == 41)

    def test_get_bonds_from_coordinates(self):
        result = self.bbl_log.get_bonds_from_coordinates()
        self.assertTrue(type(result))
        self.assertTrue(result.iloc[0, 0] == 1)
        self.assertTrue(result.iloc[10, 2] == 1.3935)
        self.assertTrue(result.iloc[30, 3] == "C")
        self.assertTrue(result.iloc[50, 1] == 41)

    def test_get_bonds_from_coordinates_pre_opt(self):
        result = self.bbl_log.get_bonds_from_coordinates(pre_optimisation=True)
        self.assertTrue(type(result))
        self.assertTrue(result.iloc[0, 0] == 1)
        self.assertTrue(result.iloc[10, 2] == 1.3935)
        self.assertTrue(result.iloc[30, 3] == "C")
        self.assertTrue(result.iloc[50, 1] == 41)

    def test_two_bond_methods(self):
        result1 = self.bbl_log.get_bonds_from_log().sort_values(["atom_id_1", "atom_id_2"])
        result2 = self.bbl_log.get_bonds_from_coordinates().sort_values(["atom_id_1", "atom_id_2"])
        pd.testing.assert_frame_equal(result1.round(3), result2.round(3))

    def test_charge_pedot(self):
        self.assertTrue(self.pedot_log.charge == 0)

    def test_charge_bbl(self):
        self.assertTrue(self.bbl_log.charge == -2)

    def test_charge_bbl_double_digit(self):
        bbl_log_double_digit_charge = GaussianParser("../test_trajectories/bbl/step1.log")
        self.assertTrue(bbl_log_double_digit_charge.charge == -12)

    def test_multiplicity_pedot(self):
        self.assertTrue(self.pedot_log.multiplicity == 1)

    def test_multiplicity_bbl(self):
        self.assertTrue(self.bbl_log.multiplicity == 3)

    def test_keywords(self):
        self.assertTrue(type(self.pedot_log.keywords) == list)

    def test_raman_pedot(self):
        self.assertTrue(self.pedot_log.raman is True)

    def test_raman_bbl(self):
        self.assertTrue(self.bbl_log.raman is False)

    def test_opt(self):
        self.assertTrue(self.pedot_log.opt is True)

    def test_complete(self):
        self.assertTrue(self.pedot_log.complete is True)

    def test_atomcount_pedot(self):
        self.assertTrue(self.pedot_log.atomcount == 28)

    def test_atomcount_bbl(self):
        self.assertTrue(self.bbl_log.atomcount == 140)

    def test_raman_frequencies_pedot(self):
        raman_frequencies = self.pedot_log.get_raman_frequencies()
        self.assertTrue(type(raman_frequencies) == pd.DataFrame)
        self.assertTrue(len(raman_frequencies) == 19)

    def test_raman_frequencies_bbl(self):
        with self.assertRaises(ValueError):
            self.bbl_log.get_raman_frequencies()

    def test_raman_spectra(self):
        raman_spectra = self.pedot_log.get_raman_spectra()
        self.assertTrue(len(raman_spectra) == 2000)

    def test_energy_pedot(self):
        energy = self.pedot_log.energy
        self.assertTrue(energy == -4096904.424959145)

    def test_energy_bbl(self):
        energy = self.bbl_log.energy
        self.assertTrue(energy == -12531769.403127551)

    def test_energy_bbl_no_SCF(self):
        energy = GaussianParser("../test_trajectories/bbl/step1_no_scf.log").energy
        self.assertTrue(energy is None)

    def test_functional_pedot(self):
        func = self.pedot_log.functional
        self.assertTrue(func == "B3LYP")

    def test_functional_bbl(self):
        func = self.bbl_log.functional
        self.assertTrue(func == "WB97XD")

    def test_basis_pedot(self):
        basis = self.pedot_log.basis
        self.assertTrue(basis == "6-311g")

    def test_basis_bbl(self):
        basis = self.bbl_log.basis
        self.assertTrue(basis == "6-311(d,p)")

    def test_unrestricted(self):
        unrestricted = self.pedot_log.unrestricted
        self.assertTrue(unrestricted is False)

    def test_atoms(self):
        atoms = self.pedot_log.atoms
        self.assertTrue(atoms == ['C', 'C', 'C', 'S', 'C', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C',
                                  'C', 'S', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'])

    def test_heavy_atoms(self):
        heavyatoms = self.pedot_log.heavyatoms
        self.assertTrue(heavyatoms == ['C', 'C', 'C', 'S', 'C', 'O', 'O', 'C', 'C', 'C', 'C', 'C', 'C', 'S', 'O', 'O',
                                       'C', 'C'])

    def test_heavy_atom_count_pedot(self):
        count = self.pedot_log.heavyatomcount
        self.assertTrue(count == 18)

    def test_heavy_atom_count_bbl(self):
        count = self.bbl_log.heavyatomcount
        self.assertTrue(count == 110)

    def test_get_mullikens_pedot(self):
        charges = self.pedot_log.get_mulliken_charges()
        self.assertTrue(charges['atom_id'].tolist() == [i for i in range(1, 29)])
        self.assertTrue(charges['element'].tolist() == self.pedot_log.atoms)

    def test_get_mullikens_bbl(self):
        charges = self.bbl_log.get_mulliken_charges()
        self.assertTrue(charges['atom_id'].tolist() == [i for i in range(1, 141)])
        self.assertTrue(charges['element'].tolist() == self.bbl_log.atoms)

    def test_get_mullikens_heavies(self):
        charges = self.pedot_log.get_mulliken_charges(heavy_atoms=True)
        self.assertTrue(charges['element'].tolist() == self.pedot_log.heavyatoms)
        self.assertTrue(charges['partial_charge'].tolist() == [0.360925, 0.321741, -0.335369, 0.413119, -0.26947,
                                                               -0.505062, -0.511759, 0.258057, 0.267784, -0.335384,
                                                               0.321773, 0.360884, -0.26943, 0.413122, -0.511813,
                                                               -0.505008, 0.268105, 0.257783])

    def test_esp_pedot(self):
        esp = self.pedot_log.esp
        self.assertTrue(esp is False)

    def test_esp_bbl(self):
        esp = self.bbl_log.esp
        self.assertTrue(esp is True)

    def test_esp_partials_pedot(self):
        with self.assertRaises(ValueError):
            self.pedot_log.get_esp_charges()

    def test_esp_partials_bbl(self):
        charges = self.bbl_log.get_esp_charges()
        self.assertTrue(type(charges) == pd.DataFrame)
        self.assertTrue(charges['partial_charge'].tolist()[0] == -0.043317)
        self.assertTrue(charges['partial_charge'].tolist()[5] == 0.625322)
        self.assertTrue(charges['partial_charge'].tolist()[10] == -0.112575)
        self.assertTrue(charges['partial_charge'].tolist()[15] == -0.142594)

    def test_esp_partials_bbl_heavies(self):
        charges = self.bbl_log.get_esp_charges(heavy_atoms=True)
        self.assertTrue(type(charges) == pd.DataFrame)
        self.assertTrue(charges['element'].tolist() == self.bbl_log.heavyatoms)
        self.assertTrue(charges['partial_charge'].tolist()[0] == 0.044489)
        self.assertTrue(charges['partial_charge'].tolist()[5] == -0.284547)
        self.assertTrue(charges['partial_charge'].tolist()[10] == 0.049955)
        self.assertTrue(charges['partial_charge'].tolist()[15] == -0.080793)

    def test_get_coordinates_pedot(self):
        coordinates = self.pedot_log.get_coordinates()
        self.assertTrue(coordinates['x'].to_list() == [-2.896713, -1.906019, -0.602732, -0.582447, -2.389927,
                                                       -4.265714, -2.257699, -4.571961, -3.656809, -2.920781,
                                                       -5.613404, -4.452347, -3.755911, -3.846831, 0.602789,
                                                       1.906062, 2.896732, 2.389966, 0.582486, 2.257544,
                                                       4.265750, 3.657369, 4.571336, 3.846797, 3.758349,
                                                       4.449836, 5.613228, 2.920821])
        self.assertTrue(coordinates['y'].to_list() == [-0.992769, 0.044153, -0.383270, -2.223480, -2.249765,
                                                       -0.700007, 1.396997, 0.678906, 1.651647, -3.182215,
                                                       0.831367, 0.768250, 1.546480, 2.680598, 0.383293,
                                                       -0.044139, 0.992787, 2.249771, 2.223486, -1.396968,
                                                       0.699883, -1.652235, -0.678299, -2.680703, -1.548920,
                                                       -0.765647, -0.831226, 3.182215])
        self.assertTrue(coordinates['z'].to_list() == [-0.031111, -0.012812, -0.011038, -0.044588, -0.049912,
                                                       -0.050345, 0.022211, 0.360569, -0.359029, -0.067273,
                                                       0.091302, 1.442186, -1.440908, -0.067731, 0.012085,
                                                       0.013091, 0.031796, 0.051678, 0.047337, -0.022861,
                                                       0.049976, 0.355428, -0.364014, 0.062057, 1.437312,
                                                       -1.445597, -0.096827, 0.069261])

    def test_get_coordinates_pedot_pre_eq(self):
        coordinates = self.pedot_log.get_coordinates(pre_optimisation=True).round(3)
        self.assertTrue(coordinates['x'].to_list()[0] == -2.98)
        self.assertTrue(coordinates['x'].to_list()[1] == -2.00)
        self.assertTrue(coordinates['x'].to_list()[2] == -0.632)

    def test_get_coordinates_pedot_heavies(self):
        coordinates = self.pedot_log.get_coordinates(heavy_atoms=True)
        self.assertTrue(len(coordinates) == self.pedot_log.heavyatomcount)

    def test_get_coordinates_bbl_heavies(self):
        coordinates = self.bbl_log.get_coordinates(heavy_atoms=True)
        self.assertTrue(len(coordinates) == self.bbl_log.heavyatomcount)

    def test_get_coordinates_bbl(self):
        coordinates = self.bbl_log.get_coordinates()
        self.assertTrue(len(coordinates) == self.bbl_log.atomcount)
        self.assertTrue(coordinates['x'].iloc[0] == 1.817506)

    def test_get_mulliken_charges_with_coordinates_pedot(self):
        data = self.pedot_log.get_mulliken_charges(with_coordinates=True)
        self.assertTrue(type(data) == pd.DataFrame)

    def test_get_mulliken_charges_with_coordinates_bbl(self):
        data = self.pedot_log.get_mulliken_charges(with_coordinates=True, heavy_atoms=True)
        self.assertTrue(type(data) == pd.DataFrame)

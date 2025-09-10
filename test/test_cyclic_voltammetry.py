from Materials_Data_Analytics.experiment_modelling.cyclic_voltammetry import CyclicVoltammogram
import unittest
import pandas as pd
from Materials_Data_Analytics.materials.electrolytes import Electrolyte
from Materials_Data_Analytics.materials.ions import Cation, Anion  
from Materials_Data_Analytics.materials.solvents import Solvent
import plotly.express as px
import base64
import mimetypes
from plotly import graph_objects as go
from copy import copy
from datetime import datetime as dt
from copy import deepcopy

class TestBiologic1(unittest.TestCase):

    def setUp(self):
        """
       Reading in a BBL CV.
        """
        self.na = Cation('Na+')
        self.cl = Anion('Cl-')
        self.water = Solvent('H2O')

        self.electrolyte = Electrolyte(
            cation=self.na, 
            anion=self.cl, 
            solvent=self.water, 
            pH=7, 
            temperature=298, 
            concentrations={self.na: 1, self.cl: 1}
        )

        self.cv = CyclicVoltammogram.from_biologic(
            potential_reference='Ag/AgCl',
            path='test_trajectories/cyclic_voltammetry/biologic1.txt', 
            electrolyte=self.electrolyte, 
            metadata={'scan_rate': 5, 'instrument': 'Biologic'}
        )

        self.cv_she = deepcopy(self.cv).set_potential_reference('she')
        self.cv_rhe = deepcopy(self.cv).set_potential_reference('rhe', open_circuit_potential=1)
        self.cv_she_agagcl = deepcopy(self.cv_she).set_potential_reference('agagcl')
        self.cv_agagcl_rhe = deepcopy(self.cv).set_potential_reference('rhe', open_circuit_potential=1)
        self.cv_rhe_agagcl = deepcopy(self.cv_rhe).set_potential_reference('agagcl', open_circuit_potential=1)
        self.cv_rhe_she = deepcopy(self.cv_rhe).set_potential_reference('she', open_circuit_potential=1)


    def test_potential_reference_conversion(self):
        """
        Test potential reference conversions.
        """

        # Ag/AgCl → SHE
        original_potential = self.cv.data['potential'].copy()
        shifted_potential = self.cv_she.data['potential']
        expected_shift = original_potential + 0.197 
        pd.testing.assert_series_equal(shifted_potential.round(5), expected_shift.round(5), check_names=False)

        # SHE → Ag/AgCl
        original_potential = self.cv_she.data['potential'].copy()
        shifted_potential = self.cv_she_agagcl.data['potential']
        expected_shift = original_potential - 0.197
        pd.testing.assert_series_equal(shifted_potential.round(5), expected_shift.round(5), check_names=False)

        # Ag/AgCl → RHE
        original_potential = self.cv.data['potential'].copy()
        shifted_potential = self.cv_agagcl_rhe.data['potential']
        expected_shift = original_potential + 1
        pd.testing.assert_series_equal(shifted_potential.round(5), expected_shift.round(5), check_names=False)

        # RHE → Ag/AgCl
        original_potential = self.cv_rhe.data['potential'].copy()
        shifted_potential = self.cv_rhe_agagcl.data['potential']
        expected_shift = original_potential - 1 
        pd.testing.assert_series_equal(shifted_potential.round(5), expected_shift.round(5), check_names=False)

        # RHE → SHE
        original_potential = self.cv_rhe.data['potential'].copy()
        shifted_potential = self.cv_rhe_she.data['potential']
        expected_shift = original_potential + 1 + 0.197
        pd.testing.assert_series_equal(shifted_potential.round(5), expected_shift.round(5), check_names=False)

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))
        
    def test_attributes(self):
        """ Test the from_biologic method """
        self.assertTrue(type(self.cv) == CyclicVoltammogram)
        self.assertTrue(type(self.cv.data) == pd.DataFrame)
        self.assertTrue(self.cv.pH == 7)
        self.assertTrue(self.cv.temperature == 298)
        self.assertTrue(self.cv.cation == self.na)
        self.assertTrue(self.cv.anion == self.cl)
        self.assertTrue(self.cv.electrolyte == self.electrolyte)
        self.assertTrue('potential' in self.cv.data.columns) 
        self.assertTrue('current' in self.cv.data.columns)
        self.assertTrue('cycle' in self.cv.data.columns)
        self.assertTrue('time' in self.cv.data.columns)
        self.assertTrue('segment' in self.cv.data.columns)
        self.assertTrue('scan_rate' in self.cv.data.columns)
        self.assertTrue('instrument' in self.cv.data.columns)
        self.assertTrue(type(self.cv._object_creation_time) == dt)
        self.assertTrue(type(self.cv.object_creation_time) == str)

        self.assertEqual(self.cv._potential_reference, 'agagcl')
        self.assertEqual(self.cv.potential_reference, 'Ag/AgCl')

        
    def test_drop_cycles(self):
        """ Test the drop_cycles method """
        data = copy(self.cv).drop_cycles(drop=[1]).data
        self.assertTrue(1 not in data['cycle'].values)

    def test_get_charges(self):
        """ Test the get_charge_passed method """
        integrals = self.cv.get_charge_passed()
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [2.4817, 0.0125, 2.4909])
        self.assertTrue('total_charge' in integrals.columns)
        self.assertTrue('anodic_charge' in integrals.columns)
        self.assertTrue('cathodic_charge' in integrals.columns)
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(all(integrals['anodic_charge'] >= 0))
        self.assertTrue(all(integrals['cathodic_charge'] >= 0))

        integrals = self.cv.get_charge_passed(average_segments=True)
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [2.4863, 0.0125])
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue('anodic_charge' in integrals.columns)
        self.assertTrue('cathodic_charge' in integrals.columns)
        self.assertTrue('anodic_charge_err' in integrals.columns)
        self.assertTrue('cathodic_charge_err' in integrals.columns)
        self.assertTrue(all(integrals['anodic_charge'] >= 0))
        self.assertTrue(all(integrals['cathodic_charge'] >= 0))

        figure = self.cv.get_charge_integration_plot(cycle=1, direction='reduction')
        # figure.show()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

        max_charges = self.cv.get_maximum_charges_passed()
        self.assertTrue(type(max_charges) == pd.DataFrame)
        self.assertTrue('total_charge' in max_charges.columns)
        self.assertTrue('section' in max_charges.columns)
        self.assertTrue('t_min' in max_charges.columns)
        self.assertTrue('t_max' in max_charges.columns)
        self.assertTrue('type' in max_charges.columns)
        self.assertTrue(all(max_charges['total_charge'] >= 0))
        self.assertTrue(set(max_charges['type']).issubset({'anodic_charge', 'cathodic_charge'}))
        self.assertTrue(max_charges.round(4).total_charge.to_list() == [0.0025, 0.0033])


class TestBiologic1FromData(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV data frame
        """
        self.na = Cation('Na+')
        self.cl = Anion('Cl-')
        self.water = Solvent('H2O')

        self.electrolyte = Electrolyte(cation=self.na, 
                                       anion=self.cl, 
                                       solvent=self.water, 
                                       pH=7, 
                                       temperature=298, 
                                       concentrations={self.na: 1, self.cl: 1})
        
        data = pd.read_table('test_trajectories/cyclic_voltammetry/biologic1.txt', sep="\t")

        self.cv = CyclicVoltammogram.from_biologic(data=data, 
                                                   electrolyte = self.electrolyte, 
                                                   metadata = {'scan_rate': 5, 'instrument': 'Biologic'}
                                                   )
        
    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))
        
    def test_attributes(self):
        """ Test the from_biologic method """
        self.assertTrue(type(self.cv) == CyclicVoltammogram)
        self.assertTrue(type(self.cv.data) == pd.DataFrame)
        self.assertTrue(self.cv.pH == 7)
        self.assertTrue(self.cv.temperature == 298)
        self.assertTrue(self.cv.cation == self.na)
        self.assertTrue(self.cv.anion == self.cl)
        self.assertTrue(self.cv.electrolyte == self.electrolyte)
        self.assertTrue('potential' in self.cv.data.columns) 
        self.assertTrue('current' in self.cv.data.columns)
        self.assertTrue('cycle' in self.cv.data.columns)
        self.assertTrue('time' in self.cv.data.columns)
        self.assertTrue('segment' in self.cv.data.columns)
        self.assertTrue('scan_rate' in self.cv.data.columns)
        self.assertTrue('instrument' in self.cv.data.columns)

    def test_drop_cycles(self):
        """ Test the drop_cycles method """
        data = copy(self.cv).drop_cycles(drop=[1]).data
        self.assertTrue(1 not in data['cycle'].values)

    def test_charges(self):
        """ Test the get_charge_passed method """
        integrals = self.cv.get_charge_passed()
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [2.4817, 0.0125, 2.4909])
        self.assertTrue('total_charge' in integrals.columns)
        self.assertTrue('anodic_charge' in integrals.columns)
        self.assertTrue('cathodic_charge' in integrals.columns)
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(all(integrals['anodic_charge'] >= 0))
        self.assertTrue(all(integrals['cathodic_charge'] >= 0))

        integrals = self.cv.get_charge_passed(average_segments=True)
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [2.4863, 0.0125])
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue('anodic_charge' in integrals.columns)
        self.assertTrue('cathodic_charge' in integrals.columns)
        self.assertTrue('anodic_charge_err' in integrals.columns)
        self.assertTrue('cathodic_charge_err' in integrals.columns)
        self.assertTrue(all(integrals['anodic_charge'] >= 0))
        self.assertTrue(all(integrals['cathodic_charge'] >= 0))

        figure = self.cv.get_charge_integration_plot(cycle=1, direction='reduction')
        # figure.show()

        max_charges = self.cv.get_maximum_charges_passed()
        self.assertTrue(type(max_charges) == pd.DataFrame)
        self.assertTrue('total_charge' in max_charges.columns)
        self.assertTrue('section' in max_charges.columns)
        self.assertTrue('t_min' in max_charges.columns)
        self.assertTrue('t_max' in max_charges.columns)
        self.assertTrue('type' in max_charges.columns)
        self.assertTrue(all(max_charges['total_charge'] >= 0))
        self.assertTrue(set(max_charges['type']).issubset({'anodic_charge', 'cathodic_charge'}))
        self.assertTrue(max_charges.round(4).total_charge.to_list() == [0.0025, 0.0033])


class TestBiologic2(unittest.TestCase):

    def setUp(self):
            """
            Reading in a PEDOT:PSS CV
            """
            self.cv = CyclicVoltammogram.from_biologic(path = 'test_trajectories/cyclic_voltammetry/biologic2.txt')
            
    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

    def test_get_charge_passed(self):
        """ Test the get_charge_passed method """
        integrals = self.cv.get_charge_passed()
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [0.0105])


class TestBiologic3(unittest.TestCase):

    def setUp(self):
        """
        Reading in a PEDOT:PSS CV
        """
        self.cv = CyclicVoltammogram.from_biologic(path = 'test_trajectories/cyclic_voltammetry/biologic3.txt')
        
    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

    def test_get_charge_passed_biologic3(self):
        """ Test the get_charge_passed method for biologic3 """
        integrals = self.cv.get_charge_passed()
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [3.3214, 61.8374, 3.2826, 0.0318])


class TestBiologic4(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV
        """
        self.cv = CyclicVoltammogram.from_biologic(path = 'test_trajectories/cyclic_voltammetry/biologic4.txt')

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

    def test_get_charge_passed_biologic4(self):
        """ Test the get_charge_passed method for biologic4 """
        integrals = self.cv.get_charge_passed()
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [5229.4038, 113.192])

    
class TestBiologic5(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV
        """
        self.cv = CyclicVoltammogram.from_biologic(path = 'test_trajectories/cyclic_voltammetry/biologic5.txt')

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))
            
    def test_get_integration_plots(self):
        """ Test the get_maximum_charge_integration_plot method for anodic """
        figure = self.cv.get_maximum_charge_integration_plot(section=3)
        # figure.show()
        self.assertTrue(type(figure) == go.Figure)

    def test_downsample(self):
        """ Test the downsample method with biologic6 """
        cv = copy(self.cv).downsample(25)
        self.assertTrue(len(cv.data.query('segment == 3')) == 27)
        # cv.get_current_time_plot().show()
        # cv.get_potential_time_plot().show()

        cv = copy(self.cv).drop_cycles(drop=[0, 1, 2]).downsample(25)
        self.assertTrue(len(cv.data.query('segment == 5')) == 26)
        # cv.get_current_time_plot().show()
        # cv.get_potential_time_plot().show()

    def test_get_peaks_biologic5(self):
        """ Test the get_peaks method for biologic5 """
        peaks = self.cv.get_peaks(window=0.1)
        self.assertTrue(type(peaks) == pd.DataFrame)
        self.assertTrue('current_peak' in peaks.columns)
        self.assertTrue('fit_current' in peaks.columns)
        self.assertTrue(all(peaks['current_peak'].notnull()))
        self.assertTrue(all(peaks['fit_current'].notnull()))

        figure = self.cv.get_peak_plot(direction='oxidation', window = 0.1, width=700, height=500)
        self.assertTrue(type(figure) == go.Figure)
        self.assertTrue(len(figure.data) > 0)
        self.assertTrue(any(trace.name.startswith("Fitted") for trace in figure.data))
        self.assertTrue(any(trace.name.startswith("Peak") for trace in figure.data))
        # figure.show()

        figure = self.cv.get_peak_plot(direction='reduction', window = 0.1, width=700, height=500)
        self.assertTrue(type(figure) == go.Figure)
        self.assertTrue(len(figure.data) > 0)
        self.assertTrue(any(trace.name.startswith("Fitted") for trace in figure.data))
        self.assertTrue(any(trace.name.startswith("Peak") for trace in figure.data))
        # figure.show()

        figure1 = self.cv.get_peak_plot(direction='oxidation', window = 0.1, polynomial_order=2, width=700, height=500)
        figure2 = self.cv.get_peak_plot(direction='oxidation', window = 0.1, polynomial_order=6, width=700, height=500)
        self.assertTrue(type(figure1) == go.Figure)
        self.assertTrue(len(figure1.data) > 0)
        self.assertTrue(len(figure2.data) > 0)
        self.assertTrue(any(trace.name.startswith("Fitted") for trace in figure1.data))
        self.assertTrue(any(trace.name.startswith("Peak") for trace in figure1.data))
        # figure1.show()
        # figure2.show()

        current_figure, potential_figure = self.cv.get_plots_peaks_with_cycle(polynomial_order=4, window=0.1, width=700, height=500)
        self.assertTrue(type(current_figure) == go.Figure)
        self.assertTrue(type(potential_figure) == go.Figure)
        self.assertTrue(len(current_figure.data) > 0)
        self.assertTrue(len(potential_figure.data) > 0)
        self.assertTrue(any(trace.name.startswith("anodic peak") or trace.name.startswith("cathodic peak") for trace in current_figure.data))
        self.assertTrue(any(trace.name.startswith("anodic peak") or trace.name.startswith("cathodic peak") for trace in potential_figure.data))
        # current_figure.show()
        # potential_figure.show()


class TestBiologic6(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV
        """
        self.cv = CyclicVoltammogram.from_biologic(path = 'test_trajectories/cyclic_voltammetry/biologic6.txt')

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

    def test_get_charge_passed_biologic7(self):
        """ Test the get_charge_passed method for biologic7 """
        integrals = self.cv.get_charge_passed()
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [5.9663, 0.0, 5.5174])


class TestAftermath1(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV
        """
        self.cv = CyclicVoltammogram.from_aftermath(path = 'test_trajectories/cyclic_voltammetry/aftermath1.csv', scan_rate=5)

        self.cv_she = copy(self.cv).set_potential_reference('she')

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))


class TestAftermath2(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV
        """
        self.cv = CyclicVoltammogram.from_aftermath(path = 'test_trajectories/cyclic_voltammetry/aftermath2.csv', scan_rate=5)

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

    
class TestHtmlBase64(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV
        """
        self.na = Cation('Na+')
        self.cl = Anion('Cl-')
        self.water = Solvent('H2O')

        self.electrolyte = Electrolyte(cation=self.na, 
                                    anion=self.cl, 
                                    solvent=self.water, 
                                    pH=7, 
                                    temperature=298, 
                                    concentrations={self.na: 1, self.cl: 1})
        
        mime_type = mimetypes.guess_type('test_trajectories/cyclic_voltammetry/biologic1.txt')[0]
        if mime_type is None:
            mime_type = 'text/plain'

        with open('test_trajectories/cyclic_voltammetry/biologic1.txt', 'rb') as file:
            file_content = file.read()
            base64_data = base64.b64encode(file_content).decode('utf-8')
            base64_data = f'data:{mime_type};base64,{base64_data}'
            self.cv = CyclicVoltammogram.from_html_base64(file_contents = base64_data, electrolyte = self.electrolyte, source='biologic')

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

    def test_attributes(self):
        """ Test the from_biologic method """
        self.assertTrue(type(self.cv) == CyclicVoltammogram)
        self.assertTrue(type(self.cv.data) == pd.DataFrame)
        self.assertTrue(self.cv.pH == 7)
        self.assertTrue(self.cv.temperature == 298)
        self.assertTrue(self.cv.cation == self.na)
        self.assertTrue(self.cv.anion == self.cl)
        self.assertTrue(self.cv.electrolyte == self.electrolyte)
        self.assertTrue('potential' in self.cv.data.columns) 
        self.assertTrue('current' in self.cv.data.columns)
        self.assertTrue('cycle' in self.cv.data.columns)
        self.assertTrue('time' in self.cv.data.columns)
        self.assertTrue('segment' in self.cv.data.columns)

    def test_drop_cycles(self):
        """ Test the drop_cycles method """
        data = copy(self.cv).drop_cycles(drop=[1]).data
        self.assertTrue(1 not in data['cycle'].values)

    def test_get_charge_passed(self):
        """ Test the get_charge_passed method """
        integrals = self.cv.get_charge_passed()
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [2.4817, 0.0125, 2.4909])
        self.assertTrue('total_charge' in integrals.columns)
        self.assertTrue('anodic_charge' in integrals.columns)
        self.assertTrue('cathodic_charge' in integrals.columns)
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(all(integrals['anodic_charge'] >= 0))
        self.assertTrue(all(integrals['cathodic_charge'] >= 0))

        integrals = self.cv.get_charge_passed(average_segments=True)
        charges = integrals.assign(anodic_charge = lambda x: x['anodic_charge']*1000).round(4)['anodic_charge'].to_list()
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue(charges == [2.4863, 0.0125])
        self.assertTrue(type(integrals) == pd.DataFrame)
        self.assertTrue('anodic_charge' in integrals.columns)
        self.assertTrue('cathodic_charge' in integrals.columns)
        self.assertTrue('anodic_charge_err' in integrals.columns)
        self.assertTrue('cathodic_charge_err' in integrals.columns)
        self.assertTrue(all(integrals['anodic_charge'] >= 0))
        self.assertTrue(all(integrals['cathodic_charge'] >= 0))

        figure = self.cv.get_charge_integration_plot(cycle=1, direction='reduction')
        # figure.show()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

        max_charges = self.cv.get_maximum_charges_passed()
        self.assertTrue(type(max_charges) == pd.DataFrame)
        self.assertTrue('total_charge' in max_charges.columns)
        self.assertTrue('section' in max_charges.columns)
        self.assertTrue('t_min' in max_charges.columns)
        self.assertTrue('t_max' in max_charges.columns)
        self.assertTrue('type' in max_charges.columns)
        self.assertTrue(all(max_charges['total_charge'] >= 0))
        self.assertTrue(set(max_charges['type']).issubset({'anodic_charge', 'cathodic_charge'}))
        self.assertTrue(max_charges.round(4).total_charge.to_list() == [0.0025, 0.0033])


class TestBiologic7(unittest.TestCase):

    def setUp(self):
        """
        Reading in a BBL CV
        """
        self.cv = CyclicVoltammogram.from_biologic(path = 'test_trajectories/cyclic_voltammetry/biologic7.txt')

    def test_show_plots(self):
        """ Test the show_current_potential, show_current_time and show_potential_time methods """
        # self.cv.show_current_potential()
        # self.cv.show_current_time()
        # self.cv.show_potential_time()
        self.assertTrue(type(self.cv.data == pd.DataFrame))

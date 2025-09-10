from Materials_Data_Analytics.experiment_modelling.chrono_amperometry import ChronoAmperometry
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
import numpy as np

class TestChronoAmperometrySynthetic(unittest.TestCase):

    def setUp(self):
        """
       Reading in a CA.
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
        
        # Generate fake chronoamperometry data for testing
        time_values = np.linspace(0, 100, num=101)  
        current_values = np.linspace(1.0, 0.1, num=101)
        fake_ca_data = pd.DataFrame({'time': time_values, 'current': current_values})

        self.ca = ChronoAmperometry(
            potential_reference='Ag/AgCl',
            time=fake_ca_data['time'],
            current=fake_ca_data['current'],
            electrolyte=self.electrolyte
        )

    def test_plot(self):
        """ Test the get_current_time_plot"""
        fig = self.ca.get_current_time_plot()
        # fig.show()
        self.assertTrue(type(self.ca.data == pd.DataFrame))
        

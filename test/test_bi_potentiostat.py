from Materials_Data_Analytics.experiment_modelling.chrono_amperometry import ChronoAmperometry
from Materials_Data_Analytics.experiment_modelling.cyclic_voltammetry import CyclicVoltammogram
from Materials_Data_Analytics.experiment_modelling.bi_potentiostat import RingDiskMeasurement

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

class TestBiPotentiostat(unittest.TestCase):

    def setUp(self):
        """
       Reading in bipotentiostat (CA-CV) data
        """
        self.bipot = RingDiskMeasurement.from_biologic(
            potential_reference='Ag/AgCl',
            path='test_trajectories/ring_disk/test_1.txt'
        )

    def test_attributes(self):

        self.assertTrue(self.bipot._chronoamperometry, ChronoAmperometry)
        self.assertTrue(self.bipot._cyclicvoltammogram, CyclicVoltammogram)

        self.assertTrue('potential' in self.bipot._cyclicvoltammogram._data.columns)
        self.assertTrue('current' in self.bipot._cyclicvoltammogram._data.columns)
        self.assertTrue('time' in self.bipot._cyclicvoltammogram._data.columns)


        self.assertTrue('current' in self.bipot._chronoamperometry._data.columns)
        self.assertTrue('time' in self.bipot._chronoamperometry._data.columns)


    def test_plots(self):

        fig1 = self.bipot._cyclicvoltammogram.get_current_potential_plot()
        fig2 = self.bipot._chronoamperometry.get_current_time_plot()
        fig1.show()
        fig2.show()

        fig3 = self.bipot.get_disk_current_potential_plot()
        fig4 = self.bipot.get_ring_current_potential_plot()
        fig3.show()
        fig4.show()

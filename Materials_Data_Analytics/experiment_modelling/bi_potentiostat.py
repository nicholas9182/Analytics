from Materials_Data_Analytics.experiment_modelling.core import ElectrochemicalMeasurement
from Materials_Data_Analytics.experiment_modelling.cyclic_voltammetry import CyclicVoltammogram
from Materials_Data_Analytics.experiment_modelling.chrono_amperometry import ChronoAmperometry
from Materials_Data_Analytics.materials.electrolytes import Electrolyte
from Materials_Data_Analytics.materials.ions import Cation, Anion

import pandas as pd

import plotly.express as px
from copy import deepcopy


class BiPotentiostat(ElectrochemicalMeasurement):

    def __init__(self,
                 potential_reference: str,
                 measurement1: ElectrochemicalMeasurement,
                 measurement2: ElectrochemicalMeasurement,
                 electrolyte: Electrolyte = None,
                 metadata: dict = None
                 ):
        
        super().__init__(potential_reference=potential_reference, 
                         electrolyte=electrolyte, 
                         metadata=metadata)
                
        pass


class RingDiskMeasurement(BiPotentiostat):

    def __init__(self,
                 potential_reference: str,
                 cyclicvoltammogram: CyclicVoltammogram,
                 chronoamperometry: ChronoAmperometry,
                 electrolyte: Electrolyte = None,
                 metadata: dict = None
                 ):
        
        super().__init__(potential_reference=potential_reference,
                         measurement1=cyclicvoltammogram, 
                         measurement2=chronoamperometry,
                         electrolyte=electrolyte,
                         metadata=metadata)
        
        self._cyclicvoltammogram = deepcopy(cyclicvoltammogram)
        self._chronoamperometry = deepcopy(chronoamperometry)

    @classmethod
    def from_biologic(cls, 
                      potential_reference: str,  
                      path: str = None, 
                      data: pd.DataFrame = None, 
                      **kwargs):
        """
        Function to read in file and from contents construct a cv object and a ca object from a biologic file.
        """

        if path is None and data is not None:
            data = data
        elif path is not None and data is None:
            data = pd.read_table(path, sep='\t')

        data = (data
                .rename({'Edisk/V': 'potential', 'Idisk/mA': 'idisk', 'Iring/mA': 'iring', 'time/s': 'time'}, axis=1)
                .filter(['potential', 'idisk', 'iring', 'time'])
                )

        cv = CyclicVoltammogram(potential_reference=potential_reference,
                                potential=data['potential'],
                                current=data['idisk'], 
                                time=data['time'], 
                                **kwargs)
        
        ca = ChronoAmperometry(potential_reference=potential_reference,
                                current=data['iring'], 
                                time=data['time'], 
                                **kwargs)

        return cls(potential_reference = potential_reference, cyclicvoltammogram = cv, chronoamperometry = ca)
    
   
    def get_disk_current_potential_plot(self, **kwargs):
        """
        Function to plot the disk data (cv).
        """
        data = self._cyclicvoltammogram._data.copy()  

        figure = px.line(
            data, 
            x='potential', 
            y='current', 
            markers=True,
            labels={'potential': 'Potential [V]', 'current': 'Disk Current [mA]'},
            title="Disk Current vs. Potential",
            **kwargs
        )

        return figure

    def get_ring_current_potential_plot(self, **kwargs):
        """
        Function to plot ring current vs. potential, using CA and CV data.
        """
        data_cv = self._cyclicvoltammogram._data.copy()
        data_ca = self._chronoamperometry._data.copy()  
        data_cv_sorted = data_cv[['time', 'potential']].sort_values('time')
        data_ca_sorted = data_ca[['time', 'current']].sort_values('time')
        data_merged = pd.merge_asof(data_ca_sorted, data_cv_sorted, on='time')

        figure = px.line(
            data_merged,
            x='potential',  
            y='current',  
            markers=True,
            labels={'potential': 'Potential [V]', 'current': 'Ring Current [mA]'},
            title="Ring Current vs. Potential",
            **kwargs
        )

        return figure

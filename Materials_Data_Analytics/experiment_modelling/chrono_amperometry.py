from Materials_Data_Analytics.experiment_modelling.core import ElectrochemicalMeasurement
from Materials_Data_Analytics.experiment_modelling.cyclic_voltammetry import CyclicVoltammogram
from Materials_Data_Analytics.materials.electrolytes import Electrolyte
from Materials_Data_Analytics.materials.ions import Cation, Anion
import pandas as pd
import numpy as np
from typing import Union
import plotly.express as px
import plotly.graph_objects as go
import scipy.integrate as integrate
import base64
import io


class ChronoAmperometry(ElectrochemicalMeasurement):

    def __init__(self,
                 potential_reference: str,
                 potential: float = None,  
                 current: Union[list, pd.Series, np.array] = None,   
                 time: Union[list, pd.Series, np.array] = None,
                 electrolyte: Electrolyte = None,
                 metadata: dict = None):
        
        super().__init__(potential_reference=potential_reference, electrolyte=electrolyte, metadata=metadata)

        self._data = pd.DataFrame()

        if current is not None:
            self._data['current'] = current
        if time is not None:
            self._data['time'] = time
        if potential is not None:
            self._data['potential'] = potential

        self._data = self._wrangle_data(self._data)

    def _wrangle_data(self, data) -> pd.DataFrame:
        """
        Wrangle data for CA measurements 
        """
        data = (data
                .dropna()
                .reset_index(drop=True)
                .sort_values(by=['time'])
                .assign(time=lambda x: x['time'] - x['time'].min())
                .groupby(['time'], as_index=False)
                .mean()
                .sort_values('time')
                .reset_index(drop=True)
                )

        return data

    def get_current_time_plot(self, **kwargs):
        """
        Function to plot the CA data
        """
        data = self._data  

        figure = px.line(
            data, 
            x='time', 
            y='current', 
            markers=True,
            labels={'time': 'Time [s]', 'current': 'Current [mA]'},
            title="Current vs. Time",
            **kwargs
        )

        return figure


    
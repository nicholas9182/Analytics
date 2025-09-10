from Materials_Data_Analytics.experiment_modelling.core import ElectrochemicalMeasurement
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
    """
    A general class for the analysis of chronoamperometric data.
    """
    def __init__(self,  
                 potential: Union[list, pd.Series, np.array] = None,
                 current: Union[list, pd.Series, np.array] = None,
                 time: Union[list, pd.Series, np.array] = None,
                 electrolyte: Electrolyte = None,
                 metadata: dict = None
                 ) -> None:
        
        super().__init__(electrolyte, metadata=metadata)

        self._data = pd.DataFrame()

        if len(potential) and len(current) and len(time) != 0:
            self._data = (pd
                          .DataFrame({'potential': potential, 'current': current, 'time': time})
                          .pipe(self._wrangle_data)
                          )
        
        self._max_cycle = self._data['cycle'].max()
        self._max_segment = self._data['segment'].max()

    def _wrangle_data(self, data, first_index = 50, remove_last_n = 50) -> pd.DataFrame:
        """
        Function to wrangle the data
        :param data: pd.DataFrame with columns potential, current, cycle, time
        """
        last_index = data['current'].last_valid_index() - remove_last_n
        data = (data
                .query('index > @first_index')
                .query('index < @last_index')
                .dropna()
                .reset_index(drop=True)
                .sort_values(by=['time'])
                .assign(time = lambda x: x['time'] - x['time'].min())
                .groupby(['potential','time'], as_index=False)
                .mean()
                .sort_values('time')
                .reset_index(drop=True)
                )
        
        data = (data
                .pipe(self._find_current_roots)
                .pipe(self._find_voltage_peaks)
                .pipe(self._add_endpoints)
                .pipe(self._check_types)
                .sort_values(by=['time', 'segment'])
                .reset_index(drop=True)
                )
        
        return data
    
    def set_ref(self, ref: str, ocp_data: pd.DataFrame = None) -> None:
        """
        Set the reference potential for the data.
        Adjust the potential column based on the reference electrode.
        
        :param ref: The target reference electrode ('RHE', 'SHE', 'Ag/AgCl').
        :param ocp_data: DataFrame containing OCV measurements, used to calculate RHE_shift for RHE.
        """
        # References
        ref_offset = {'RHE': None,        # To be calculated w/ocp_data (varies on pH)
                      'SHE': 0.0,         # No offset for SHE
                      'Ag/AgCl': 0.197}  # Typical Ag/AgCl offset 

        # Calculate offset for RHE
        if ref == 'RHE':
            if ocp_data is None:
                raise ValueError("OCV data is required when setting ref to RHE")
            
            RHE_shift = (ocp_data
                        .iloc[-10:] 
                        .assign(RHEshift=lambda x: x['potential'].mean())  
                        .iloc[0]  
                        ['RHEshift'])  

            ref_offset['RHE'] = RHE_shift

        # Adjust potential based on chosen reference
        self._data['potential'] = self._data['potential'] + ref_offset[ref]

    def _find_current_roots(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function to find the time and voltage points where the current passes through 0
        """
        raw_roots_indexes = (data
                             .copy()
                             .assign(sign_diff = lambda x: np.sign(x['current']).diff())
                             .query('sign_diff != 0 and sign_diff.notna()')
                             .index
                             )
        
        for i in raw_roots_indexes:
            data_interpolate = data.iloc[i-1:i+1]
            time_root = self.get_root_linear_interpolation(data_interpolate['time'], data_interpolate['current'])
            potential_root = self.get_root_linear_interpolation(data_interpolate['potential'], data_interpolate['current'])

            new_row = (data_interpolate
                       .query('index == index.min()')
                       .assign(potential = potential_root)
                       .assign(time = time_root)
                       .assign(current = 0)
                       )

            data = pd.concat([data, new_row], ignore_index=True) 

        return data.sort_values(by=['time']).reset_index(drop=True)
    
    def _add_endpoints(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function to double up data points at the end of the cycles so that each cycle is complete when filtered by direction
        """
        for s in range(data['segment'].min()+1, data['segment'].max()+1):
            prev_seg_num = s - 1
            prev_time = data.query('segment == @prev_seg_num')['time'].iloc[-1]
            prev_current = data.query('segment == @prev_seg_num')['current'].iloc[-1]
            prev_potential = data.query('segment == @prev_seg_num')['potential'].iloc[-1]

            new_row = (data
                    .query('segment == @s')
                    .query('time == time.min()')
                    .assign(time = prev_time, current = prev_current, potential = prev_potential)
                    )
            
            data = pd.concat([data, new_row], ignore_index=True).sort_values(by=['time'])

        return data
    
    def _check_types(self, data) -> pd.DataFrame:
        """
        Function to check the data types in the data columns
        :param data: pd.DataFrame with columns potential, current, cycle, time
        """
        return (data
                .assign(cycle = lambda x: x['cycle'].astype(int))
                .assign(time = lambda x: x['time'].astype(float))
                .assign(potential = lambda x: x['potential'].astype(float))
                .assign(current = lambda x: x['current'].astype(float))
                .assign(segment = lambda x: x['segment'].astype(int))
                .assign(direction = lambda x: x['direction'].astype(str))
                )

    @property
    def data(self) -> pd.DataFrame:
        
        data = self._data.copy()
        metadata = self.metadata

        for k in metadata.keys():
            data[k] = self.metadata[k]

        return data
    
    @property
    def steps_per_cycle(self) -> int:
        return self._data.query('segment == 0')['time'].count()


    @classmethod
    def from_biologic(cls, path: str = None, data: pd.DataFrame = None, **kwargs):
        """
        Function to make a ChronoAmperometry object from a Biologic file
        """

        if path is None and data is not None:
            data = data
        elif path is not None and data is None:
            data = pd.read_table(path, sep='\t')

        data = (data
                .rename({'Edisk/V': 'potential', 'Iring/mA': 'current', 'time/s': 'time'}, axis=1)
                .filter(['potential', 'current', 'time'])
                )
        
        ca = cls(potential=data['potential'], current=data['current'], time=data['time'], **kwargs)
        
        return ca
    
    @classmethod
    def from_aftermath(cls, path: str = None, scan_rate: float = None, data: pd.DataFrame = None, **kwargs):
        """
        Function to make a ChronoAmperometry object from an AfterMath file
        """

        if path is None and data is not None:
            data = data
        elif path is not None and data is None:
            data = pd.read_table(path, sep=",")

        if type(scan_rate) != float:
            scan_rate = float(scan_rate)

        scan_rate = scan_rate/1000

        data = (data
                .rename({'Potential (V)': 'potential', 'Current (A)': 'current'}, axis=1)
                .filter(['potential', 'current'])
                .assign(current = lambda x: x['current']*1000)
                .assign(dv = lambda x: x['potential'] - x['potential'].shift(1))
                .assign(time = lambda x: x['dv'].abs().cumsum()/scan_rate)
                )
        
        data.loc[0, 'time'] = 0

        ca = cls(potential=data['potential'], current=data['current'], time=data['time'], **kwargs)

        return ca
    
    @classmethod
    def nyquist(cls, path: str = None, format: str = None, data: pd.DataFrame = None, **kwargs):
        """
        Function to process Nyquist data and calculate electrolyte resistance.
        """

        valid_formats = ['aftermath', 'biologic']
        if format not in valid_formats:
            raise ValueError(f"Invalid format specified. Choose from {valid_formats}.")
        
        if path is None and data is not None:
            data = data
        elif path is not None and data is None:
            if format == 'aftermath':
                data = pd.read_table(
                    path, 
                    delimiter=',', 
                    names=['segment', 'real_impedence', 'imaginary_impedence'], 
                    header=0, 
                    dtype=float
                )
            elif format == 'biologic':
                data = pd.read_table(
                    path, 
                    delim_whitespace=True, 
                    names=['frequency', 'real_impedence', 'imaginary_impedence', 'magnitude_impedence', 'phase'], 
                    header=0, 
                    dtype=float
                )

        electrolyte_resistance = (data
                                  .apply(lambda df: df[df['imaginary_impedence'] == 
                                                       df.query('imaginary_impedence > 0')['imaginary_impedence'].min()])
                                  .rename(columns={'real_impedence': 'electrolyte_resistance'})
                                  )

        return electrolyte_resistance


    def drop_cycles(self, drop: list[int] | int = None, keep: list[int] | int = None) -> pd.DataFrame:
        """
        Function to edit which cycles are being considered
        :param drop: list of cycles to drop
        :param keep: list of cycles to keep
        """
        if type(drop) == int:
            drop = [int]

        if type(keep) == int:
            keep = [keep]

        if drop is not None:
            self._data = self._data.query('cycle not in @drop')

        if keep is not None:
            self._data = self._data.query('cycle in @keep')

        return self
    
    
    def get_ring_plot(self, **kwargs):
        """
        Function to plot the ring data from an RDE measurement
        """
        data = self.data.assign(cycle_direction = lambda x: x['cycle'].astype('str') + ', ' + x['direction'])

        figure = px.line(data, x='potential', y='current', color='cycle_direction', markers=True, 
                         labels={'potential': 'Potential [V]', 'current': 'Current [mA]'}, **kwargs)
        
        return figure
    
    def show_ring(self, **kwargs):
        """
        Function to show the ring data
        """
        figure = self.get_ring_plot(**kwargs)
        figure.show()
        return self
    
    @property
    def pH(self) -> float:
        return self.electrolyte.pH
    
    @property
    def temperature(self) -> Electrolyte:
        return self.electrolyte.temperature
    
    @property
    def cation(self) -> Cation:
        return self.electrolyte.cation
    
    @property
    def anion(self) -> Anion:
        return self.electrolyte.anion
    
    @property
    def max_cycle(self) -> int:
        return self._max_cycle
    
    def downsample(self, n: int | list[float] = 400) -> pd.DataFrame:
        """
        Function to downsample the data
        """
        # from a range and number of intervals, get the bin edges
        def get_bins(df, n):
            t_min = df['time'].min()
            t_max = df['time'].max()
            t_mids = np.linspace(t_min, t_max, n)
            dt = (t_max - t_min) / n
            bins = np.concatenate(([t_min - dt / 2], t_mids + dt / 2))
            return bins

        # get the original data and remove the additional segment points and current roots
        data = (self
                ._data
                .query('current != 0')
                .groupby(['segment'], group_keys=False)
                .apply(lambda df: df.query('index != index.max()'))
                )

        # downsample the data. Attention to use the .agg as it is much faster than .apply
        down_sampled_data = (data
                             .groupby(['cycle', 'direction', 'segment'], group_keys=False)
                             .apply(lambda df: df.assign(time_bin = pd.cut(df['time'], bins=get_bins(df, n), labels=False)))
                             .groupby(['cycle', 'direction', 'segment', 'time_bin'], group_keys=False)
                             .agg(
                                 potential = ('potential', 'mean'),
                                 current = ('current', 'mean'),
                                 time = ('time', 'mean')
                                 )
                             .reset_index()
                             .drop(columns=['time_bin'])
                             .pipe(self._add_endpoints)
                             .pipe(self._find_current_roots)
                             )
        
        self._data = down_sampled_data
        return self
    
# add in a function to calculate amt of product (h2o, h2o2) produced in an RRDE measurement 

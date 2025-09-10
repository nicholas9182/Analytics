from Materials_Data_Analytics.materials.electrolytes import Electrolyte
import pandas as pd
import scipy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import plotly.express as px
from datetime import datetime as dt

class Measurement:
    """
    A general measurement class
    Main contributors:
    Nicholas Siemons
    Contributors:
    """
    def __init__(self, metadata: dict = None) -> None:
        self._data = pd.DataFrame()
        self._metadata = metadata if metadata is not None else {}
        self._object_creation_time = dt.now()

    @property
    def object_creation_time(self) -> str:
        return self._object_creation_time.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def metadata(self) -> dict:
        return self._metadata
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the processed CA data along with metadata.
        """
        data = self._data.copy()
        metadata = self.metadata

        for k in metadata.keys():
            data[k] = self.metadata[k]

        return data
    
    @metadata.setter
    def metadata(self, value: dict):
        """
        Add items to the metadata dictionary.  If the key is already in the metadata, then it will overwrite the
        existing value
        """
        if type(value) != dict:
            raise ValueError('metadata must be a dictionary')
        
        for k in value.keys():
            self._metadata[k] = value[k]

    @staticmethod
    def get_root_linear_interpolation(x: pd.Series, y: pd.Series):
        """
        Function to get the root of the line given by two points
        """

        if len(x) != 2 or len(y) != 2:
            raise ValueError('The x and y series must have exactly two points to find the root with linear interpolation')
        
        x1 = x.iloc[0]
        x2 = x.iloc[1]
        y1 = y.iloc[0]
        y2 = y.iloc[1]
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        root = -intercept / slope

        return root
    
    @staticmethod
    def get_root_cube_spline(x: pd.Series, y: pd.Series):
        """
        Function to get the root of a data using cubic spline interpolation
        """
        spline = sp.interpolate.CubicSpline(x, y)
        roots = spline.roots()

        if len(roots) == 0 or len(roots) > 1:
            raise ValueError('There are no or multiple roots at a crossing point! Check the noise around current crossing points')
        
        return roots[0]
    
    @staticmethod
    def find_local_peak_with_polynomial(data, 
                                        y_col: str, 
                                        x_col: str, 
                                        initial_guess: float, 
                                        window = 0.01, 
                                        polynomial_order = 4) -> pd.DataFrame:
        """
        Function to find a local peak in a data set. 
        1. Fit a polynomial to the data
        2. Find the derivative of the polynomial
        3. Find the roots of the derivative
        4. Find the y values of the roots
        5. Find the maximum y value
        6. Return the x and y values of the peak
        :param data: The data frame with the x and y values
        :param y_col: The column name of the y values
        :param x_col: The column name of the x values
        :param initial_guess: The initial guess of the x value of the peak
        :param window: The x window around the initial guess to search for the peak
        :param polynomial_order: The order of the polynomial to fit to the data
        :return: a data frame with the interpolated polynomial and the peak values
        """

        # check fitting with even polynomial order
        if polynomial_order % 2 != 0:
            raise ValueError('Polynomial order must be an even number')
        
        # check that the length of data is atleast twice the polynomial order
        if len(data) < polynomial_order * 2:
            raise ValueError('Dataframe must have at least twice the polynomial order number of points. Increase the window size or decrease polynomial order')

        # find the window around the initial guess
        data = data[(data[x_col] < initial_guess + window) & (data[x_col] > initial_guess - window)]
        x_min = data[x_col].min()
        x_max = data[x_col].max()
        x_range = (x_min, x_max)

        try:
            # get the coefficients and covariance matrix of the polynomial fit
            coefficients = np.polyfit(data[x_col], data[y_col], polynomial_order, cov=True)[0]
        except np.linalg.LinAlgError:
            raise ValueError('The polynomial fit failed. Increase the window size or decrease polynomial order')

        # create a polymonial object and its derivative from those coefficients
        polynomial = np.poly1d(coefficients)
        differential = np.polyder(polynomial)

        # find the roots of the derivative
        roots = np.roots(differential)

        # get the roots which are in the range of the x values
        critical_points_in_range = [np.real(x) for x in roots if x_range[0] <= x <= x_range[1] and np.isreal(x)]

        # get the y values of the critical points and the boundaries
        x_values_to_check = critical_points_in_range + [x_range[0], x_range[1]]
        y_values = [polynomial(x) for x in x_values_to_check]

        # get the maximum y value
        max_y = max(y_values)
        max_x = x_values_to_check[y_values.index(max_y)]

        # check that the max isnt an edge case
        if max_x == x_range[0] or max_x == x_range[1]:
            raise ValueError('The peak is at the edge of the window. Increase the window size or decrease polynomial order')

        data['fit_' + y_col] = polynomial(data[x_col])
        data[y_col + '_peak'] = data['fit_' + y_col].max()

        data[y_col + '_peak'] = max_y
        data[x_col + '_peak'] = max_x

        return data
    

    @staticmethod
    def binary_search(x0:float,
                   y0:float,
                   intensity_flatten_vector:np.array,
                   x_range: tuple,
                   y_range:tuple,
                   nodes_x:int,
                   nodes_y:int):
        """
        Function to find the intensity of a point in a 2D grid using bilinear interpolation
        
        :param x0: The x value of the point
        :param y0: The y value of the point
        :param intensity_flatten_vector: The intensity vector of the 2D grid
        :param x_range: The x range of the grid
        :param y_range: The y range of the grid
        :param nodes_x: The number of nodes in the x direction
        :param nodes_y: The number of nodes in the y direction
        :return: The intensity of the point
        """

            
        dq = (x_range[1] - x_range[0]) / (nodes_x-1)
        dchi = (y_range[1] - y_range[0]) / (nodes_y-1)

        q_index = int((x0 - x_range[0]) / dq)
        chi_index = int((y0 - y_range[0]) / dchi)

        corner = np.zeros(4).tolist()           
        corner[0] = chi_index * nodes_x + q_index
        corner[1] = chi_index * nodes_x + q_index + 1
        corner[2] = (chi_index + 1) * nodes_x + q_index
        corner[3] = (chi_index + 1) * nodes_x + q_index + 1

        weight_q = ((x0 - x_range[0])-dq*q_index)/dq
        weight_chi = ((y0 - y_range[0])-dchi*chi_index)/dchi

        weight = np.zeros(4).tolist()
        weight[0] = (1-weight_q) * (1-weight_chi)
        weight[1] = (weight_q) * (1-weight_chi)
        weight[2] = (1-weight_q) * (weight_chi)
        weight[3] = (weight_q) * (weight_chi)

        skipList = []

        if weight_q == 1:
            skipList.append(0)
            skipList.append(2)
        if weight_q == 0:
            skipList.append(1)
            skipList.append(3)
        if weight_chi == 1:
            skipList.append(0)
            skipList.append(1)
        if weight_chi == 0:
            skipList.append(2)
            skipList.append(3)

        if q_index > nodes_x-1 or q_index < 0:
            skipList.append(2)
            skipList.append(0)
            skipList.append(1)
            skipList.append(3)
        elif chi_index > nodes_y-1 or chi_index < 0:
            skipList.append(0)
            skipList.append(1)
            skipList.append(2)
            skipList.append(3)
        if q_index == nodes_x-1:
            skipList.append(1)
            skipList.append(3)
        if chi_index == nodes_y-1:
            skipList.append(2)
            skipList.append(3)
            
        intensity_List = np.zeros(4).tolist()
        for i in range (0,4):
            if i not in skipList:
                intensity_List[i] = intensity_flatten_vector[corner[i]]
                if bool(np.isnan(intensity_flatten_vector[corner[i]])):
                    skipList.append(i)
            else:
                intensity_List[i] = np.nan
                skipList.append(i)
        
        skipList = list(set(skipList))

        
        if len(skipList) == 4:
            return None
        
        elif len(skipList) == 0: 
            intensity_binsearch = (weight[0]*intensity_List[0]+weight[1]*intensity_List[1]+weight[2]*intensity_List[2]+weight[3]*intensity_List[3])
            # print(f"q = {q}, q_corner1 = {dq*q_index + q_range[0]}, q_corner2 = {dq*q_index + q_range[0] + dq}")
            # print(f"chi = {chi}, chi_corner1 = {dchi*chi_index + chi_range[0]}, chi_corner2 = {dchi*chi_index + chi_range[0] + dchi}")
        
        elif len(skipList) == 1:
            if 0 in skipList:
                intensity_binsearch = (weight[1]*intensity_List[1]+weight[2]*intensity_List[2]+weight[3]*intensity_List[3])/(weight[1]+weight[2]+weight[3])
            elif 1 in skipList:
                intensity_binsearch = (weight[0]*intensity_List[0]+weight[2]*intensity_List[2]+weight[3]*intensity_List[3])/(weight[0]+weight[2]+weight[3])
            elif 2 in skipList:
                intensity_binsearch = (weight[0]*intensity_List[0]+weight[1]*intensity_List[1]+weight[3]*intensity_List[3])/(weight[0]+weight[1]+weight[3])
            elif 3 in skipList:
                intensity_binsearch = (weight[0]*intensity_List[0]+weight[1]*intensity_List[1]+weight[2]*intensity_List[2])/(weight[0]+weight[1]+weight[2])
        
        elif len(skipList) == 2:
            if 0 in skipList and 1 in skipList:
                intensity_binsearch = (weight[2]*intensity_List[2]+weight[3]*intensity_List[3])/(weight[2]+weight[3])
            elif 0 in skipList and 2 in skipList:
                intensity_binsearch = (weight[1]*intensity_List[1]+weight[3]*intensity_List[3])/(weight[1]+weight[3])
            elif 0 in skipList and 3 in skipList:
                intensity_binsearch = (weight[1]*intensity_List[1]+weight[2]*intensity_List[2])/(weight[1]+weight[2])
            elif 1 in skipList and 2 in skipList:
                intensity_binsearch = (weight[0]*intensity_List[0]+weight[3]*intensity_List[3])/(weight[0]+weight[3])
            elif 1 in skipList and 3 in skipList:
                intensity_binsearch = (weight[0]*intensity_List[0]+weight[2]*intensity_List[2])/(weight[0]+weight[2])
            elif 2 in skipList and 3 in skipList:
                intensity_binsearch = (weight[0]*intensity_List[0]+weight[1]*intensity_List[1])/(weight[0]+weight[1])
        
        elif len(skipList) == 3:
            if 0 not in skipList:
                intensity_binsearch = intensity_List[0]
            elif 1 not in skipList:
                intensity_binsearch = intensity_List[1]
            elif 2 not in skipList:
                intensity_binsearch = intensity_List[2]
            elif 3 not in skipList:
                intensity_binsearch = intensity_List[3]
        
        if bool(np.isnan(intensity_binsearch)):
            print(f"q = {x0}, q_index = {q_index}, chi = {y0}, chi_index = {chi_index}")
        return intensity_binsearch
    
    def plot_pixel_map_px(self,
                       data: pd.DataFrame,
                       x: str,
                       y: str,
                       z: str,
                       colorscale: str, 
                       x_label: str = None,
                       y_label: str = None,
                       z_label: str = None,
                       xlim: tuple = None,
                       ylim: tuple = None,
                       log_scale: bool = False,
                       z_lower_cuttoff: float = None,
                       aspect: str = 'equal',
                       **kwargs) -> go.Figure:
        
        """
        Function for plotting a pixel map of data
        :param data: The data frame with the x, y, and z values
        :param x_col: The column name of the x values
        :param y_col: The column name of the y values
        :param z_col: The column name of the z values
        :param colorscale: The colorscale of the pixel map
        :param x_label: The x axis label
        :param y_label: The y axis label
        :param z_label: The z axis label
        :param xlim: The x limits of the plot
        :param ylim: The y limits of the plot
        :param width: The width of the plot
        :param height: The height of the plot
        :param title: The title of the plot
        :param log_scale: Whether to plot the z values on a log scale
        :param template: The plotly template to use
        :return: a plotly figure of the pixel map
        """
        if x not in data.columns or y not in data.columns or z not in data.columns:
            raise ValueError('The x, y, and z columns must be in the data frame')
        
        if z_lower_cuttoff is None and log_scale and min(data[z]) <= 0:
            raise ValueError('The z values must be positive to plot on a log scale. Either add a z_lower_cuttoff or remove the log_scale')
        
        data = data.query(f'{z} > 0')

        if xlim is not None:
            data = data.query(f'{x} >= {xlim[0]} and {x} <= {xlim[1]}')
        if ylim is not None:
            data = data.query(f'{y} >= {ylim[0]} and {y} <= {ylim[1]}')
        if z_lower_cuttoff is not None:
            data = data.query(f'{z} >= {z_lower_cuttoff}')

        square_data = data.pivot(index=y, columns=x, values=z)
        square_data = square_data.fillna(z_lower_cuttoff) if z_lower_cuttoff is not None else square_data.fillna(0)
        square_data = np.log10(square_data) if log_scale else square_data
        z_label = 'log(' + z_label + ')' if log_scale else z_label
        
        z_data = square_data.values
        x_vec = square_data.columns
        y_vec = square_data.index

        figure = px.imshow(z_data, x=x_vec, y=y_vec, color_continuous_scale=colorscale, aspect=aspect, **kwargs)
    
        if x_label is not None:
            figure.update_xaxes(title_text=x_label)
        if y_label is not None:
            figure.update_yaxes(title_text=y_label)
        if z_label is not None:
            figure.update_layout(coloraxis_colorbar={'title': z_label})

        return figure
    
    def plot_pixel_map_hv(self,
                       data: pd.DataFrame,
                       x: str,
                       y: str,
                       z: str,
                       log_scale: bool = False,
                       aspect: str = 'equal',
                       **kwargs):
        
        """
        Function for plotting a pixel map of data
        ùparam data: The data frame with the x, y, and z values
        :param x_col: The column name of the x values
        :param y_col: The column name of the y values
        :param z_col: The column name of the z values
        :param log_scale: Whether to plot the z values on a log scale
        :param aspect: The aspect ratio of the plot
        :return: a holoviews image of the pixel map
        """

        import holoviews as hv
        hv.extension('bokeh')        
        
        if x not in data.columns or y not in data.columns or z not in data.columns:
            raise ValueError('The x, y, and z columns must be in the data frame')
     

        square_data = data.pivot(index=y, columns=x, values=z)
        
        z_data = square_data.values
        x_vec = square_data.columns
        y_vec = square_data.index
        figure = hv.Image((x_vec, y_vec, z_data), kdims=['x', 'y'], vdims=['z']).opts(aspect = aspect,
                                                                                      logz = log_scale,
                                                                                      **kwargs)

        return figure
        
    
    def plot_contour_map(self, 
                         data: pd.DataFrame,
                         x: str,
                         y: str,
                         z: str,
                         colorscale: str,
                         x_label: str = None,
                         y_label: str = None,
                         z_label: str = None,
                         xlim: tuple = None,
                         ylim: tuple = None,
                         width: int = None,
                         height: int = None,
                         title = None,
                         ncontours: int = 200,
                         log_scale: bool = False, 
                         z_lower_cuttoff: float = None,
                         template: str = None) -> go.Figure:
        """
        Function for plotting a contour map of data
        :param data: The data frame with the x, y, and z values
        :param x_col: The column name of the x values
        :param y_col: The column name of the y values
        :param z_col: The column name of the z values
        :param colorscale: The colorscale of the contour map
        :param x_label: The x axis label
        :param y_label: The y axis label
        :param z_label: The z axis label
        :param xlim: The x limits of the plot
        :param ylim: The y limits of the plot
        :param width: The width of the plot
        :param height: The height of the plot
        :param title: The title of the plot
        :param ncontours: The number of contours to plot
        :param template: The plotly template to use
        :param log_scale: Whether to plot the z values on a log scale
        :return: a plotly figure of the contour map
        """
        if x not in data.columns or y not in data.columns or z not in data.columns:
            raise ValueError('The x, y, and z columns must be in the data frame')
        
        data = data.query(f'{z} > 0')

        if xlim is not None:
            data = data.query(f'{x} >= {xlim[0]} and {x} <= {xlim[1]}')
        if ylim is not None:
            data = data.query(f'{y} >= {ylim[0]} and {y} <= {ylim[1]}')
        if z_lower_cuttoff is not None:
            data = data.query(f'{z} >= {z_lower_cuttoff}')
        if log_scale:
            data[z] = np.log10(data[z])
            z_label = 'log(' + z_label + ')'

        figure = go.Figure()   
        figure.add_trace(go.Contour(x=data[x], y=data[y], z=data[z], colorscale=colorscale, contours_showlines=False, ncontours=ncontours))

        if title is not None:
            figure.update_layout(title_text=title)
        if width is not None:
            figure.update_layout(width=width)
        if height is not None:
            figure.update_layout(height=height)
        if x_label is not None:
            figure.update_xaxes(title_text=x_label)
        if y_label is not None:
            figure.update_yaxes(title_text=y_label)
        if template is not None:
            figure.update_layout(template=template)
        if z_label is not None:
            figure.update_traces(colorbar={'title': z_label})

        return figure
    
        

class ElectrochemicalMeasurement(Measurement):
    """
    A general electrochemical measurement class
    Main contributors:
    Nicholas Siemons
    Contributors:
    """
    def __init__(self, 
                 potential_reference: str,
                 electrolyte: Electrolyte = None,
                 metadata: dict = None) -> None:
        
        super().__init__(metadata=metadata)

        self._electrolyte = electrolyte
        self._check_potential_reference(potential_reference)

    def _check_potential_reference(self, value: str):

        value = value.lower().replace('/', '')

        allowed_references = ['agagcl', 'she', 'rhe']

        if value in allowed_references:
            self._potential_reference = value
        else:
            raise ValueError(f'allowed references are {allowed_references}')

    @property
    def electrolyte(self) -> Electrolyte:
        return self._electrolyte

    @property
    def potential_reference(self) -> str:
        """
        convert stored potential reference str to standard display titles 
        """
        outputs = {
            'rhe': 'RHE',
            'she': 'SHE',
            'agagcl': 'Ag/AgCl'
        }

        return outputs[self._potential_reference]

    def set_potential_reference(self, 
                                reference: str, 
                                open_circuit_potential: float = None
                                ):
        """
        Function to change between reference potentials
        
        :param potential_reference: str, reference to change to 
        :param reference: str, the current reference
        :param open_circuit_potential: float, the OCV measured between rhe electrode and agagcl reference
        """
        if open_circuit_potential is None and (self._potential_reference == 'rhe' or reference == 'rhe'):
            raise ValueError("open_circuit_potential cannot be None when converting to/from RHE")
    
        reference = reference.lower().replace('/', '')
        valid_references = {'rhe', 'she', 'agagcl'}

        data = self._data.copy()

        if reference not in valid_references:
            raise ValueError(f'Please enter one of {valid_references}')

        if self._potential_reference == 'agagcl' and reference == 'she':
            data['potential'] += 0.197
        elif self._potential_reference == 'agagcl' and reference == 'rhe':
            data['potential'] += open_circuit_potential
        elif self._potential_reference == 'she' and reference == 'rhe':
            data['potential'] -= open_circuit_potential + 0.197
        elif self._potential_reference == 'she' and reference == 'agagcl':
            data['potential'] -= 0.197
        elif self._potential_reference == 'rhe' and reference == 'agagcl':
            data['potential'] -= open_circuit_potential
        elif self._potential_reference == 'rhe' and reference == 'she':
            data['potential'] += open_circuit_potential + 0.197
        else: 
            raise ValueError(f'Chaning potential reference unsuccessful')

        self._potential_reference = reference 
        self._data = data

        return self

    
    @staticmethod
    def _find_voltage_peaks(data):
        """
        Function to find the voltage peaks in a data set
        """
        data = data.sort_values('time').reset_index(drop=True)

        # get the potential range and peak height
        potential_max = data['potential'].max()
        potential_min = data['potential'].min()
        potential_range = potential_max - potential_min
        peak_height = 0.1 * potential_range

        # find the peaks
        positive_peaks, _ = find_peaks(data['potential'], prominence=peak_height)
        negative_peaks, _ = find_peaks(-data['potential'], prominence=peak_height)

        # determine the direction of the first peak
        if positive_peaks[0] < negative_peaks[0]:
            direction = 'oxidation'
        else:
            direction = 'reduction'

        # assign the direction to each point and determine its segment
        direction_list = []
        segment_list = []
        segment = 0
        for index, row in data.iterrows():
            direction_list.append(direction)
            segment_list.append(segment)
            if index in positive_peaks:
                direction = 'reduction'
                segment += 1
            if index in negative_peaks:
                direction = 'oxidation'
                segment += 1
        
        data['direction'] = direction_list
        data['segment'] = segment_list
        data['cycle'] = ((data['segment']-1) // 2) + 1 

        return data


class ScatteringMeasurement(Measurement):

    def __init__(self, metadata: dict = None) -> None:
        super().__init__(metadata=metadata)


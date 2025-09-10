from __future__ import annotations
from Materials_Data_Analytics.metadynamics.free_energy import FreeEnergySpace


class Universe:
    """
    The universe class.  Everything starts with the universe. This class will be expanded as new functionality is added.
    """
    def __init__(self, tpr_file: str = None, xtc_file: str = None, fes: FreeEnergySpace | list[FreeEnergySpace] = None):
        """
        The current idea is that we have our own universe, but a lot of the internal working will use the mda universe, so that is stored as a
        separate attribute. Also stored will be a list of free energy shapes. These will have different temperatures, or there may just be one at
        a given temperature. The only thing that can change between them is the temperature as they must exist for the same universe.

        :param tpr_file:
        :param xtc_file:
        :param fes:
        :return:
        """
        if type(fes) == FreeEnergySpace:
            self._fes = [fes]
        elif type(fes) == list:
            self._fes = fes
        elif fes is None:
            self._fes = fes
        else:
            raise ValueError("fes must be a FreeEnergyShape, or list of FreeEnergyShapes")

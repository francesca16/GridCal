# This file is part of GridCal.
#
# GridCal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GridCal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GridCal.  If not, see <http://www.gnu.org/licenses/>.

from enum import Enum


class BranchType(Enum):
    Branch = 'branch'
    Line = 'line'
    DCLine = 'DC-line'
    VSC = 'VSC'
    Transformer = 'transformer'
    Reactance = 'reactance'
    Switch = 'switch'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return BranchType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class TransformerControlType(Enum):

    fixed = '0:Fixed'
    power = '1:power'
    # v_from = '3:Control V from'
    v_to = '4:Control V to'
    # power_v_from = '5:Control Angle + V from'
    power_v_to = '6:Control Angle + V to'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return TransformerControlType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ConverterControlType(Enum):

    # Type I
    # theta_vac = '1:Angle+Vac'
    # pf_qac = '2:Pflow + Qflow'
    # pf_vac = '3:Pflow + Vac'
    #
    # # Type II
    # vdc_qac = '4:Vdc+Qflow'
    # vdc_vac = '5:Vdc+Vac'
    #
    # # type III
    # vdc_droop_qac = '6:VdcDroop+Qac'
    # vdc_droop_vac = '7:VdcDroop+Vac'

    type_1_free = '1a:Free'
    type_1_pf = '1b:Pflow'
    type_1_qf = '1c:Qflow'
    type_1_vac = '1d:Vac'
    type_2_vdc = '2a:Vdc'
    type_2_vdc_pf = '2b:Vdc+Pflow'
    type_3 = '3a:Droop'
    type_4 = '4a:Droop-slack'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return ConverterControlType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class TimeFrame(Enum):
    Continuous = 'Continuous'


class DeviceType(Enum):
    CircuitDevice = 'Circuit'
    BusDevice = 'Bus'
    BranchDevice = 'Branch'
    BranchTypeDevice = 'Branch template'
    LineDevice = 'Line'
    LineTypeDevice = 'Line Template'
    Transformer2WDevice = 'Transformer'
    HVDCLineDevice = 'HVDC Line'
    DCLineDevice = 'DC line'
    VscDevice = 'VSC'
    BatteryDevice = 'Battery'
    LoadDevice = 'Load'
    GeneratorDevice = 'Generator'
    StaticGeneratorDevice = 'Static Generator'
    ShuntDevice = 'Shunt'
    ExternalGridDevice = 'External grid'
    WireDevice = 'Wire'
    SequenceLineDevice = 'Sequence line'
    UnderGroundLineDevice = 'Underground line'
    TowerDevice = 'Tower'
    TransformerTypeDevice = 'Transformer type'

    GenericArea = 'Generic Area'
    SubstationDevice = 'Substation'
    AreaDevice = 'Area'
    ZoneDevice = 'Zone'
    CountryDevice = 'Country'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return DeviceType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class GeneratorTechnologyType(Enum):
    Nuclear = 'Nuclear'
    CombinedCycle = 'Combined cycle'
    OpenCycleGasTurbine = 'Open Cycle Gas Turbine'
    SteamCycle = 'Steam cycle (coal, waste, etc)'
    Photovoltaic = 'Photovoltaic'
    SolarConcentrated = 'Concentrated solar power'
    OnShoreWind = 'On-shore wind power'
    OffShoreWind = 'Off-shore wind power'
    HydroReservoir = 'Hydro with reservoir'
    HydroRunOfRiver = 'Hydro run-of-river'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return GeneratorTechnologyType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
# GridCal
# Copyright (C) 2015 - 2024 Santiago Peñate Vera
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from enum import Enum


class BusMode(Enum):
    """
    Emumetarion of bus modes
    """
    PQ = 1  # control P, Q
    PV = 2  # Control P, Vm
    Slack = 3  # Contol Vm, Va (slack)
    PQV = 4  # control P, Q and Vm
    D = 5  # only control the voltage angle (Va)

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return BusMode[s]
        except KeyError:
            return s


class CpfStopAt(Enum):
    Nose = 'Nose'
    ExtraOverloads = 'Extra overloads'
    Full = 'Full curve'


class CpfParametrization(Enum):
    Natural = 'Natural'
    ArcLength = 'Arc Length'
    PseudoArcLength = 'Pseudo Arc Length'


class ExternalGridMode(Enum):
    """
    Modes of operation of external grids
    """
    PQ = "PQ"
    PV = "PV"
    VD = "VD"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return ExternalGridMode[s]
        except KeyError:
            return s


class InvestmentEvaluationMethod(Enum):
    """
    Investment evaluation methods
    """
    Independent = "Independent"
    Hyperopt = "Hyperopt"
    MVRSM = "MVRSM"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return InvestmentEvaluationMethod[s]
        except KeyError:
            return s


class BranchImpedanceMode(Enum):
    """
    Enumeration of branch impedance modes
    """
    Specified = 0
    Upper = 1
    Lower = 2

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return BranchImpedanceMode[s]
        except KeyError:
            return s


class SolverType(Enum):
    """
    Refer to the :ref:`Power Flow section<power_flow>` for details about the different
    algorithms supported by **GridCal**.
    """

    NR = 'Newton Raphson'
    NRD = 'Newton Raphson Decoupled'
    NRFD_XB = 'Fast decoupled XB'
    NRFD_BX = 'Fast decoupled BX'
    GAUSS = 'Gauss-Seidel'
    DC = 'Linear DC'
    HELM = 'Holomorphic Embedding'
    ZBUS = 'Z-Gauss-Seidel'
    PowellDogLeg = "Powell's Dog Leg"
    IWAMOTO = 'Iwamoto-Newton-Raphson'
    CONTINUATION_NR = 'Continuation-Newton-Raphson'
    HELMZ = 'HELM-Z'
    LM = 'Levenberg-Marquardt'
    FASTDECOUPLED = 'Fast decoupled'
    LACPF = 'Linear AC'
    LINEAR_OPF = 'Linear OPF'
    NONLINEAR_OPF = 'Nonlinear OPF'
    SIMPLE_OPF = 'Simple dispatch'
    Proportional_OPF = 'Proportional OPF'
    NRI = 'Newton-Raphson in current'
    DYCORS_OPF = 'DYCORS OPF'
    GA_OPF = 'Genetic Algorithm OPF'
    NELDER_MEAD_OPF = 'Nelder Mead OPF'
    BFS = 'Backwards-Forward substitution'
    BFS_linear = 'Backwards-Forward substitution (linear)'
    Constant_Impedance_linear = 'Constant impedance linear'
    NoSolver = 'No Solver'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return SolverType[s]
        except KeyError:
            return s


class ReactivePowerControlMode(Enum):
    """
    The :ref:`ReactivePowerControlMode<q_control>` offers 3 modes to control how
    :ref:`Generator<generator>` objects supply reactive power:

    **NoControl**: In this mode, the :ref:`generators<generator>` don't try to regulate
    the voltage at their :ref:`bus<bus>`.

    **Direct**: In this mode, the :ref:`generators<generator>` try to regulate the
    voltage at their :ref:`bus<bus>`. **GridCal** does so by applying the following
    algorithm in an outer control loop. For grids with numerous
    :ref:`generators<generator>` tied to the same system, for example wind farms, this
    control method sometimes fails with some :ref:`generators<generator>` not trying
    hard enough*. In this case, the simulation converges but the voltage controlled
    :ref:`buses<bus>` do not reach their target voltage, while their
    :ref:`generator(s)<generator>` haven't reached their reactive power limit. In this
    case, the slower **Iterative** control mode may be used (see below).

        ON PV-PQ BUS TYPE SWITCHING LOGIC IN POWER FLOW COMPUTATION
        Jinquan Zhao

        1) Bus i is a PQ bus in the previous iteration and its
           reactive power was fixed at its lower limit:

            If its voltage magnitude Vi >= Viset, then

                it is still a PQ bus at current iteration and set Qi = Qimin .

                If Vi < Viset , then

                    compare Qi with the upper and lower limits.

                    If Qi >= Qimax , then
                        it is still a PQ bus but set Qi = Qimax .
                    If Qi <= Qimin , then
                        it is still a PQ bus and set Qi = Qimin .
                    If Qimin < Qi < Qi max , then
                        it is switched to PV bus, set Vinew = Viset.

        2) Bus i is a PQ bus in the previous iteration and
           its reactive power was fixed at its upper limit:

            If its voltage magnitude Vi <= Viset , then:
                bus i still a PQ bus and set Q i = Q i max.

                If Vi > Viset , then

                    Compare between Qi and its upper/lower limits

                    If Qi >= Qimax , then
                        it is still a PQ bus and set Q i = Qimax .
                    If Qi <= Qimin , then
                        it is still a PQ bus but let Qi = Qimin in current iteration.
                    If Qimin < Qi < Qimax , then
                        it is switched to PV bus and set Vinew = Viset

        3) Bus i is a PV bus in the previous iteration.

            Compare Q i with its upper and lower limits.

            If Qi >= Qimax , then
                it is switched to PQ and set Qi = Qimax .
            If Qi <= Qimin , then
                it is switched to PQ and set Qi = Qimin .
            If Qi min < Qi < Qimax , then
                it is still a PV bus.

    **Iterative**: As mentioned above, the **Direct** control mode may not yield
    satisfying results in some isolated cases. The **Direct** control mode tries to
    jump to the final solution in a single or few iterations, but in grids where a
    significant change in reactive power at one :ref:`generator<generator>` has a
    significant impact on other :ref:`generators<generator>`, additional iterations may
    be required to reach a satisfying solution.

    Instead of trying to jump to the final solution, the **Iterative** mode raises or
    lowers each :ref:`generator's<generator>` reactive power incrementally. The
    increment is determined using a logistic function based on the difference between
    the current :ref:`bus<bus>` voltage its target voltage. The steepness factor
    :code:`k` of the logistic function was determined through trial and error, with the
    intent of reducing the number of iterations while avoiding instability. Other
    values may be specified in :ref:`PowerFlowOptions<pf_options>`.

    The :math:`Q_{Increment}` in per unit is determined by:

    .. math::

        Q_{Increment} = 2 * \\left[\\frac{1}{1 + e^{-k|V_2 - V_1|}}-0.5\\right]

    Where:

        k = 30 (by default)

    """
    NoControl = "NoControl"
    Direct = "Direct"
    Iterative = "Iterative"


class TapsControlMode(Enum):
    """
    The :ref:`TapsControlMode<taps_control>` offers 3 modes to control how
    :ref:`transformers<transformer>`' :ref:`tap changer<tap_changer>` regulate
    voltage on their regulated :ref:`bus<bus>`:

    **NoControl**: In this mode, the :ref:`transformers<transformer>` don't try to
    regulate the voltage at their :ref:`bus<bus>`.

    **Direct**: In this mode, the :ref:`transformers<transformer>` try to regulate
    the voltage at their bus. **GridCal** does so by jumping straight to the tap that
    corresponds to the desired transformation ratio, or the highest or lowest tap if
    the desired ratio is outside of the tap range.

    This behavior may fail in certain cases, especially if both the
    :ref:`TapControlMode<taps_control>` and :ref:`ReactivePowerControlMode<q_control>`
    are set to **Direct**. In this case, the simulation converges but the voltage
    controlled :ref:`buses<bus>` do not reach their target voltage, while their
    :ref:`generator(s)<generator>` haven't reached their reactive power limit. When
    this happens, the slower **Iterative** control mode may be used (see below).

    **Iterative**: As mentioned above, the **Direct** control mode may not yield
    satisfying results in some isolated cases. The **Direct** control mode tries to
    jump to the final solution in a single or few iterations, but in grids where a
    significant change of tap at one :ref:`transformer<transformer>` has a
    significant impact on other :ref:`transformers<transformer>`, additional
    iterations may be required to reach a satisfying solution.

    Instead of trying to jump to the final solution, the **Iterative** mode raises or
    lowers each :ref:`transformer's<transformer>` tap incrementally.
    """

    NoControl = "NoControl"
    Direct = "Direct"
    Iterative = "Iterative"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return TapsControlMode[s]
        except KeyError:
            return s


class SyncIssueType(Enum):
    """
    Sync issues enumeration
    """
    Added = 'Added'
    Deleted = 'Deleted'
    Conflict = 'Conflict'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return SyncIssueType[s]
        except KeyError:
            return s


class EngineType(Enum):
    """
    Available engines enumeration
    """
    GridCal = 'GridCal'
    Bentayga = 'Bentayga'
    NewtonPA = 'Newton Power Analytics'
    PGM = 'Power Grid Model'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return EngineType[s]
        except KeyError:
            return s


class MIPSolvers(Enum):
    """
    MIP solvers enumeration
    """
    GLOP = "GLOP"
    CBC = 'CBC'
    HIGHS = 'HIGHS'
    SCIP = 'SCIP'
    CPLEX = 'CPLEX'
    GUROBI = 'GUROBI'
    XPRESS = 'XPRESS'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return MIPSolvers[s]
        except KeyError:
            return s


class TimeGrouping(Enum):
    """
    Time groupings enumeration
    """
    NoGrouping = 'No grouping'
    Monthly = 'Monthly'
    Weekly = 'Weekly'
    Daily = 'Daily'
    Hourly = 'Hourly'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return TimeGrouping[s]
        except KeyError:
            return s


class ZonalGrouping(Enum):
    """
    Zonal groupings enumeration
    """
    NoGrouping = 'No grouping'
    Area = 'Area'
    All = 'All (copper plate)'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return ZonalGrouping[s]
        except KeyError:
            return s


class ContingencyMethod(Enum):
    """
    Enumeratio of contingency calculation engines
    """
    PowerFlow = 'Power flow'
    HELM = 'HELM'
    PTDF = 'PTDF'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return ZonalGrouping[s]
        except KeyError:
            return s


class DiagramType(Enum):
    """
    Types of diagrams
    """
    BusBranch = 'bus-branch'
    SubstationLineMap = 'substation-line-map'
    NodeBreaker = 'node-breaker'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return DiagramType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class TransformerControlType(Enum):
    """
    Transformer control types
    """
    fixed = '0:Fixed'
    Pt = '1:Pt'
    Qt = '2:Qt'
    PtQt = '3:Pt+Qt'
    Vt = '4:Vt'
    PtVt = '5:Pt+Vt'

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
    """
    Converter control types
    """
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

    type_0_free = '0:Free'

    type_I_1 = '1:Vac'
    type_I_2 = '2:Pdc+Qac'
    type_I_3 = '3:Pdc+Vac'

    type_II_4 = '4:Vdc+Qac'
    type_II_5 = '5:Vdc+Vac'

    type_III_6 = '6:Droop+Qac'
    type_III_7 = '7:Droop+Vac'

    type_IV_I = '8:Vdc'
    type_IV_II = '9:Pdc'

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


class HvdcControlType(Enum):
    """
    Simple HVDC control types
    """
    type_0_free = '0:Free'
    type_1_Pset = '1:Pdc'

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


class GenerationNtcFormulation(Enum):
    """
    NTC formulation type
    """
    Proportional = 'Proportional'
    Optimal = 'Optimal'


class TimeFrame(Enum):
    """
    Time frame
    """
    Continuous = 'Continuous'


class FaultType(Enum):
    """
    Short circuit type
    """
    ph3 = '3x'
    LG = 'LG'
    LL = 'LL'
    LLG = 'LLG'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return FaultType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class WindingsConnection(Enum):
    """
    Transformer windings connection types
    """
    # G: grounded star
    # S: ungrounded star
    # D: delta
    GG = 'GG'
    GS = 'GS'
    GD = 'GD'
    SS = 'SS'
    SD = 'SD'
    DD = 'DD'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return WindingsConnection[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class DeviceType(Enum):
    """
    Device types
    """
    NoDevice = "NoDevice"
    CircuitDevice = 'Circuit'
    BusDevice = 'Bus'
    BranchDevice = 'Branch'
    BranchTypeDevice = 'Branch template'
    LineDevice = 'Line'
    LineTypeDevice = 'Line Template'
    Transformer2WDevice = 'Transformer'
    Transformer3WDevice = 'Transformer3W'
    WindingDevice = 'Winding'
    HVDCLineDevice = 'HVDC Line'
    DCLineDevice = 'DC line'
    VscDevice = 'VSC'
    BatteryDevice = 'Battery'
    LoadDevice = 'Load'
    GeneratorDevice = 'Generator'
    StaticGeneratorDevice = 'Static Generator'
    ShuntDevice = 'Shunt'
    UpfcDevice = 'UPFC'  # unified power flow controller
    ExternalGridDevice = 'External grid'
    WireDevice = 'Wire'
    SequenceLineDevice = 'Sequence line'
    UnderGroundLineDevice = 'Underground line'
    OverheadLineTypeDevice = 'Tower'
    TransformerTypeDevice = 'Transformer type'
    SwitchDevice = 'Switch'

    GenericArea = 'Generic Area'
    SubstationDevice = 'Substation'
    ConnectivityNodeDevice = 'Connectivity Node'
    AreaDevice = 'Area'
    ZoneDevice = 'Zone'
    CountryDevice = 'Country'
    BusBarDevice = 'BusBar'

    Technology = 'Technology'
    TechnologyGroup = 'Technology Group'
    TechnologyCategory = 'Technology Category'

    ContingencyDevice = 'Contingency'
    ContingencyGroupDevice = 'Contingency Group'

    InvestmentDevice = 'Investment'
    InvestmentsGroupDevice = 'Investments Group'

    FuelDevice = 'Fuel'
    EmissionGasDevice = 'Emission'

    GeneratorEmissionAssociation = 'Generator Emission'
    GeneratorFuelAssociation = 'Generator Fuel'
    GeneratorTechnologyAssociation = 'Generator Technology'

    DiagramDevice = 'Diagram'

    GeneratorQCurve = 'Generator Q curve'

    FluidInjectionDevice = 'Fluid Injection'
    FluidTurbineDevice = 'Fluid Turbine'
    FluidPumpDevice = 'Fluid Pump'
    FluidP2XDevice = 'Fluid P2X'
    FluidPathDevice = 'Fluid path'
    FluidNodeDevice = 'Fluid node'

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


class BuildStatus(Enum):
    """
    Asset build status options
    """
    Commissioned = 'Commissioned'  # Already there, not planned for decommissioning
    Decommissioned = 'Decommissioned'  # Already retired (does not exist)
    Planned = 'Planned'  # Planned for commissioning some time, does not exist yet)
    Candidate = 'Candidate'  # Candidate for commissioning, does not exist yet, might be selected for commissioning
    PlannedDecommission = 'PlannedDecommission'  # Already there, but it has been selected for decommissioning

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return BuildStatus[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class StudyResultsType(Enum):
    """
    Types of simulation results
    """
    PowerFlow = 'PowerFlow'
    PowerFlowTimeSeries = 'PowerFlowTimeSeries'
    OptimalPowerFlow = 'PowerFlow'
    OptimalPowerFlowTimeSeries = 'PowerFlowTimeSeries'
    ShortCircuit = 'ShortCircuit'
    ContinuationPowerFlow = 'ContinuationPowerFlow'
    ContingencyAnalysis = 'ContingencyAnalysis'
    ContingencyAnalysisTimeSeries = 'ContingencyAnalysisTimeSeries'
    SigmaAnalysis = 'SigmaAnalysis'
    LinearAnalysis = 'LinearAnalysis'
    LinearAnalysisTimeSeries = 'LinearAnalysisTimeSeries'
    AvailableTransferCapacity = 'AvailableTransferCapacity'
    AvailableTransferCapacityTimeSeries = 'AvailableTransferCapacityTimeSeries'
    Clustering = 'Clustering'
    StateEstimation = 'StateEstimation'
    InputsAnalysis = 'InputsAnalysis'
    InvestmentEvaluations = 'InvestmentEvaluations'
    NetTransferCapacity = 'NetTransferCapacity'
    NetTransferCapacityTimeSeries = 'NetTransferCapacityTimeSeries'
    StochasticPowerFlow = 'StochasticPowerFlow'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return StudyResultsType[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class AvailableTransferMode(Enum):
    """
    AvailableTransferMode
    """

    Generation = "Generation"
    InstalledPower = "InstalledPower"
    Load = "Load"
    GenerationAndLoad = "GenerationAndLoad"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return AvailableTransferMode[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class InvestmentsEvaluationObjectives(Enum):
    """
    Types of investment optimization objectives
    """
    PowerFlow = 'PowerFlow'
    TimeSeriesPowerFlow = 'TimeSeriesPowerFlow'
    OptimalPowerFlow = 'OptimalPowerFlow'
    TimeSeriesOptimalPowerFlow = 'TimeSeriesOptimalPowerFlow'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return InvestmentsEvaluationObjectives[s]
        except KeyError:
            return s

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class LogSeverity(Enum):
    """
    Enumeration of logs severities
    """
    Error = 'Error'
    Warning = 'Warning'
    Information = 'Information'
    Divergence = 'Divergence'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        """

        :param s:
        :return:
        """
        try:
            return LogSeverity[s]
        except KeyError:
            return s
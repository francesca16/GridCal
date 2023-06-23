# GridCal
# Copyright (C) 2022 Santiago Peñate Vera
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

import time
import numpy as np
from typing import Union
from itertools import combinations
from GridCal.Engine.Core.multi_circuit import MultiCircuit
from GridCal.Engine.Core.numerical_circuit import compile_numerical_circuit_at, NumericalCircuit
import GridCal.Engine.basic_structures as bs
from GridCal.Engine.basic_structures import DateVec, IntVec, StrVec, Vec, Mat, CxVec, IntMat, CxMat
from GridCal.Engine.Simulations.ContingencyAnalysis.contingency_analysis_results import ContingencyAnalysisResults
from GridCal.Engine.Simulations.NonLinearFactors.nonlinear_analysis import NonLinearAnalysis
from GridCal.Engine.Simulations.driver_types import SimulationTypes
from GridCal.Engine.Simulations.driver_template import DriverTemplate
from GridCal.Engine.Simulations.PowerFlow.power_flow_worker import get_hvdc_power, multi_island_pf_nc
from GridCal.Engine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions, SolverType
from GridCal.Engine.Simulations.LinearFactors.linear_analysis import LinearAnalysis


def enumerate_states_n_k(m: int, k: int = 1):
    """
    Enumerates the states to produce the so called N-k failures
    :param m: number of Branches
    :param k: failure level
    :return: binary array (number of states, m)
    """

    # num = int(math.factorial(k) / math.factorial(m-k))
    states = list()
    indices = list()
    arr = np.ones(m, dtype=int).tolist()

    idx = list(range(m))
    for k1 in range(k + 1):
        for failed in combinations(idx, k1):
            indices.append(failed)
            arr2 = arr.copy()
            for j in failed:
                arr2[j] = 0
            states.append(arr2)

    return np.array(states), indices


class ContingencyAnalysisOptions:

    def __init__(
            self,
            distributed_slack: bool = True,
            correct_values: bool = True,
            use_provided_flows: bool = False,
            Pf: Vec = None,  # TODO: Consider moving this to the only place where it is used: ContingencyAnalysisDriver
            pf_results=None,
            engine=bs.ContingencyEngine.PowerFlow,
            pf_options=PowerFlowOptions(SolverType.DC)
    ):
        """

        :param distributed_slack:
        :param correct_values:
        :param use_provided_flows:
        :param Pf:
        :param pf_results:
        :param engine:
        :param pf_options:
        """
        self.distributed_slack = distributed_slack

        self.correct_values = correct_values

        self.use_provided_flows = use_provided_flows

        self.Pf: Vec = Pf

        self.pf_results = pf_results

        self.engine = engine

        self.pf_options = pf_options


class ContingencyAnalysisDriver(DriverTemplate):
    name = 'Contingency Analysis'
    tpe = SimulationTypes.ContingencyAnalysis_run

    def __init__(self, grid: MultiCircuit, options: ContingencyAnalysisOptions):
        """
        N - k class constructor
        @param grid: MultiCircuit Object
        @param options: N-k options
        @:param pf_options: power flow options
        """
        DriverTemplate.__init__(self, grid=grid)

        # Options to use
        self.options = options

        # N-K results
        self.results = ContingencyAnalysisResults(
            ncon=0,
            nbus=0,
            nbr=0,
            bus_names=(),
            branch_names=(),
            bus_types=(),
            con_names=()
        )

    def get_steps(self):
        """
        Get variations list of strings
        """
        if self.results is not None:
            return ['#' + v for v in self.results.branch_names]
        else:
            return list()

    def n_minus_k(self, t=None):
        """
        Run N-1 simulation in series
        :param t: time index, if None the snapshot is used
        :return: returns the results
        """
        # set the numerical circuit
        numerical_circuit = compile_numerical_circuit_at(self.grid, t_idx=t)

        if self.options.pf_options is None:
            pf_opts = PowerFlowOptions(
                solver_type=SolverType.DC,
                ignore_single_node_islands=True
            )

        else:
            pf_opts = self.options.pf_options

        # declare the results
        results = ContingencyAnalysisResults(
            ncon=len(self.grid.contingency_groups),
            nbr=numerical_circuit.nbr,
            nbus=numerical_circuit.nbus,
            branch_names=numerical_circuit.branch_names,
            bus_names=numerical_circuit.bus_names,
            bus_types=numerical_circuit.bus_types,
            con_names=self.grid.get_contingency_group_names()
        )

        # get contingency groups dictionary
        cg_dict = self.grid.get_contingency_group_dict()

        branches_dict = self.grid.get_branches_wo_hvdc_dict()

        # keep the original states
        original_br_active = numerical_circuit.branch_data.active.copy()
        original_gen_active = numerical_circuit.generator_data.active.copy()
        original_gen_p = numerical_circuit.generator_data.p.copy()

        # run 0
        pf_res_0 = multi_island_pf_nc(
            nc=numerical_circuit,
            options=pf_opts
        )

        # for each contingency group
        for ic, contingency_group in enumerate(self.grid.contingency_groups):

            # get the group's contingencies
            contingencies = cg_dict[contingency_group.idtag]

            # apply the contingencies
            for cnt in contingencies:

                # search for the contingency in the Branches
                if cnt.device_idtag in branches_dict:
                    br_idx = branches_dict[cnt.device_idtag]

                    if cnt.prop == 'active':
                        numerical_circuit.branch_data.active[br_idx] = int(cnt.value)
                    else:
                        print(f'Unknown contingency property {cnt.prop} at {cnt.name} {cnt.idtag}')
                else:
                    pass

            # report progress
            if t is None:
                self.progress_text.emit(f'Contingency group: {contingency_group.name}')
                self.progress_signal.emit((ic + 1) / len(self.grid.contingency_groups) * 100)

            # run
            pf_res = multi_island_pf_nc(
                nc=numerical_circuit,
                options=pf_opts,
                V_guess=pf_res_0.voltage
            )

            results.Sf[ic, :] = pf_res.Sf
            results.S[ic, :] = pf_res.Sbus
            results.loading[ic, :] = pf_res.loading

            # revert the states for the next run
            numerical_circuit.branch_data.active = original_br_active.copy()
            numerical_circuit.generator_data.active = original_gen_active.copy()
            numerical_circuit.generator_data.p = original_gen_p.copy()

            if self.__cancel__:
                return results

        return results

    def n_minus_k_nl(self, t: Union[int, None] = None):
        """
        Run N-1 simulation in series with HELM, non-linear solution
        :param t: time index, if None the snapshot is used
        :return: returns the results
        """

        self.progress_text.emit('Analyzing outage distribution factors in a non-linear fashion...')
        nonlinear_analysis = NonLinearAnalysis(
            grid=self.grid,
            distributed_slack=self.options.distributed_slack,
            correct_values=self.options.correct_values,
            pf_options=self.options.pf_options,
            t_idx=t
        )

        nonlinear_analysis.run()

        # set the numerical circuit
        numerical_circuit = nonlinear_analysis.numerical_circuit

        # declare the results
        results = ContingencyAnalysisResults(
            ncon=len(self.grid.contingency_groups),
            nbr=numerical_circuit.nbr,
            nbus=numerical_circuit.nbus,
            branch_names=numerical_circuit.branch_names,
            bus_names=numerical_circuit.bus_names,
            bus_types=numerical_circuit.bus_types,
            con_names=self.grid.get_contingency_group_names()
        )

        # get the contingency branch indices
        br_idx = nonlinear_analysis.numerical_circuit.branch_data.get_contingency_enabled_indices()
        mon_idx = nonlinear_analysis.numerical_circuit.branch_data.get_monitor_enabled_indices()
        Pbus = numerical_circuit.get_injections(False).real
        PTDF = nonlinear_analysis.PTDF
        LODF = nonlinear_analysis.LODF

        # compute the branch Sf in "n"
        if self.options.use_provided_flows:
            flows_n = self.options.Pf

            if self.options.Pf is None:
                msg = 'The option to use the provided flows is enabled, but no flows are available'
                self.logger.add_error(msg)
                raise Exception(msg)
        else:
            flows_n = nonlinear_analysis.get_flows(numerical_circuit.Sbus)

        self.progress_text.emit('Computing loading...')

        for ic, c in enumerate(br_idx):  # branch that fails (contingency)
            results.Sf[mon_idx, c] = flows_n[mon_idx] + LODF[mon_idx, c] * flows_n[c]
            results.loading[mon_idx, c] = results.Sf[mon_idx, c] / (numerical_circuit.ContingencyRates[mon_idx] + 1e-9)
            results.S[c, :] = Pbus

            self.progress_signal.emit((ic + 1) / len(br_idx) * 100)

        results.lodf = LODF

        return results

    def run(self):
        """

        :return:
        """
        start = time.time()

        if self.options.engine == bs.ContingencyEngine.PowerFlow:
            self.results = self.n_minus_k()

        elif self.options.engine == bs.ContingencyEngine.PTDF:

            linear = LinearAnalysis(
                grid=self.grid, )
            self.results = self.n_minus_k_ptdf(numerical_circuit=nc)

        elif self.options.engine == bs.ContingencyEngine.HELM:
            self.results = self.n_minus_k_nl()

        else:
            self.results = self.n_minus_k()

        end = time.time()
        self.elapsed = end - start


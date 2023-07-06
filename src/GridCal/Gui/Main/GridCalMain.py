# GridCal
# Copyright (C) 2015 - 2023 Santiago Peñate Vera
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
import datetime as dtelib
import gc
import json
import sys
import ctypes
import threading
import os.path
import platform
import webbrowser
from collections import OrderedDict
from typing import List, Tuple, Dict, Union
import numpy as np
import networkx as nx
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pandas.plotting import register_matplotlib_converters
from warnings import warn
import darkdetect
import qdarktheme

# Engine imports
import GridCal.Engine.Core as core
import GridCal.Engine.Core.Devices as dev
import GridCal.Gui.Session.export_results_driver as exprtdrv
import GridCal.Gui.Session.file_handler as filedrv
import GridCal.Gui.Session.synchronization_driver as syncdrv
import GridCal.Engine.Simulations as sim
import GridCal.Gui.Visualization.visualization as viz
import GridCal.Engine.basic_structures as bs
import GridCal.Engine.grid_analysis as grid_analysis
from GridCal.Engine.IO.file_system import get_create_gridcal_folder
from GridCal.Engine.IO.contingency_parser import import_contingencies_from_json, export_contingencies_json_file
from GridCal.Engine.Core.Compilers.circuit_to_bentayga import BENTAYGA_AVAILABLE
from GridCal.Engine.Core.Compilers.circuit_to_newton_pa import NEWTON_PA_AVAILABLE, get_newton_mip_solvers_list
from GridCal.Engine.Core.Compilers.circuit_to_pgm import PGM_AVAILABLE
from GridCal.Engine.Simulations.driver_types import SimulationTypes
from GridCal.ThirdParty.ortools.ortools_extra import get_or_tools_available_solvers
from GridCal.Gui.Analysis.object_plot_analysis import object_histogram_analysis

# GUI imports
from PySide6 import QtGui, QtWidgets, QtCore
import GridCal.Gui.GuiFunctions as gf
from GridCal.Gui.Analysis.AnalysisDialogue import GridAnalysisGUI
from GridCal.Gui.BusViewer.bus_viewer_dialogue import BusViewerGUI
from GridCal.Gui.CoordinatesInput.coordinates_dialogue import CoordinatesInputGUI
from GridCal.Gui.GeneralDialogues import LogsDialogue, clear_qt_layout, NewProfilesStructureDialogue, ElementsDialogue, \
    TimeReIndexDialogue
from GridCal.Gui.GridEditorWidget import GridEditor
from GridCal.Gui.GridEditorWidget.messages import yes_no_question, error_msg, warning_msg, info_msg

from GridCal.Gui.Main.MainWindow import Ui_mainWindow, QMainWindow, QApplication
from GridCal.Gui.Main.object_select_window import ObjectSelectWindow

from GridCal.Gui.ProfilesInput.profile_dialogue import ProfileInputGUI
from GridCal.Gui.ProfilesInput.models_dialogue import ModelsInputGUI
from GridCal.Gui.SigmaAnalysis.sigma_analysis_dialogue import SigmaAnalysisGUI
from GridCal.Gui.SyncDialogue.sync_dialogue import SyncDialogueWindow
from GridCal.Gui.TowerBuilder.LineBuilderDialogue import TowerBuilderGUI
from GridCal.Gui.Session.session import SimulationSession, ResultsModel, GcThread
from GridCal.Gui.AboutDialogue.about_dialogue import AboutDialogueGuiGUI
from GridCal.Gui.GridGenerator.grid_generator_dialogue import GridGeneratorGUI
from GridCal.Gui.ContingencyPlanner.contingency_planner_dialogue import ContingencyPlannerGUI
from GridCal.Gui.pySlipQt.pySlipQt import PySlipQt
from GridCal.Gui.pySlipQt.blue_marble import BlueMarbleTiles
from GridCal.Gui.pySlipQt.cartodb import CartoDbTiles
import GridCal.Gui.Visualization.palettes as palettes

from GridCal.__version__ import __GridCal_VERSION__

try:
    from GridCal.Gui.ConsoleWidget import ConsoleWidget

    qt_console_available = True
except ModuleNotFoundError:
    print('No qtconsole available')
    qt_console_available = False

# try:
#     from PySide6.QtWebEngineWidgets import QWebEngineView as QWebView, QWebEnginePage as QWebPage
#     qt_web_engine_available = True
# except ModuleNotFoundError:

qt_web_engine_available = False

from matplotlib import pyplot as plt

__author__ = 'Santiago Peñate Vera'

"""
This class is the handler of the main gui of GridCal.
"""


def terminate_thread(thread):
    """Terminates a python thread from another thread.

    :param thread: a threading.Thread instance
    """
    if not thread.is_alive():
        return False

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

    return True


def traverse_objects(name, obj, lst: list, i=0):
    lst.append((name, sys.getsizeof(obj)))
    if i < 10:
        if hasattr(obj, '__dict__'):
            for name2, obj2 in obj.__dict__.items():
                if isinstance(obj2, np.ndarray):
                    lst.append((name + "/" + name2, sys.getsizeof(obj2)))
                else:
                    if isinstance(obj2, list):
                        # list or
                        for k, obj3 in enumerate(obj2):
                            traverse_objects(name=name + "/" + name2 + '[' + str(k) + ']',
                                             obj=obj3, lst=lst, i=i + 1)
                    elif isinstance(obj2, dict):
                        # list or
                        for name3, obj3 in obj2.items():
                            traverse_objects(name=name + "/" + name2 + '[' + name3 + ']',
                                             obj=obj3, lst=lst, i=i + 1)
                    else:
                        # normal obj
                        if obj2 != obj:
                            traverse_objects(name=name + "/" + name2, obj=obj2, lst=lst, i=i + 1)


########################################################################################################################
# Main Window
########################################################################################################################

class MainGUI(QMainWindow):

    def __init__(self, parent=None):
        """

        @param parent:
        """

        # create main window
        QMainWindow.__init__(self, parent)
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('GridCal ' + __GridCal_VERSION__)
        self.setAcceptDrops(True)

        # configure matplotlib for pandas time series
        register_matplotlib_converters()

        # Declare circuit
        self.circuit = core.MultiCircuit()

        self.calculation_inputs_to_display = None

        self.project_directory = os.path.expanduser("~")

        # solvers dictionary
        self.solvers_dict = OrderedDict()

        mdl = gf.get_list_model(self.circuit.profile_magnitudes.keys())
        self.ui.profile_device_type_comboBox.setModel(mdl)
        self.profile_device_type_changed()

        # reactive power controls
        self.q_control_modes_dict = OrderedDict()
        self.q_control_modes_dict['No control'] = bs.ReactivePowerControlMode.NoControl
        self.q_control_modes_dict['Direct'] = bs.ReactivePowerControlMode.Direct
        lst = list(self.q_control_modes_dict.keys())
        self.ui.reactive_power_control_mode_comboBox.setModel(gf.get_list_model(lst))

        # taps controls (transformer voltage regulator)
        self.taps_control_modes_dict = OrderedDict()
        self.taps_control_modes_dict['No control'] = bs.TapsControlMode.NoControl
        self.taps_control_modes_dict['Direct'] = bs.TapsControlMode.Direct
        lst = list(self.taps_control_modes_dict.keys())
        self.ui.taps_control_mode_comboBox.setModel(gf.get_list_model(lst))

        # transfer modes
        self.transfer_modes_dict = OrderedDict()
        self.transfer_modes_dict['Area generation'] = sim.AvailableTransferMode.Generation
        self.transfer_modes_dict['Area installed power'] = sim.AvailableTransferMode.InstalledPower
        self.transfer_modes_dict['Area load'] = sim.AvailableTransferMode.Load
        self.transfer_modes_dict['Area nodes'] = sim.AvailableTransferMode.GenerationAndLoad
        lst = list(self.transfer_modes_dict.keys())
        self.ui.transferMethodComboBox.setModel(gf.get_list_model(lst))
        self.ui.transferMethodComboBox.setCurrentIndex(1)

        self.accepted_extensions = ['.gridcal', '.xlsx', '.xls', '.sqlite', '.gch5',
                                    '.dgs', '.m', '.raw', '.RAW', '.json',
                                    '.ejson2', '.ejson3',
                                    '.xml', '.rawx', '.zip', '.dpx', '.epc']

        # ptdf grouping modes
        self.ptdf_group_modes = OrderedDict()

        # Automatic layout modes
        self.layout_algorithms_dict = dict()
        self.layout_algorithms_dict['circular_layout'] = nx.circular_layout
        self.layout_algorithms_dict['random_layout'] = nx.random_layout
        self.layout_algorithms_dict['shell_layout'] = nx.shell_layout
        self.layout_algorithms_dict['spring_layout'] = nx.spring_layout
        self.layout_algorithms_dict['spectral_layout'] = nx.spectral_layout
        self.layout_algorithms_dict['fruchterman_reingold_layout'] = nx.fruchterman_reingold_layout
        self.layout_algorithms_dict['kamada_kawai'] = nx.kamada_kawai_layout
        self.layout_algorithms_dict['graphviz_neato'] = nx.nx_agraph.graphviz_layout
        self.layout_algorithms_dict['graphviz_dot'] = nx.nx_agraph.graphviz_layout
        mdl = gf.get_list_model(list(self.layout_algorithms_dict.keys()))
        self.ui.automatic_layout_comboBox.setModel(mdl)
        self.ui.automatic_layout_comboBox.setCurrentIndex(6)

        # list of stochastic power flow methods
        self.stochastic_pf_methods_dict = OrderedDict()
        self.stochastic_pf_methods_dict[
            sim.StochasticPowerFlowType.LatinHypercube.value] = sim.StochasticPowerFlowType.LatinHypercube
        self.stochastic_pf_methods_dict[
            sim.StochasticPowerFlowType.MonteCarlo.value] = sim.StochasticPowerFlowType.MonteCarlo
        mdl = gf.get_list_model(list(self.stochastic_pf_methods_dict.keys()))
        self.ui.stochastic_pf_method_comboBox.setModel(mdl)

        # reactive power controls
        self.contingency_engines_dict = OrderedDict()
        self.contingency_engines_dict[bs.ContingencyEngine.PowerFlow.value] = bs.ContingencyEngine.PowerFlow
        self.contingency_engines_dict[bs.ContingencyEngine.PTDF.value] = bs.ContingencyEngine.PTDF
        self.contingency_engines_dict[bs.ContingencyEngine.HELM.value] = bs.ContingencyEngine.HELM
        self.ui.contingencyEngineComboBox.setModel(gf.get_list_model(list(self.contingency_engines_dict.keys())))

        # list of styles
        plt_styles = plt.style.available
        self.ui.plt_style_comboBox.setModel(gf.get_list_model(plt_styles))

        if 'fivethirtyeight' in plt_styles:
            self.ui.plt_style_comboBox.setCurrentText('fivethirtyeight')

        # dark mode detection
        is_dark = darkdetect.theme() == "Dark"
        self.ui.dark_mode_checkBox.setChecked(is_dark)

        # branch types for reduction
        mdl = gf.get_list_model(dev.BranchType.list(), checks=True)
        self.ui.removeByTypeListView.setModel(mdl)

        # opf solvers dictionary
        self.lp_solvers_dict = OrderedDict()
        self.lp_solvers_dict[bs.SolverType.DC_OPF.value] = bs.SolverType.DC_OPF
        if NEWTON_PA_AVAILABLE:
            self.lp_solvers_dict[bs.SolverType.AC_OPF.value] = bs.SolverType.AC_OPF
        self.lp_solvers_dict[bs.SolverType.Simple_OPF.value] = bs.SolverType.Simple_OPF
        self.ui.lpf_solver_comboBox.setModel(gf.get_list_model(list(self.lp_solvers_dict.keys())))

        self.opf_time_groups = OrderedDict()
        self.opf_time_groups[bs.TimeGrouping.NoGrouping.value] = bs.TimeGrouping.NoGrouping
        self.opf_time_groups[bs.TimeGrouping.Monthly.value] = bs.TimeGrouping.Monthly
        self.opf_time_groups[bs.TimeGrouping.Weekly.value] = bs.TimeGrouping.Weekly
        self.opf_time_groups[bs.TimeGrouping.Daily.value] = bs.TimeGrouping.Daily
        self.opf_time_groups[bs.TimeGrouping.Hourly.value] = bs.TimeGrouping.Hourly
        self.ui.opf_time_grouping_comboBox.setModel(gf.get_list_model(list(self.opf_time_groups.keys())))

        self.opf_zonal_groups = OrderedDict()
        self.opf_zonal_groups[bs.ZonalGrouping.NoGrouping.value] = bs.ZonalGrouping.NoGrouping
        # self.opf_zonal_groups[bs.ZonalGrouping.Area.value] = bs.ZonalGrouping.Area
        self.opf_zonal_groups[bs.ZonalGrouping.All.value] = bs.ZonalGrouping.All
        self.ui.opfZonalGroupByComboBox.setModel(gf.get_list_model(list(self.opf_zonal_groups.keys())))

        self.mip_solvers_dict = OrderedDict()
        self.mip_solvers_dict[bs.MIPSolvers.CBC.value] = bs.MIPSolvers.CBC
        self.mip_solvers_dict[bs.MIPSolvers.HIGHS.value] = bs.MIPSolvers.HIGHS
        self.mip_solvers_dict[bs.MIPSolvers.GLOP.value] = bs.MIPSolvers.GLOP
        self.mip_solvers_dict[bs.MIPSolvers.SCIP.value] = bs.MIPSolvers.SCIP
        self.mip_solvers_dict[bs.MIPSolvers.CPLEX.value] = bs.MIPSolvers.CPLEX
        self.mip_solvers_dict[bs.MIPSolvers.GUROBI.value] = bs.MIPSolvers.GUROBI
        self.mip_solvers_dict[bs.MIPSolvers.XPRESS.value] = bs.MIPSolvers.XPRESS

        # the MIP combobox models assigning is done in modify_ui_options_according_to_the_engine

        # voltage collapse mode (full, nose)
        self.ui.vc_stop_at_comboBox.setModel(gf.get_list_model([sim.CpfStopAt.Nose.value,
                                                                sim.CpfStopAt.ExtraOverloads.value]))
        self.ui.vc_stop_at_comboBox.setCurrentIndex(0)

        # available engines
        engine_lst = [bs.EngineType.GridCal]
        if NEWTON_PA_AVAILABLE:
            engine_lst.append(bs.EngineType.NewtonPA)
        if BENTAYGA_AVAILABLE:
            engine_lst.append(bs.EngineType.Bentayga)
        if PGM_AVAILABLE:
            engine_lst.append(bs.EngineType.PGM)

        self.ui.engineComboBox.setModel(gf.get_list_model([x.value for x in engine_lst]))
        self.ui.engineComboBox.setCurrentIndex(0)
        self.engine_dict = {x.value: x for x in engine_lst}

        # array modes
        self.ui.arrayModeComboBox.addItem('real')
        self.ui.arrayModeComboBox.addItem('imag')
        self.ui.arrayModeComboBox.addItem('abs')
        self.ui.arrayModeComboBox.addItem('complex')

        # list of pointers to the GIS windows
        self.gis_dialogues = list()
        self.files_to_delete_at_exit = list()
        self.bus_viewer_windows = list()

        ################################################################################################################
        # Declare the schematic editor
        ################################################################################################################

        # create diagram editor object
        self.grid_editor = GridEditor(self.circuit)

        self.ui.dataStructuresTreeView.setModel(gf.get_tree_model(self.circuit.get_objects_with_profiles_str_dict()))
        self.expand_object_tree_nodes()

        self.add_default_catalogue()

        self.ui.simulationDataStructuresListView.setModel(gf.get_list_model(core.NumericalCircuit.available_structures))

        self.schematic_list_steps = list()

        # add the widgets
        self.ui.schematic_layout.addWidget(self.grid_editor)
        # self.grid_editor.setStretchFactor(1, 10)

        # 1:4
        self.ui.dataStructuresSplitter.setStretchFactor(0, 2)
        self.ui.dataStructuresSplitter.setStretchFactor(1, 4)

        self.ui.simulationDataSplitter.setStretchFactor(1, 15)

        self.ui.results_splitter.setStretchFactor(0, 2)
        self.ui.results_splitter.setStretchFactor(1, 4)

        self.lock_ui = False
        self.ui.progress_frame.setVisible(self.lock_ui)

        ################################################################################################################
        # Declare the map
        self.tile_sources = {
            'Blue Marble': BlueMarbleTiles(tiles_dir=os.path.join(get_create_gridcal_folder(), 'tiles', 'blue_marble')),
            'Carto positron': CartoDbTiles(
                tiles_dir=os.path.join(get_create_gridcal_folder(), 'tiles', 'carto_db_positron'),
                tile_servers=['http://basemaps.cartocdn.com/light_all/']),
            'Carto dark matter': CartoDbTiles(
                tiles_dir=os.path.join(get_create_gridcal_folder(), 'tiles', 'carto_db_dark_matter'),
                tile_servers=["http://basemaps.cartocdn.com/dark_all/"])
        }

        palettes_list = [palettes.Colormaps.GridCal,
                         palettes.Colormaps.Green2Red,
                         palettes.Colormaps.Heatmap,
                         palettes.Colormaps.TSO]
        self.cmap_dict = {e.value: e for e in palettes_list}
        self.ui.palette_comboBox.setModel(gf.get_list_model([e.value for e in palettes_list]))

        self.ui.tile_provider_comboBox.setModel(gf.get_list_model(list(self.tile_sources.keys())))
        self.ui.tile_provider_comboBox.setCurrentIndex(0)

        # These get initialized by create_map()
        self.tile_source = None
        self.map_widget: Union[PySlipQt, None] = None
        self.polyline_layer_id: Union[int, None] = None

        self.create_map()

        self.ui.map_time_horizontalSlider.setMinimum(0)
        self.ui.map_time_horizontalSlider.setMaximum(0)
        self.ui.map_time_horizontalSlider.setSliderPosition(0)
        ################################################################################################################

        # set initial view position
        self.map_widget.GotoLevelAndPosition(5, -15.41, 40.11)
        self.map_list_steps = list()

        ################################################################################################################
        # simulations session ------------------------------------------------------------------------------------------
        self.session: SimulationSession = SimulationSession(name='GUI session')

        ################################################################################################################
        # threads ------------------------------------------------------------------------------------------------------
        self.painter = None
        self.open_file_thread_object = None
        self.save_file_thread_object = None
        self.last_file_driver = None
        self.delete_and_reduce_driver = None
        self.export_all_thread_object = None
        self.topology_reduction = None
        self.find_node_groups_driver: Union[sim.NodeGroupsDriver, None] = None
        self.file_sync_thread = syncdrv.FileSyncThread(self.circuit, None, None)
        self.stuff_running_now = list()

        # window pointers ----------------------------------------------------------------------------------------------
        self.file_sync_window: Union[SyncDialogueWindow, None] = None
        self.sigma_dialogue: Union[SigmaAnalysisGUI, None] = None
        self.grid_generator_dialogue: Union[GridGeneratorGUI, None] = None
        self.contingency_planner_dialogue: Union[ContingencyPlannerGUI, None] = None
        self.analysis_dialogue: Union[GridAnalysisGUI, None] = None
        self.profile_input_dialogue: Union[ProfileInputGUI, None] = None
        self.models_input_dialogue: Union[ModelsInputGUI, None] = None
        self.object_select_window: Union[ObjectSelectWindow, None] = None
        self.coordinates_window: Union[CoordinatesInputGUI, None] = None
        self.about_msg_window: Union[AboutDialogueGuiGUI, None] = None
        self.tower_builder_window: Union[TowerBuilderGUI, None] = None

        self.file_name = ''

        # current results model ----------------------------------------------------------------------------------------
        self.results_mdl: sim.ResultsTable = sim.ResultsTable(data=np.zeros((0, 0)),
                                                              columns=np.zeros(0),
                                                              index=np.zeros(0))

        # list of all the objects of the selected type under the Objects tab
        self.type_objects_list = list()

        self.buses_for_storage: Union[List[dev.Bus], None] = None

        # dictionaries for available results
        self.available_results_dict: Union[Dict[str, List[sim.ResultTypes]], None] = None
        self.available_results_steps_dict = None

        ################################################################################################################
        # Console
        ################################################################################################################

        self.console: Union[ConsoleWidget, None] = None
        try:
            self.create_console()
        except TypeError:
            error_msg('The console has failed because the QtConsole guys have a bug in their package :(')

        ################################################################################################################
        # Connections
        ################################################################################################################
        self.ui.actionNew_project.triggered.connect(self.new_project)

        self.ui.actionOpen_file.triggered.connect(self.open_file)

        self.ui.actionAdd_circuit.triggered.connect(self.add_circuit)

        self.ui.actionSave.triggered.connect(self.save_file)

        self.ui.actionSave_as.triggered.connect(self.save_file_as)

        self.ui.actionPower_flow.triggered.connect(self.run_power_flow)

        self.ui.actionShort_Circuit.triggered.connect(self.run_short_circuit)

        self.ui.actionVoltage_stability.triggered.connect(self.run_continuation_power_flow)

        self.ui.actionPower_Flow_Time_series.triggered.connect(self.run_power_flow_time_series)

        self.ui.actionPower_flow_Stochastic.triggered.connect(self.run_stochastic)

        self.ui.actionBlackout_cascade.triggered.connect(self.view_cascade_menu)

        self.ui.actionOPF.triggered.connect(self.run_opf)

        self.ui.actionOPF_time_series.triggered.connect(self.run_opf_time_series)

        self.ui.actionOptimal_Net_Transfer_Capacity.triggered.connect(self.run_opf_ntc)

        self.ui.actionOptimal_Net_Transfer_Capacity_Time_Series.triggered.connect(self.run_opf_ntc_ts)

        self.ui.actionOptimal_NTC_time_series_clustering.triggered.connect(self.run_opf_ntc_ts)

        self.ui.actionAbout.triggered.connect(self.about_box)

        self.ui.actionExport.triggered.connect(self.export_diagram)

        self.ui.actionAuto_rate_branches.triggered.connect(self.auto_rate_branches)

        self.ui.actionDetect_transformers.triggered.connect(self.detect_transformers)

        self.ui.actionExport_all_the_device_s_profiles.triggered.connect(self.export_object_profiles)

        self.ui.actionGrid_Reduction.triggered.connect(self.reduce_grid)

        self.ui.actionInputs_analysis.triggered.connect(self.run_inputs_analysis)

        self.ui.actionStorage_location_suggestion.triggered.connect(self.storage_location)

        self.ui.actionLaunch_data_analysis_tool.triggered.connect(self.display_grid_analysis)

        self.ui.actionOnline_documentation.triggered.connect(self.show_online_docs)

        self.ui.actionExport_all_results.triggered.connect(self.export_all)

        self.ui.actionDelete_selected.triggered.connect(self.delete_selected_from_the_schematic)

        self.ui.actionLinearAnalysis.triggered.connect(self.run_linear_analysis)

        self.ui.actionContingency_analysis.triggered.connect(self.run_contingency_analysis)

        self.ui.actionOTDF_time_series.triggered.connect(self.run_contingency_analysis_ts)

        self.ui.actionATC.triggered.connect(self.run_available_transfer_capacity)

        self.ui.actionATC_Time_Series.triggered.connect(self.run_available_transfer_capacity_ts)

        self.ui.actionATC_clustering.triggered.connect(self.run_available_transfer_capacity_clustering)

        self.ui.actionReset_console.triggered.connect(self.create_console)

        self.ui.actionTry_to_fix_buses_location.triggered.connect(self.try_to_fix_buses_location)

        self.ui.actionPTDF_time_series.triggered.connect(self.run_linear_analysis_ts)

        self.ui.actionSet_OPF_generation_to_profiles.triggered.connect(self.copy_opf_to_profiles)

        self.ui.actionClustering.triggered.connect(self.run_clustering)

        self.ui.actionDrawSchematic.triggered.connect(self.draw_schematic)

        self.ui.actionSet_schematic_positions_from_GPS_coordinates.triggered.connect(self.set_xy_from_lat_lon)

        self.ui.actionSigma_analysis.triggered.connect(self.run_sigma_analysis)

        self.ui.actionAdd_default_catalogue.triggered.connect(self.add_default_catalogue)

        self.ui.actionClear_stuff_running_right_now.triggered.connect(self.clear_stuff_running)

        self.ui.actionFind_node_groups.triggered.connect(self.run_find_node_groups)

        self.ui.actiongrid_Generator.triggered.connect(self.grid_generator)

        self.ui.actionImport_bus_coordinates.triggered.connect(self.import_bus_coordinates)

        self.ui.actionImport_contingencies.triggered.connect(self.import_contingencies)

        self.ui.actionSetSelectedBusCountry.triggered.connect(lambda: self.set_selected_bus_property('country'))
        self.ui.actionSetSelectedBusArea.triggered.connect(lambda: self.set_selected_bus_property('area'))
        self.ui.actionSetSelectedBusZone.triggered.connect(lambda: self.set_selected_bus_property('zone'))

        self.ui.actionFuse_devices.triggered.connect(self.fuse_devices)

        self.ui.actionDelete_inconsistencies.triggered.connect(self.delete_inconsistencies)

        self.ui.actionFix_generators_active_based_on_the_power.triggered.connect(
            self.fix_generators_active_based_on_the_power)

        self.ui.actionre_index_time.triggered.connect(self.re_index_time)

        self.ui.actionFix_loads_active_based_on_the_power.triggered.connect(self.fix_loads_active_based_on_the_power)

        self.ui.actionInitialize_contingencies.triggered.connect(self.initialize_contingencies)

        self.ui.actionExport_contingencies.triggered.connect(self.export_contingencies)

        self.ui.actionAdd_selected_to_contingency.triggered.connect(self.add_selected_to_contingency)

        self.ui.actionAdd_selected_as_new_investment.triggered.connect(self.add_selected_to_investment)

        self.ui.actionZoom_in.triggered.connect(self.zoom_in)

        self.ui.actionZoom_out.triggered.connect(self.zoom_out)

        # Buttons

        self.ui.cancelButton.clicked.connect(self.set_cancel_state)

        self.ui.new_profiles_structure_pushButton.clicked.connect(self.new_profiles_structure)

        self.ui.delete_profiles_structure_pushButton.clicked.connect(self.delete_profiles_structure)

        # self.ui.set_profile_state_button.clicked.connect(self.set_profiles_state_to_grid)

        self.ui.edit_profiles_pushButton.clicked.connect(self.import_profiles)

        self.ui.edit_profiles_from_models_pushButton.clicked.connect(self.import_profiles_from_models)

        self.ui.saveResultsButton.clicked.connect(self.save_results_df)

        self.ui.set_profile_state_button.clicked.connect(self.set_state)

        self.ui.setValueToColumnButton.clicked.connect(self.set_value_to_column)

        self.ui.run_cascade_pushButton.clicked.connect(self.run_cascade)

        self.ui.clear_cascade_pushButton.clicked.connect(self.clear_cascade)

        self.ui.run_cascade_step_pushButton.clicked.connect(self.run_cascade_step)

        self.ui.exportSimulationDataButton.clicked.connect(self.export_simulation_data)

        self.ui.filter_pushButton.clicked.connect(self.smart_search)

        self.ui.profile_add_pushButton.clicked.connect(lambda: self.modify_profiles('+'))

        self.ui.profile_subtract_pushButton.clicked.connect(lambda: self.modify_profiles('-'))

        self.ui.profile_multiply_pushButton.clicked.connect(lambda: self.modify_profiles('*'))

        self.ui.profile_divide_pushButton.clicked.connect(lambda: self.modify_profiles('/'))

        self.ui.set_profile_value_pushButton.clicked.connect(lambda: self.modify_profiles('set'))

        self.ui.set_linear_combination_profile_pushButton.clicked.connect(self.set_profile_as_linear_combination)

        self.ui.plot_time_series_pushButton.clicked.connect(self.plot_profiles)

        self.ui.catalogue_edit_pushButton.clicked.connect(self.edit_from_catalogue)

        self.ui.compute_simulation_data_pushButton.clicked.connect(self.update_islands_to_display)

        self.ui.copy_profile_pushButton.clicked.connect(self.copy_profiles)

        self.ui.paste_profiles_pushButton.clicked.connect(self.paste_profiles)

        self.ui.colour_results_pushButton.clicked.connect(self.colour_schematic)

        self.ui.view_previous_simulation_step_pushButton.clicked.connect(self.colour_previous_simulation_step)

        self.ui.view_next_simulation_step_pushButton.clicked.connect(self.colour_next_simulation_step)

        self.ui.view_previous_simulation_step_map_pushButton.clicked.connect(self.colour_previous_simulation_step_map)

        self.ui.view_next_simulation_step_map_pushButton.clicked.connect(self.colour_next_simulation_step_map)

        self.ui.copy_results_pushButton.clicked.connect(self.copy_results_data)

        self.ui.copyObjectsTableButton.clicked.connect(self.copy_objects_data)

        self.ui.copy_numpy_button.clicked.connect(self.copy_results_data_as_numpy)

        self.ui.undo_pushButton.clicked.connect(self.undo)

        self.ui.redo_pushButton.clicked.connect(self.redo)

        self.ui.delete_selected_objects_pushButton.clicked.connect(self.delete_selected_objects)

        self.ui.add_object_pushButton.clicked.connect(self.add_objects)

        self.ui.delete_and_reduce_pushButton.clicked.connect(self.delete_and_reduce_selected_objects)

        self.ui.highlight_selection_buses_pushButton.clicked.connect(self.highlight_selection_buses)

        self.ui.clear_highlight_pushButton.clicked.connect(self.clear_big_bus_markers)

        self.ui.highlight_by_property_pushButton.clicked.connect(self.highlight_based_on_property)

        self.ui.plot_data_pushButton.clicked.connect(self.plot_results)

        self.ui.busViewerButton.clicked.connect(self.bus_viewer)

        self.ui.search_results_Button.clicked.connect(self.search_in_results)

        self.ui.deleteDriverButton.clicked.connect(self.delete_results_driver)

        self.ui.loadResultFromDiskButton.clicked.connect(self.load_results_driver)

        self.ui.plotArraysButton.clicked.connect(self.plot_simulation_objects_data)

        self.ui.copyArraysButton.clicked.connect(self.copy_simulation_objects_data)

        self.ui.copyArraysToNumpyButton.clicked.connect(self.copy_simulation_objects_data_to_numpy)

        self.ui.structure_analysis_pushButton.clicked.connect(self.structure_analysis_plot)

        self.ui.draw_map_button.clicked.connect(self.colour_map)

        # node size
        self.ui.actionBigger_nodes.triggered.connect(self.bigger_nodes)

        self.ui.actionSmaller_nodes.triggered.connect(self.smaller_nodes)

        self.ui.actionCenter_view.triggered.connect(self.center_nodes)

        self.ui.actionAutoatic_layout.triggered.connect(self.auto_layout)

        # list clicks

        self.ui.dataStructuresTreeView.clicked.connect(self.view_objects_data)

        self.ui.simulationDataStructuresListView.clicked.connect(self.view_simulation_objects_data)

        # tree-view clicks
        self.ui.results_treeView.clicked.connect(self.results_tree_view_click)

        # Table clicks
        self.ui.cascade_tableView.clicked.connect(self.cascade_table_click)

        # combobox
        self.ui.profile_device_type_comboBox.currentTextChanged.connect(self.profile_device_type_changed)

        self.ui.device_type_magnitude_comboBox.currentTextChanged.connect(self.display_profiles)

        self.ui.plt_style_comboBox.currentTextChanged.connect(self.plot_style_change)

        self.ui.available_results_to_color_comboBox.currentTextChanged.connect(self.update_available_steps_to_color)
        self.ui.available_results_to_color_map_comboBox.currentTextChanged.connect(
            self.update_available_steps_to_color_map)

        self.ui.engineComboBox.currentTextChanged.connect(self.modify_ui_options_according_to_the_engine)

        self.ui.tile_provider_comboBox.currentTextChanged.connect(self.create_map)

        # sliders
        self.ui.profile_start_slider.valueChanged.connect(self.profile_sliders_changed)
        self.ui.profile_end_slider.valueChanged.connect(self.profile_sliders_changed)
        self.ui.simulation_results_step_slider.valueChanged.connect(self.schematic_time_slider_change)
        self.ui.map_time_horizontalSlider.valueChanged.connect(self.map_time_slider_change)

        # doubleSpinBox
        self.ui.fbase_doubleSpinBox.valueChanged.connect(self.change_circuit_base)
        self.ui.sbase_doubleSpinBox.valueChanged.connect(self.change_circuit_base)

        self.ui.explosion_factor_doubleSpinBox.valueChanged.connect(self.explosion_factor_change)

        # line edit enter
        self.ui.smart_search_lineEdit.returnPressed.connect(self.smart_search)
        self.ui.sear_results_lineEdit.returnPressed.connect(self.search_in_results)

        # check boxes
        self.ui.draw_schematic_checkBox.clicked.connect(self.set_grid_editor_state)
        self.ui.dark_mode_checkBox.clicked.connect(self.change_theme_mode)

        # Radio Button
        self.ui.proportionalRedispatchRadioButton.clicked.connect(self.default_options_opf_ntc_proportional)
        self.ui.optimalRedispatchRadioButton.clicked.connect(self.default_options_opf_ntc_optimal)

        ################################################################################################################
        # Other actions
        ################################################################################################################

        self.ui.grid_colouring_frame.setVisible(True)

        self.ui.actionSync.setVisible(False)

        self.modify_ui_options_according_to_the_engine()

        # this is the contingency planner tab, invisible until done
        self.ui.tabWidget_3.setTabVisible(4, True)

        self.view_cascade_menu()

        self.clear_results()

        self.load_gui_config()

    def LOCK(self, val: bool = True) -> None:
        """
        Lock the interface to prevent new simulation launches
        :param val:
        :return:
        """
        self.lock_ui = val
        self.ui.progress_frame.setVisible(self.lock_ui)
        QtGui.QGuiApplication.processEvents()

    def UNLOCK(self) -> None:
        """
        Unlock the interface
        """
        if not self.any_thread_running():
            self.LOCK(False)

    def modify_ui_options_according_to_the_engine(self) -> None:
        """
        Change the UI depending on the engine options
        :return:
        """
        eng = self.get_preferred_engine()

        if eng == bs.EngineType.NewtonPA:
            self.ui.opfUnitCommitmentCheckBox.setVisible(True)
            self.ui.maxVoltageModuleStepSpinBox.setVisible(True)
            self.ui.maxVoltageAngleStepSpinBox.setVisible(True)
            self.ui.maxVoltageModuleStepLabel.setVisible(True)
            self.ui.maxVoltageAngleStepLabel.setVisible(True)

            # add the AC_OPF option
            self.lp_solvers_dict = OrderedDict()
            self.lp_solvers_dict[bs.SolverType.DC_OPF.value] = bs.SolverType.DC_OPF
            self.lp_solvers_dict[bs.SolverType.AC_OPF.value] = bs.SolverType.AC_OPF
            self.lp_solvers_dict[bs.SolverType.Simple_OPF.value] = bs.SolverType.Simple_OPF
            self.ui.lpf_solver_comboBox.setModel(gf.get_list_model(list(self.lp_solvers_dict.keys())))

            # Power Flow Methods
            self.solvers_dict[bs.SolverType.NR.value] = bs.SolverType.NR
            self.solvers_dict[bs.SolverType.NRI.value] = bs.SolverType.NRI
            self.solvers_dict[bs.SolverType.IWAMOTO.value] = bs.SolverType.IWAMOTO
            self.solvers_dict[bs.SolverType.LM.value] = bs.SolverType.LM
            self.solvers_dict[bs.SolverType.FASTDECOUPLED.value] = bs.SolverType.FASTDECOUPLED
            self.solvers_dict[bs.SolverType.HELM.value] = bs.SolverType.HELM
            self.solvers_dict[bs.SolverType.GAUSS.value] = bs.SolverType.GAUSS
            self.solvers_dict[bs.SolverType.LACPF.value] = bs.SolverType.LACPF
            self.solvers_dict[bs.SolverType.DC.value] = bs.SolverType.DC

            self.ui.solver_comboBox.setModel(gf.get_list_model(list(self.solvers_dict.keys())))
            self.ui.solver_comboBox.setCurrentIndex(0)

            mip_solvers = get_newton_mip_solvers_list()
            self.ui.mip_solver_comboBox.setModel(gf.get_list_model(mip_solvers))

        elif eng == bs.EngineType.GridCal:
            self.ui.opfUnitCommitmentCheckBox.setVisible(True)
            self.ui.maxVoltageModuleStepSpinBox.setVisible(False)
            self.ui.maxVoltageAngleStepSpinBox.setVisible(False)
            self.ui.maxVoltageModuleStepLabel.setVisible(False)
            self.ui.maxVoltageAngleStepLabel.setVisible(False)

            # no AC opf option
            self.lp_solvers_dict = OrderedDict()
            self.lp_solvers_dict[bs.SolverType.DC_OPF.value] = bs.SolverType.DC_OPF
            self.lp_solvers_dict[bs.SolverType.Simple_OPF.value] = bs.SolverType.Simple_OPF
            self.ui.lpf_solver_comboBox.setModel(gf.get_list_model(list(self.lp_solvers_dict.keys())))

            # Power Flow Methods
            self.solvers_dict = OrderedDict()
            self.solvers_dict[bs.SolverType.NR.value] = bs.SolverType.NR
            self.solvers_dict[bs.SolverType.NRI.value] = bs.SolverType.NRI
            self.solvers_dict[bs.SolverType.IWAMOTO.value] = bs.SolverType.IWAMOTO
            self.solvers_dict[bs.SolverType.LM.value] = bs.SolverType.LM
            self.solvers_dict[bs.SolverType.FASTDECOUPLED.value] = bs.SolverType.FASTDECOUPLED
            self.solvers_dict[bs.SolverType.HELM.value] = bs.SolverType.HELM
            self.solvers_dict[bs.SolverType.GAUSS.value] = bs.SolverType.GAUSS
            self.solvers_dict[bs.SolverType.LACPF.value] = bs.SolverType.LACPF
            self.solvers_dict[bs.SolverType.DC.value] = bs.SolverType.DC

            self.ui.solver_comboBox.setModel(gf.get_list_model(list(self.solvers_dict.keys())))
            self.ui.solver_comboBox.setCurrentIndex(0)

            # MIP solvers
            mip_solvers = get_or_tools_available_solvers()
            self.ui.mip_solver_comboBox.setModel(gf.get_list_model(mip_solvers))

        elif eng == bs.EngineType.Bentayga:
            self.ui.opfUnitCommitmentCheckBox.setVisible(False)
            self.ui.maxVoltageModuleStepSpinBox.setVisible(False)
            self.ui.maxVoltageAngleStepSpinBox.setVisible(False)
            self.ui.maxVoltageModuleStepLabel.setVisible(False)
            self.ui.maxVoltageAngleStepLabel.setVisible(False)

            # no AC opf option
            self.lp_solvers_dict = OrderedDict()
            self.lp_solvers_dict[bs.SolverType.DC_OPF.value] = bs.SolverType.DC_OPF
            self.lp_solvers_dict[bs.SolverType.Simple_OPF.value] = bs.SolverType.Simple_OPF
            self.ui.lpf_solver_comboBox.setModel(gf.get_list_model(list(self.lp_solvers_dict.keys())))

            # Power Flow Methods
            self.solvers_dict = OrderedDict()
            self.solvers_dict[bs.SolverType.NR.value] = bs.SolverType.NR
            self.solvers_dict[bs.SolverType.NRI.value] = bs.SolverType.NRI
            self.solvers_dict[bs.SolverType.IWAMOTO.value] = bs.SolverType.IWAMOTO
            self.solvers_dict[bs.SolverType.LM.value] = bs.SolverType.LM
            self.solvers_dict[bs.SolverType.FASTDECOUPLED.value] = bs.SolverType.FASTDECOUPLED
            self.solvers_dict[bs.SolverType.HELM.value] = bs.SolverType.HELM
            self.solvers_dict[bs.SolverType.GAUSS.value] = bs.SolverType.GAUSS
            self.solvers_dict[bs.SolverType.LACPF.value] = bs.SolverType.LACPF
            self.solvers_dict[bs.SolverType.DC.value] = bs.SolverType.DC

            self.ui.solver_comboBox.setModel(gf.get_list_model(list(self.solvers_dict.keys())))
            self.ui.solver_comboBox.setCurrentIndex(0)

        elif eng == bs.EngineType.PGM:
            self.ui.opfUnitCommitmentCheckBox.setVisible(False)
            self.ui.maxVoltageModuleStepSpinBox.setVisible(False)
            self.ui.maxVoltageAngleStepSpinBox.setVisible(False)
            self.ui.maxVoltageModuleStepLabel.setVisible(False)
            self.ui.maxVoltageAngleStepLabel.setVisible(False)

            # no AC opf option
            self.lp_solvers_dict = OrderedDict()
            self.lp_solvers_dict[bs.SolverType.DC_OPF.value] = bs.SolverType.DC_OPF
            self.lp_solvers_dict[bs.SolverType.Simple_OPF.value] = bs.SolverType.Simple_OPF
            self.ui.lpf_solver_comboBox.setModel(gf.get_list_model(list(self.lp_solvers_dict.keys())))

            # Power Flow Methods
            self.solvers_dict = OrderedDict()
            self.solvers_dict[bs.SolverType.NR.value] = bs.SolverType.NR
            self.solvers_dict[bs.SolverType.BFS.value] = bs.SolverType.BFS
            self.solvers_dict[bs.SolverType.BFS_linear.value] = bs.SolverType.BFS_linear
            self.solvers_dict[bs.SolverType.Constant_Impedance_linear.value] = bs.SolverType.Constant_Impedance_linear

            self.ui.solver_comboBox.setModel(gf.get_list_model(list(self.solvers_dict.keys())))
            self.ui.solver_comboBox.setCurrentIndex(0)

        else:
            raise Exception('Unsupported engine' + str(eng.value))

    @staticmethod
    def collect_memory():
        for i in (0, 1, 2):
            gc.collect(generation=i)

    def get_preferred_engine(self):
        """
        Get the currently selected engine
        :return:
        """
        val = self.ui.engineComboBox.currentText()
        return self.engine_dict[val]

    def get_simulation_threads(self):
        """
        Get all threads that has to do with simulation
        :return: list of simulation threads
        """

        all_threads = list(self.session.threads.values())

        return all_threads

    def get_simulations(self):
        """
        Get all threads that have to do with simulation
        :return: list of simulation threads
        """

        all_threads = list(self.session.drivers.values())

        # set the threads so that the diagram scene objects can plot them
        self.grid_editor.diagramScene.set_results_to_plot(all_threads)

        return all_threads

    def get_process_threads(self):
        """
        Get all threads that has to do with processing
        :return: list of process threads
        """
        all_threads = [self.open_file_thread_object,
                       self.save_file_thread_object,
                       self.painter,
                       self.delete_and_reduce_driver,
                       self.export_all_thread_object,
                       self.find_node_groups_driver,
                       self.file_sync_thread]
        return all_threads

    def get_all_threads(self) -> List[GcThread]:
        """
        Get all threads
        :return: list of all threads
        """
        all_threads = self.get_simulation_threads() + self.get_process_threads()
        return all_threads

    def stop_all_threads(self):
        """
        Stop all running threads
        """
        for thr in self.get_all_threads():
            if thr is not None:
                thr.quit()

        for thread in threading.enumerate():
            print(thread.name, end="")
            if "MainThread" not in thread.name:
                stat = terminate_thread(thread)
                if stat:
                    print(" killed")
                else:
                    print(" not killed")
            else:
                print(" Skipped")

        # second pass, kill main too
        for thread in threading.enumerate():
            print(thread.name, end="")
            stat = terminate_thread(thread)
            if stat:
                print(" killed")
            else:
                print(" not killed")

    def any_thread_running(self):
        """
        Checks if any thread is running
        :return: True/False
        """
        val = False

        # this list cannot be created only once, because the None will be copied
        # instead of being a pointer to the future value like it would in a typed language
        all_threads = self.get_all_threads()

        for thr in all_threads:
            if thr is not None:
                if thr.isRunning():
                    return True
        return val

    def set_grid_editor_state(self):
        """
        Enable/disable the grid editor
        """
        if self.ui.draw_schematic_checkBox.isChecked():
            self.grid_editor.setEnabled(True)
        else:
            self.grid_editor.setDisabled(True)

    def create_console(self):
        """
        Create console
        """
        if qt_console_available:
            if self.console is not None:
                clear_qt_layout(self.ui.pythonConsoleTab.layout())

            self.console = ConsoleWidget(customBanner="GridCal console.\n\n"
                                                      "type hlp() to see the available specific commands.\n\n"
                                                      "the following libraries are already loaded:\n"
                                                      "np: numpy\n"
                                                      "pd: pandas\n"
                                                      "plt: matplotlib\n"
                                                      "app: This instance of GridCal\n"
                                                      "circuit: The current grid\n\n")

            self.console.buffer_size = 10000

            # add the console widget to the user interface
            self.ui.pythonConsoleTab.layout().addWidget(self.console)

            # push some variables to the console
            self.console.push_vars({"hlp": self.print_console_help,
                                    "np": np,
                                    "pd": pd,
                                    "plt": plt,
                                    "clc": self.clc,
                                    'app': self,
                                    'circuit': self.circuit})

    def get_time_indices(self) -> np.ndarray:
        """
        Get an array of indices of the time steps selected within the start-end interval
        :return: np.array[int]
        """

        start = self.ui.profile_start_slider.value()
        end = self.ui.profile_end_slider.value()

        if start > end:
            self.ui.profile_end_slider.setValue(start)
            end = start

        return np.arange(start, end + 1)

    def create_map(self):
        """
        Create the map widget
        """

        # remove all widgets from the layout
        for i in reversed(range(self.ui.map_layout.count())):
            widget_to_remove = self.ui.map_layout.itemAt(i).widget()
            # remove it from the layout list
            self.ui.map_layout.removeWidget(widget_to_remove)
            # remove it from the gui
            widget_to_remove.setParent(None)

        # select the tile source
        self.tile_source = self.tile_sources[self.ui.tile_provider_comboBox.currentText()]

        # create the map widget
        self.map_widget = PySlipQt(self, tile_src=self.tile_source, start_level=5)

        # add lines layer
        self.polyline_layer_id = self.map_widget.AddPolylineLayer(data=[],
                                                                  map_rel=True,
                                                                  visible=True,
                                                                  delta=40,
                                                                  show_levels=list(range(20)),
                                                                  # levels at which to show the polylines
                                                                  name='<polyline_layer>')
        # add to the layout
        self.ui.map_layout.addWidget(self.map_widget)

        self.map_widget.setLayerSelectable(self.polyline_layer_id, True)

    def clear_stuff_running(self):
        """
        This clears the list of stuff running right now
        this list blocks new executions of the same threads.
        Cleaning is useful if a particular thread crashes and you want to retry.
        """
        self.stuff_running_now.clear()

    def change_theme_mode(self):
        """
        Change the GUI theme
        :return:
        """
        custom_colors = {"primary": "#00aa88ff",
                         "primary>list.selectionBackground": "#00aa88be"}

        if self.ui.dark_mode_checkBox.isChecked():
            qdarktheme.setup_theme(theme='dark',
                                   custom_colors=custom_colors)
            self.grid_editor.set_dark_mode()
            self.colour_schematic()
        else:
            qdarktheme.setup_theme(theme='light',
                                   custom_colors=custom_colors)
            self.grid_editor.set_light_mode()
            self.colour_schematic()

    def dragEnterEvent(self, event):
        """

        :param event:
        :return:
        """
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """

        :param event:
        :return:
        """
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Drop file on the GUI, the default behaviour is to load the file
        :param event: event containing all the information
        """
        if event.mimeData().hasUrls:
            events = event.mimeData().urls()
            if len(events) > 0:

                file_names = list()

                for event in events:
                    file_name = event.toLocalFile()
                    name, file_extension = os.path.splitext(file_name)
                    if file_extension.lower() in self.accepted_extensions:
                        file_names.append(file_name)
                    else:
                        error_msg('The file type ' + file_extension.lower() + ' is not accepted :(')

                if len(self.circuit.buses) > 0:
                    quit_msg = "Are you sure that you want to quit the current grid and open a new one?" \
                               "\n If the process is cancelled the grid will remain."
                    reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                                           QtWidgets.QMessageBox.Yes,
                                                           QtWidgets.QMessageBox.No)

                    if reply == QtWidgets.QMessageBox.Yes:
                        self.open_file_now(filenames=file_names)
                else:
                    # Just open the file
                    self.open_file_now(filenames=file_names)

    def add_simulation(self, val: sim.SimulationTypes):
        """
        Add a simulation to the simulations list
        :param val: simulation type
        """
        self.stuff_running_now.append(val)

    def remove_simulation(self, val: sim.SimulationTypes):
        """
        Remove a simulation from the simulations list
        :param val: simulation type
        """
        if val in self.stuff_running_now:
            self.stuff_running_now.remove(val)

    def view_cascade_menu(self):
        """
        show/hide the cascade simulation menu
        """
        self.ui.cascade_menu.setVisible(self.ui.actionBlackout_cascade.isChecked())
        self.ui.cascade_grid_splitter.setStretchFactor(1, 4)

    def about_box(self):
        """
        Display about box
        :return:
        """

        self.about_msg_window = AboutDialogueGuiGUI(self)
        self.about_msg_window.setVisible(True)

    @staticmethod
    def show_online_docs():
        """
        Open the online documentation in a web browser
        """
        webbrowser.open('https://gridcal.readthedocs.io/en/latest/', new=2)

    @staticmethod
    def print_console_help():
        """
        Print the console help in the console
        @return:
        """
        print('GridCal internal commands.\n')
        print('If a command is unavailable is because the study has not been executed yet.')

        print('\n\nclc():\tclear the console.')

        print('\n\nApp functions:')
        print('\tapp.new_project(): Clear all.')
        print('\tapp.open_file(): Prompt to load GridCal compatible file')
        print('\tapp.save_file(): Prompt to save GridCal file')
        print('\tapp.export_diagram(): Prompt to export the diagram in png.')
        print('\tapp.create_schematic_from_api(): Create the schematic from the circuit information.')
        print('\tapp.adjust_all_node_width(): Adjust the width of all the nodes according to their name.')
        print('\tapp.numerical_circuit: get compilation of the assets.')
        print('\tapp.islands: get compilation of the assets split into the topological islands.')

        print('\n\nCircuit functions:')
        print('\tapp.circuit.plot_graph(): Plot a graph in a Matplotlib window. Call plt.show() after.')

        print('\n\nPower flow results:')
        print('\tapp.session.power_flow.voltage:\t the nodal voltages in per unit')
        print('\tapp.session.power_flow.current:\t the branch currents in per unit')
        print('\tapp.session.power_flow.loading:\t the branch loading in %')
        print('\tapp.session.power_flow.losses:\t the branch losses in per unit')
        print('\tapp.session.power_flow.power:\t the nodal power Injections in per unit')
        print('\tapp.session.power_flow.Sf:\t the branch power Injections in per unit at the "from" side')
        print('\tapp.session.power_flow.St:\t the branch power Injections in per unit at the "to" side')

        print('\n\nShort circuit results:')
        print('\tapp.session.short_circuit.voltage:\t the nodal voltages in per unit')
        print('\tapp.session.short_circuit.current:\t the branch currents in per unit')
        print('\tapp.session.short_circuit.loading:\t the branch loading in %')
        print('\tapp.session.short_circuit.losses:\t the branch losses in per unit')
        print('\tapp.session.short_circuit.power:\t the nodal power Injections in per unit')
        print('\tapp.session.short_circuit.power_from:\t the branch power Injections in per unit at the "from" side')
        print('\tapp.session.short_circuit.power_to:\t the branch power Injections in per unit at the "to" side')
        print('\tapp.session.short_circuit.short_circuit_power:\t Short circuit power in MVA of the grid nodes')

        print('\n\nOptimal power flow results:')
        print('\tapp.session.optimal_power_flow.voltage:\t the nodal voltages angles in rad')
        print('\tapp.session.optimal_power_flow.load_shedding:\t the branch loading in %')
        print('\tapp.session.optimal_power_flow.losses:\t the branch losses in per unit')
        print('\tapp.session.optimal_power_flow.Sbus:\t the nodal power Injections in MW')
        print('\tapp.session.optimal_power_flow.Sf:\t the branch power Sf')

        print('\n\nTime series power flow results:')
        print('\tapp.session.power_flow_ts.time:\t Profiles time index (pandas DateTimeIndex object)')
        print('\tapp.session.power_flow_ts.load_profiles:\t Load profiles matrix (row: time, col: node)')
        print('\tapp.session.power_flow_ts.gen_profiles:\t Generation profiles matrix (row: time, col: node)')
        print('\tapp.session.power_flow_ts.voltages:\t nodal voltages results matrix (row: time, col: node)')
        print('\tapp.session.power_flow_ts.currents:\t Branches currents results matrix (row: time, col: branch)')
        print('\tapp.session.power_flow_ts.loadings:\t Branches loadings results matrix (row: time, col: branch)')
        print('\tapp.session.power_flow_ts.losses:\t Branches losses results matrix (row: time, col: branch)')

        print('\n\nVoltage stability power flow results:')
        print('\tapp.session.continuation_power_flow.voltage:\t Voltage values for every power multiplication factor.')
        print('\tapp.session.continuation_power_flow.lambda:\t Value of power multiplication factor applied')
        print('\tapp.session.continuation_power_flow.Sf:\t Power values for every power multiplication factor.')

        print('\n\nMonte Carlo power flow results:')
        print('\tapp.session.stochastic_power_flow.V_avg:\t nodal voltage average result.')
        print('\tapp.session.stochastic_power_flow.I_avg:\t branch current average result.')
        print('\tapp.session.stochastic_power_flow.Loading_avg:\t branch loading average result.')
        print('\tapp.session.stochastic_power_flow.Losses_avg:\t branch losses average result.')
        print('\tapp.session.stochastic_power_flow.V_std:\t nodal voltage standard deviation result.')
        print('\tapp.session.stochastic_power_flow.I_std:\t branch current standard deviation result.')
        print('\tapp.session.stochastic_power_flow.Loading_std:\t branch loading standard deviation result.')
        print('\tapp.session.stochastic_power_flow.Losses_std:\t branch losses standard deviation result.')
        print('\tapp.session.stochastic_power_flow.V_avg_series:\t nodal voltage average series.')
        print('\tapp.session.stochastic_power_flow.V_std_series:\t branch current standard deviation series.')
        print('\tapp.session.stochastic_power_flow.error_series:\t Monte Carlo error series (the convergence value).')
        print('The same for app.latin_hypercube_sampling')

    def clc(self):
        """
        Clear the console
        """
        self.console.clear()

    def clear_text_output(self):
        self.ui.outputTextEdit.setPlainText("")

    def console_msg(self, *msg_):
        """
        Print some message in the console.

        Arguments:

            **msg_** (str): Message

        """
        dte = dtelib.datetime.now().strftime("%b %d %Y %H:%M:%S")

        txt = self.ui.outputTextEdit.toPlainText()

        for e in msg_:
            if isinstance(e, list):
                txt += '\n' + dte + '->\n'
                for elm in e:
                    txt += str(elm) + "\n"
            else:
                txt += '\n' + dte + '->'
                txt += " " + str(e)

        self.ui.outputTextEdit.setPlainText(txt)

    def auto_layout(self):
        """
        Automatic layout of the nodes
        """

        # guilty assumption
        do_it = True

        # if the ask, checkbox is checked, then ask
        if self.ui.ask_before_appliying_layout_checkBox.isChecked():
            reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                   'Are you sure that you want to try an automatic layout?',
                                                   QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                do_it = True
                # build the graph always in this step
                self.circuit.build_graph()
            else:
                do_it = False

        if do_it:
            if self.circuit.graph is None:
                self.circuit.build_graph()

            sel = self.ui.automatic_layout_comboBox.currentText()
            pos_alg = self.layout_algorithms_dict[sel]

            # get the positions of a spring layout of the graph
            if sel == 'random_layout':
                pos = pos_alg(self.circuit.graph)
            elif sel == 'spring_layout':
                pos = pos_alg(self.circuit.graph, iterations=100, scale=10)
            elif sel == 'graphviz_neato':
                pos = pos_alg(self.circuit.graph, prog='neato')
            elif sel == 'graphviz_dot':
                pos = pos_alg(self.circuit.graph, prog='dot')
            else:
                pos = pos_alg(self.circuit.graph, scale=10)

            # assign the positions to the graphical objects of the nodes
            for i, bus in enumerate(self.circuit.buses):
                try:
                    x = pos[i][0] * 500
                    y = pos[i][1] * 500
                    bus.graphic_obj.setPos(QtCore.QPoint(x, y))
                except KeyError as ex:
                    warn('Node ' + str(i) + ' not in graph!!!! \n' + str(ex))
            # adjust the view
            self.center_nodes()

        else:
            pass  # asked and decided ot to change the layout

    def bigger_nodes(self):
        """
        Move the nodes more separated
        """
        if self.grid_editor is not None:
            self.grid_editor.expand_node_distances()

    def smaller_nodes(self):
        """
        Move the nodes closer
        """
        if self.grid_editor is not None:
            self.grid_editor.shrink_node_distances()

    def center_nodes(self):
        """
        Center the nodes in the screen
        """
        if self.grid_editor is not None:

            selected = self.get_selected_buses()

            if len(selected) == 0:
                buses = self.circuit.buses
            else:
                buses = [b for i, b in selected]

            self.grid_editor.align_schematic(buses=buses)

    def new_project_now(self):
        """
        New project right now without asking questions
        """
        # clear the circuit model
        self.circuit = core.MultiCircuit()

        # clear the file name
        self.file_name = ''

        self.grid_editor.clear()

        self.ui.dataStructuresTreeView.setModel(gf.get_tree_model(self.circuit.get_objects_with_profiles_str_dict(),
                                                                  top='Objects'))
        self.expand_object_tree_nodes()

        # clear the results
        self.ui.resultsTableView.setModel(None)

        # clear the comments
        self.ui.comments_textEdit.setText("")

        # clear the simulation objects
        for thread in self.get_all_threads():
            thread = None

        # close all the bus view windows
        for hndl in self.bus_viewer_windows:
            if hndl is not None:
                hndl.close()
        self.bus_viewer_windows.clear()

        if self.analysis_dialogue is not None:
            self.analysis_dialogue.close()

        self.clear_stuff_running()
        self.clear_results()
        self.add_default_catalogue()
        self.create_console()
        self.create_map()
        self.collect_memory()

    def new_project(self):
        """
        Create new grid
        :return:
        """
        if len(self.circuit.buses) > 0:
            quit_msg = "Are you sure that you want to quit the current grid and create a new one?"
            reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                                   QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                self.new_project_now()

    def open_file(self):
        """
        Open GridCal file
        @return:
        """
        if ('file_save' not in self.stuff_running_now) and ('file_open' not in self.stuff_running_now):
            if len(self.circuit.buses) > 0:
                quit_msg = "Are you sure that you want to quit the current grid and open a new one?" \
                           "\n If the process is cancelled the grid will remain."
                reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                                       QtWidgets.QMessageBox.Yes,
                                                       QtWidgets.QMessageBox.No)

                if reply == QtWidgets.QMessageBox.Yes:
                    self.new_project_now()
                    self.open_file_threaded()
                else:
                    pass
            else:
                # Just open the file
                self.open_file_threaded()

        else:
            warning_msg('There is a file being processed now.')

    def open_file_threaded(self, post_function=None):
        """
        Open file from a Qt thread to remain responsive
        """

        files_types = "Formats (*.gridcal *.gch5 *.xlsx *.xls *.sqlite *.dgs " \
                      "*.m *.raw *.RAW *.rawx *.json *.ejson2 *.ejson3 *.xml *.zip *.dpx *.epc *.nc *.hdf5)"
        # files_types = ''
        # call dialog to select the file

        # options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog

        filenames, type_selected = QtWidgets.QFileDialog.getOpenFileNames(caption='Open file',
                                                                          dir=self.project_directory,
                                                                          filter=files_types)

        if len(filenames) > 0:
            self.open_file_now(filenames, post_function)

    def select_csv_file(self, caption='Open CSV file'):
        """
        Select a CSV file
        :return: csv file path
        """
        files_types = "CSV (*.csv)"

        filename, type_selected = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                                        caption=caption,
                                                                        dir=self.project_directory,
                                                                        filter=files_types)

        if len(filename) > 0:
            return filename
        else:
            return None

    def open_file_now(self, filenames, post_function=None):
        """
        Open a file without questions
        :param filenames: list of file names (may be more than one because of CIM TP and EQ files)
        :param post_function: function callback
        :return: Nothing
        """
        if len(filenames) > 0:
            self.file_name = filenames[0]

            # store the working directory
            self.project_directory = os.path.dirname(self.file_name)

            # lock the ui
            self.LOCK()

            # create thread
            self.open_file_thread_object = filedrv.FileOpenThread(
                file_name=filenames if len(filenames) > 1 else filenames[0]
            )

            # make connections
            self.open_file_thread_object.progress_signal.connect(self.ui.progressBar.setValue)
            self.open_file_thread_object.progress_text.connect(self.ui.progress_label.setText)
            self.open_file_thread_object.done_signal.connect(self.UNLOCK)
            if post_function is None:
                self.open_file_thread_object.done_signal.connect(self.post_open_file)
            else:
                self.open_file_thread_object.done_signal.connect(post_function)

            # thread start
            self.open_file_thread_object.start()

            # register as the latest file driver
            self.last_file_driver = self.open_file_thread_object

            # register thread
            self.stuff_running_now.append('file_open')

    def post_open_file(self):
        """
        Actions to perform after a file has been loaded
        """

        self.stuff_running_now.remove('file_open')

        if self.open_file_thread_object is not None:

            if len(self.open_file_thread_object.logger) > 0:
                dlg = LogsDialogue('Open file logger', self.open_file_thread_object.logger)
                dlg.exec_()

            if self.open_file_thread_object.valid:

                # assign the loaded circuit
                self.new_project_now()
                self.circuit = self.open_file_thread_object.circuit
                self.file_name = self.open_file_thread_object.file_name

                if len(self.circuit.buses) > 1500:
                    quit_msg = "The grid is quite large, hence the schematic might be slow.\n" \
                               "Do you want to enable the schematic?\n" \
                               "(you can always enable the drawing later)"
                    reply = QtWidgets.QMessageBox.question(self, 'Enable schematic', quit_msg,
                                                           QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

                    if reply == QtWidgets.QMessageBox.No:
                        self.ui.draw_schematic_checkBox.setChecked(False)
                        self.set_grid_editor_state()
                else:
                    if not self.ui.draw_schematic_checkBox.isChecked():
                        # the schematic is disabled but the grid size is ok
                        self.ui.draw_schematic_checkBox.setChecked(True)
                        self.set_grid_editor_state()

                # create schematic
                self.create_schematic_from_api(explode_factor=1, show_msg=False)

                # set circuit name
                self.grid_editor.name_label.setText(str(self.circuit.name))

                # set base magnitudes
                self.ui.sbase_doubleSpinBox.setValue(self.circuit.Sbase)
                self.ui.fbase_doubleSpinBox.setValue(self.circuit.fBase)

                # set circuit comments
                try:
                    self.ui.comments_textEdit.setText(str(self.circuit.comments))
                except:
                    pass

                # update the drop-down menus that display dates
                self.update_date_dependent_combos()
                self.update_area_combos()

                # get the session tree structure
                session_data_dict = self.open_file_thread_object.get_session_tree()
                mdl = gf.get_tree_model(session_data_dict, 'Sessions')
                self.ui.diskSessionsTreeView.setModel(mdl)

                # apply the GUI settings if found:
                if 'gui_config' in self.open_file_thread_object.json_files:
                    self.apply_gui_config(data=self.open_file_thread_object.json_files['gui_config'])

                # clear the results
                self.clear_results()

            else:
                warn('The file was not valid')
        else:
            # center nodes
            self.grid_editor.align_schematic()

        self.collect_memory()

    def add_circuit(self):
        """
        Prompt to add another circuit
        """
        self.open_file_threaded(post_function=self.post_add_circuit)

    def post_add_circuit(self):
        """
        Stuff to do after opening another circuit
        :return: Nothing
        """
        self.stuff_running_now.remove('file_open')

        if self.open_file_thread_object is not None:

            if len(self.open_file_thread_object.logger) > 0:
                dlg = LogsDialogue('Open file logger', self.open_file_thread_object.logger)
                dlg.exec_()

            if self.open_file_thread_object.valid:

                if len(self.circuit.buses) == 0:
                    # load the circuit
                    self.stuff_running_now.append('file_open')
                    self.post_open_file()
                else:
                    # add the circuit
                    buses = self.circuit.add_circuit(self.open_file_thread_object.circuit, angle=0)

                    # add to schematic
                    self.grid_editor.add_circuit_to_schematic(self.open_file_thread_object.circuit, explode_factor=1.0)
                    self.grid_editor.align_schematic()

                    for bus in buses:
                        if bus.graphic_obj is not None:
                            bus.graphic_obj.setSelected(True)

    def update_date_dependent_combos(self):
        """
        update the drop down menus that display dates
        """
        if self.circuit.time_profile is not None:
            mdl = gf.get_list_model(self.circuit.time_profile)
            # setup profile sliders
            self.set_up_profile_sliders()
        else:
            mdl = QtGui.QStandardItemModel()
        self.ui.profile_time_selection_comboBox.setModel(mdl)
        self.ui.vs_departure_comboBox.setModel(mdl)
        self.ui.vs_target_comboBox.setModel(mdl)

    def update_area_combos(self):
        """
        Update the area dependent combos
        """
        n = len(self.circuit.areas)
        mdl1 = gf.get_list_model([str(elm) for elm in self.circuit.areas], checks=True)
        mdl2 = gf.get_list_model([str(elm) for elm in self.circuit.areas], checks=True)

        self.ui.areaFromListView.setModel(mdl1)
        self.ui.areaToListView.setModel(mdl2)

        if n > 1:
            self.ui.areaFromListView.model().item(0).setCheckState(QtCore.Qt.Checked)
            self.ui.areaToListView.model().item(1).setCheckState(QtCore.Qt.Checked)

    def save_file_as(self):
        """
        Save this file as...
        """
        # by deleting the file_name, the save_file function will ask for it
        self.file_name = ''
        self.save_file()

    def save_file(self):
        """
        Save the circuit case to a file
        """
        # declare the allowed file types
        files_types = "GridCal zip (*.gridcal);;" \
                      "GridCal HDF5 (*.gch5);;" \
                      "Excel (*.xlsx);;" \
                      "CIM (*.xml);;" \
                      "Electrical Json V3 (*.ejson3);;" \
                      "Rawx (*.rawx);;" \
                      "Sqlite (*.sqlite);;"

        if NEWTON_PA_AVAILABLE:
            files_types += "Newton (*.newton);;"

        if PGM_AVAILABLE:
            files_types += "PGM Json (*.pgm);;"

        # call dialog to select the file
        if self.project_directory is None:
            self.project_directory = ''

        # set grid name
        self.circuit.name = self.grid_editor.name_label.text()

        # gather comments
        self.circuit.comments = self.ui.comments_textEdit.toPlainText()

        if self.file_name == '':
            # if the global file_name is empty, ask where to save
            fname = os.path.join(self.project_directory, self.grid_editor.name_label.text())

            filename, type_selected = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', fname, files_types)

            if filename != '':

                # if the user did not enter the extension, add it automatically
                name, file_extension = os.path.splitext(filename)

                extension = dict()
                extension['Excel (*.xlsx)'] = '.xlsx'
                extension['CIM (*.xml)'] = '.xml'
                extension['Electrical Json V2 (*.ejson2)'] = '.ejson2'
                extension['Electrical Json V3 (*.ejson3)'] = '.ejson3'
                extension['GridCal zip (*.gridcal)'] = '.gridcal'
                extension['PSSe rawx (*.rawx)'] = '.rawx'
                extension['GridCal HDF5 (*.gch5)'] = '.gch5'
                extension['Sqlite (*.sqlite)'] = '.sqlite'
                extension['Newton (*.newton)'] = '.newton'
                extension['PGM Json (*.pgm)'] = '.pgm'

                if file_extension == '':
                    filename = name + extension[type_selected]

                # we were able to compose the file correctly, now save it
                self.file_name = filename
                self.save_file_now(self.file_name)
        else:
            # save directly
            self.save_file_now(self.file_name)

    def save_file_now(self, filename):
        """
        Save the file right now, without questions
        :param filename: filename to save to
        """

        if ('file_save' not in self.stuff_running_now) and ('file_open' not in self.stuff_running_now):
            # lock the ui
            self.LOCK()

            # check to not to kill threads avoiding segmentation faults
            if self.save_file_thread_object is not None:
                if self.save_file_thread_object.isRunning():
                    ok = yes_no_question("There is a saving procedure running.\nCancel and retry?")
                    if ok:
                        self.save_file_thread_object.quit()

            simulation_drivers = self.get_simulations()

            if self.ui.saveResultsCheckBox.isChecked():
                sessions = [self.session]
            else:
                sessions = []

            # get json files to store
            json_files = {"gui_config": self.get_gui_config_data()}

            self.save_file_thread_object = filedrv.FileSaveThread(circuit=self.circuit,
                                                                  file_name=filename,
                                                                  simulation_drivers=simulation_drivers,
                                                                  sessions=sessions,
                                                                  json_files=json_files)

            # make connections
            self.save_file_thread_object.progress_signal.connect(self.ui.progressBar.setValue)
            self.save_file_thread_object.progress_text.connect(self.ui.progress_label.setText)
            self.save_file_thread_object.done_signal.connect(self.UNLOCK)
            self.save_file_thread_object.done_signal.connect(self.post_file_save)

            # thread start
            self.save_file_thread_object.start()

            # register as the latest file driver
            self.last_file_driver = self.save_file_thread_object

            self.stuff_running_now.append('file_save')

        else:
            warning_msg('There is a file being processed..')

    def post_file_save(self):
        """
        Actions after the threaded file save
        """
        if len(self.save_file_thread_object.logger) > 0:
            dlg = LogsDialogue('Save file logger', self.save_file_thread_object.logger)
            dlg.exec_()

        self.stuff_running_now.remove('file_save')

        self.ui.model_version_label.setText('Model v. ' + str(self.circuit.model_version))

        # get the session tree structure
        session_data_dict = self.save_file_thread_object.get_session_tree()
        mdl = gf.get_tree_model(session_data_dict, 'Sessions')
        self.ui.diskSessionsTreeView.setModel(mdl)

        # call the garbage collector to free memory
        self.collect_memory()

    def closeEvent(self, event):
        """
        Close event
        :param event:
        :return:
        """
        if len(self.circuit.buses) > 0:
            quit_msg = "Are you sure that you want to exit GridCal?"
            reply = QtWidgets.QMessageBox.question(self, 'Close', quit_msg,
                                                   QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                self.delete_created_files()
                # save config regardless
                self.save_gui_config()
                self.stop_all_threads()
                event.accept()
            else:
                # save config regardless
                self.save_gui_config()
                event.ignore()
        else:
            # no buses so exit
            self.delete_created_files()
            # save config regardless
            self.save_gui_config()
            self.stop_all_threads()
            event.accept()

    def export_object_profiles(self):
        """
        Export object profiles
        """
        if self.circuit.time_profile is not None:

            # declare the allowed file types
            files_types = "Excel file (*.xlsx)"
            # call dialog to select the file
            if self.project_directory is None:
                self.project_directory = ''

            # set grid name
            self.circuit.name = self.grid_editor.name_label.text()

            fname = os.path.join(self.project_directory, 'profiles of ' + self.grid_editor.name_label.text())

            filename, type_selected = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', fname, files_types)

            if filename != "":
                if not filename.endswith('.xlsx'):
                    filename += '.xlsx'
                # TODO: correct this function
                self.circuit.export_profiles(file_name=filename)
        else:
            warning_msg('There are no profiles!', 'Export object profiles')

    def export_all(self):
        """
        Export all the results
        :return:
        """

        # set grid name
        self.circuit.name = self.grid_editor.name_label.text()

        available_results = self.get_available_results()

        if len(available_results) > 0:

            files_types = "Zip file (*.zip)"
            fname = os.path.join(self.project_directory, 'Results of ' + self.grid_editor.name_label.text())

            filename, type_selected = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', fname, files_types)

            if filename != "":
                self.LOCK()

                self.stuff_running_now.append('export_all')
                self.export_all_thread_object = exprtdrv.ExportAllThread(circuit=self.circuit,
                                                                         simulations_list=available_results,
                                                                         file_name=filename)

                self.export_all_thread_object.progress_signal.connect(self.ui.progressBar.setValue)
                self.export_all_thread_object.progress_text.connect(self.ui.progress_label.setText)
                self.export_all_thread_object.done_signal.connect(self.post_export_all)
                self.export_all_thread_object.start()
        else:
            warning_msg('There are no result available :/')

    def post_export_all(self):
        """
        Actions post export all
        """
        self.stuff_running_now.remove('export_all')

        if self.export_all_thread_object is not None:
            if self.export_all_thread_object.logger.has_logs():
                dlg = LogsDialogue('Export all', self.export_all_thread_object.logger)
                dlg.exec_()

        if len(self.stuff_running_now) == 0:
            self.UNLOCK()

    def export_simulation_data(self):
        """
        Export the calculation objects to file
        """

        # declare the allowed file types
        files_types = "Excel file (*.xlsx)"
        # call dialog to select the file
        if self.project_directory is None:
            self.project_directory = ''

        # set grid name
        self.circuit.name = self.grid_editor.name_label.text()

        fname = os.path.join(self.project_directory, self.grid_editor.name_label.text())

        filename, type_selected = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', fname, files_types)

        if filename != "":
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'

            numerical_circuit = core.compile_numerical_circuit_at(circuit=self.circuit)
            calculation_inputs = numerical_circuit.split_into_islands()

            with pd.ExcelWriter(filename) as writer:  # pylint: disable=abstract-class-instantiated

                for c, calc_input in enumerate(calculation_inputs):

                    for elm_type in calc_input.available_structures:
                        name = elm_type + '_' + str(c)
                        df = calc_input.get_structure(elm_type).astype(str)
                        df.to_excel(writer, name)

    def export_diagram(self):
        """
        Save the schematic
        :return:
        """
        if self.grid_editor is not None:

            # declare the allowed file types
            files_types = "Scalable Vector Graphics (*.svg);;Portable Network Graphics (*.png)"

            fname = os.path.join(self.project_directory, self.grid_editor.name_label.text())

            # call dialog to select the file
            filename, type_selected = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', fname, files_types)

            if not (filename.endswith('.svg') or filename.endswith('.png')):
                filename += ".svg"

            if filename != "":
                # save in factor * K
                factor = self.ui.resolution_factor_spinBox.value()
                w = 1920 * factor
                h = 1080 * factor
                self.grid_editor.export(filename, w, h)

    def draw_schematic(self):
        """
        Sandbox to call create_schematic_from_api from the action item without affecting the explode factor variable
        """
        self.ui.draw_schematic_checkBox.setChecked(True)
        self.set_grid_editor_state()
        self.create_schematic_from_api()

    def set_xy_from_lat_lon(self):
        """
        Get the x, y coordinates of the buses from their latitude and longitude
        """
        if len(self.circuit.buses) > 0:
            if yes_no_question("All nodes will be positioned to a 2D plane projection of their latitude and longitude. "
                               "Are you sure of this?"):
                logger = self.circuit.fill_xy_from_lat_lon(destructive=True, factor=0.01, remove_offset=True)

                if len(logger) > 0:
                    dlg = LogsDialogue('Set xy from lat lon', logger)
                    dlg.exec_()

                self.create_schematic_from_api()

    def create_schematic_from_api(self, explode_factor=1.0, show_msg=True):
        """
        This function explores the API values and draws a schematic layout
        @return:
        """
        if self.ui.draw_schematic_checkBox.isChecked():
            # set pointer to the circuit
            self.grid_editor.circuit = self.circuit

            self.grid_editor.schematic_from_api(explode_factor=explode_factor)

            # center nodes
            self.grid_editor.align_schematic()
        else:
            if show_msg:
                info_msg('The schematic drawing is disabled')

    def post_create_schematic(self):
        """

        :return:
        """
        self.UNLOCK()

    def auto_rate_branches(self):
        """
        Rate the Branches that do not have rate
        """

        branches = self.circuit.get_branches()

        if len(branches) > 0:
            pf_drv, pf_results = self.session.get_driver_results(sim.SimulationTypes.PowerFlow_run)

            if pf_results is not None:
                factor = self.ui.branch_rating_doubleSpinBox.value()

                for i, branch in enumerate(branches):

                    S = pf_results.Sf[i]

                    if branch.rate < 1e-3 or self.ui.rating_override_checkBox.isChecked():
                        r = np.round(abs(S) * factor, 1)
                        branch.rate = r if r > 0.0 else 1.0
                    else:
                        pass  # the rate is ok

            else:
                info_msg('Run a power flow simulation first.\nThe results are needed in this function.')

        else:
            warning_msg('There are no Branches!')

    def detect_transformers(self):
        """
        Detect which Branches are transformers
        """
        if len(self.circuit.lines) > 0:

            for elm in self.circuit.lines:

                v1 = elm.bus_from.Vnom
                v2 = elm.bus_to.Vnom

                if abs(v1 - v2) > 1.0:
                    self.circuit.convert_line_to_transformer(elm)
                else:

                    pass  # is a line

        else:
            warning_msg('There are no Branches!')

    def create_objects_model(self, elements, elm_type):

        dictionary_of_lists = dict()

        if elm_type == dev.DeviceType.BusDevice.value:
            elm = dev.Bus()
            dictionary_of_lists = {dev.DeviceType.AreaDevice.value: self.circuit.areas,
                                   dev.DeviceType.ZoneDevice.value: self.circuit.zones,
                                   dev.DeviceType.SubstationDevice.value: self.circuit.substations,
                                   dev.DeviceType.CountryDevice.value: self.circuit.countries,
                                   }

        elif elm_type == dev.DeviceType.BranchDevice.value:

            elm = dev.Branch(None, None)
            elements = list()

            self.view_template_controls(True)

        elif elm_type == dev.DeviceType.LoadDevice.value:
            elm = dev.Load()

        elif elm_type == dev.DeviceType.StaticGeneratorDevice.value:
            elm = dev.StaticGenerator()

        elif elm_type == dev.DeviceType.GeneratorDevice.value:
            elm = dev.Generator()
            dictionary_of_lists = {dev.DeviceType.Technology.value: self.circuit.technologies,
                                   dev.DeviceType.FuelDevice.value: self.circuit.fuels,
                                   dev.DeviceType.EmissionGasDevice.value: self.circuit.emission_gases, }

        elif elm_type == dev.DeviceType.BatteryDevice.value:
            elm = dev.Battery()
            dictionary_of_lists = {dev.DeviceType.Technology.value: self.circuit.technologies, }

        elif elm_type == dev.DeviceType.ShuntDevice.value:
            elm = dev.Shunt()

        elif elm_type == dev.DeviceType.ExternalGridDevice.value:
            elm = dev.ExternalGrid()

        elif elm_type == dev.DeviceType.LineDevice.value:
            elm = dev.Line(None, None)

        elif elm_type == dev.DeviceType.Transformer2WDevice.value:
            elm = dev.Transformer2W(None, None)

        elif elm_type == dev.DeviceType.WindingDevice.value:
            elm = dev.Winding(None, None)

        elif elm_type == dev.DeviceType.Transformer3WDevice.value:
            elm = dev.Transformer3W()

        elif elm_type == dev.DeviceType.HVDCLineDevice.value:
            elm = dev.HvdcLine(None, None)

        elif elm_type == dev.DeviceType.VscDevice.value:
            elm = dev.VSC(None, None)

        elif elm_type == dev.DeviceType.UpfcDevice.value:
            elm = dev.UPFC(None, None)

        elif elm_type == dev.DeviceType.DCLineDevice.value:
            elm = dev.DcLine(None, None)

        elif elm_type == dev.DeviceType.SubstationDevice.value:
            elm = dev.Substation()

        elif elm_type == dev.DeviceType.ZoneDevice.value:
            elm = dev.Zone()

        elif elm_type == dev.DeviceType.AreaDevice.value:
            elm = dev.Area()

        elif elm_type == dev.DeviceType.CountryDevice.value:
            elm = dev.Country()

        elif elm_type == dev.DeviceType.ContingencyDevice.value:
            elm = dev.Contingency()
            dictionary_of_lists = {dev.DeviceType.ContingencyGroupDevice.value: self.circuit.contingency_groups, }

        elif elm_type == dev.DeviceType.ContingencyGroupDevice.value:
            elm = dev.ContingencyGroup()

        elif elm_type == dev.DeviceType.InvestmentDevice.value:
            elm = dev.Investment()
            dictionary_of_lists = {dev.DeviceType.InvestmentsGroupDevice.value: self.circuit.investments_groups, }

        elif elm_type == dev.DeviceType.InvestmentsGroupDevice.value:
            elm = dev.InvestmentsGroup()

        elif elm_type == dev.DeviceType.Technology.value:
            elm = dev.Technology()

        elif elm_type == dev.DeviceType.FuelDevice.value:
            elm = dev.Fuel()

        elif elm_type == dev.DeviceType.EmissionGasDevice.value:
            elm = dev.EmissionGas()

        elif elm_type == dev.DeviceType.WireDevice.value:
            elm = dev.Wire()

        elif elm_type == dev.DeviceType.OverheadLineTypeDevice.value:
            elm = dev.OverheadLineType()

        elif elm_type == dev.DeviceType.SequenceLineDevice.value:
            elm = dev.SequenceLineType()

        elif elm_type == dev.DeviceType.UnderGroundLineDevice.value:
            elm = dev.UndergroundLineType()

        elif elm_type == dev.DeviceType.TransformerTypeDevice.value:
            elm = dev.TransformerType()

        elif elm_type == dev.DeviceType.GeneratorTechnologyAssociation.value:
            elm = dev.GeneratorTechnology()
            dictionary_of_lists = {dev.DeviceType.GeneratorDevice.value: self.circuit.get_generators(),
                                   dev.DeviceType.Technology.value: self.circuit.technologies, }

        elif elm_type == dev.DeviceType.GeneratorFuelAssociation.value:
            elm = dev.GeneratorFuel()
            dictionary_of_lists = {dev.DeviceType.GeneratorDevice.value: self.circuit.get_generators(),
                                   dev.DeviceType.FuelDevice.value: self.circuit.fuels, }

        elif elm_type == dev.DeviceType.GeneratorEmissionAssociation.value:
            elm = dev.GeneratorEmission()
            dictionary_of_lists = {dev.DeviceType.GeneratorDevice.value: self.circuit.get_generators(),
                                   dev.DeviceType.EmissionGasDevice.value: self.circuit.emission_gases, }


        else:
            raise Exception('elm_type not understood: ' + elm_type)

        if elm_type == 'Branches':
            mdl = gf.BranchObjectModel(elements, elm.editable_headers,
                                       parent=self.ui.dataStructureTableView,
                                       editable=True,
                                       non_editable_attributes=elm.non_editable_attributes)
        else:

            mdl = gf.ObjectsModel(elements, elm.editable_headers,
                                  parent=self.ui.dataStructureTableView,
                                  editable=True,
                                  non_editable_attributes=elm.non_editable_attributes,
                                  dictionary_of_lists=dictionary_of_lists)

        return mdl

    def display_filter(self, elements):
        """
        Display a list of elements that comes from a filter
        :param elements:
        """
        if len(elements) > 0:

            elm = elements[0]

            mdl = self.create_objects_model(elements=elements, elm_type=elm.device_type.value)

            self.ui.dataStructureTableView.setModel(mdl)

        else:
            self.ui.dataStructureTableView.setModel(None)

    def view_objects_data(self):
        """
        On click, display the objects properties
        """
        if self.ui.dataStructuresTreeView.selectedIndexes()[
            0].parent().row() > -1:  # if the clicked element has a valid parent

            elm_type = self.ui.dataStructuresTreeView.selectedIndexes()[0].data(role=QtCore.Qt.DisplayRole)

            elements = self.circuit.get_elements_by_type(element_type=dev.DeviceType(elm_type))

            mdl = self.create_objects_model(elements=elements, elm_type=elm_type)

            self.type_objects_list = elements
            self.ui.dataStructureTableView.setModel(mdl)
            self.ui.property_comboBox.clear()
            self.ui.property_comboBox.addItems(mdl.attributes)
        else:
            self.ui.dataStructureTableView.setModel(None)
            self.ui.property_comboBox.clear()

    def view_simulation_objects_data(self):
        """
        Simulation data structure clicked
        """

        i = self.ui.simulation_data_island_comboBox.currentIndex()

        if i > -1 and len(self.circuit.buses) > 0:
            elm_type = self.ui.simulationDataStructuresListView.selectedIndexes()[0].data(role=QtCore.Qt.DisplayRole)

            df = self.calculation_inputs_to_display[i].get_structure(elm_type)

            mdl = gf.PandasModel(df)

            self.ui.simulationDataStructureTableView.setModel(mdl)

    def copy_simulation_objects_data(self):
        """
        Copy the arrays of the compiled arrays view to the clipboard
        """
        mdl = self.ui.simulationDataStructureTableView.model()
        mode = self.ui.arrayModeComboBox.currentText()
        mdl.copy_to_clipboard(mode=mode)

    def copy_simulation_objects_data_to_numpy(self):
        """
        Copy the arrays of the compiled arrays view to the clipboard
        """
        mdl = self.ui.simulationDataStructureTableView.model()
        mode = 'numpy'
        mdl.copy_to_clipboard(mode=mode)

    def plot_simulation_objects_data(self):
        """
        Plot the arrays of the compiled arrays view
        """
        mdl = self.ui.simulationDataStructureTableView.model()
        data = mdl.data_c

        # declare figure
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        if mdl.is_2d():
            ax1.spy(data)

        else:
            if mdl.is_complex():
                ax1.scatter(data.real, data.imag)
                ax1.set_xlabel('Real')
                ax1.set_ylabel('Imag')
            else:
                arr = np.arange(data.shape[0])
                ax1.scatter(arr, data)
                ax1.set_xlabel('Position')
                ax1.set_ylabel('Value')

        fig.tight_layout()
        plt.show()

    def profile_device_type_changed(self):
        """

        """
        dev_type = self.ui.profile_device_type_comboBox.currentText()
        mdl = gf.get_list_model(self.circuit.profile_magnitudes[dev_type][0])
        self.ui.device_type_magnitude_comboBox.setModel(mdl)
        self.ui.device_type_magnitude_comboBox_2.setModel(mdl)

    def new_profiles_structure(self):
        """
        Create new profiles structure
        :return:
        """
        dlg = NewProfilesStructureDialogue()
        if dlg.exec_():
            steps, step_length, step_unit, time_base = dlg.get_values()

            self.ui.profiles_tableView.setModel(None)

            self.circuit.create_profiles(steps, step_length, step_unit, time_base)

            self.display_profiles()

            self.set_up_profile_sliders()

            self.update_date_dependent_combos()

    def delete_profiles_structure(self):
        """
        Delete all profiles
        :return: Nothing
        """

        if self.circuit.time_profile is not None:
            quit_msg = "Are you sure that you want to remove the profiles?"
            reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                                   QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                for bus in self.circuit.buses:
                    bus.delete_profiles()
                self.circuit.time_profile = None
                self.ui.profiles_tableView.setModel(None)
                self.set_up_profile_sliders()
                self.update_date_dependent_combos()
                self.update_area_combos()
            else:
                pass
        else:
            warning_msg('There are no profiles', 'Delete profiles')

    def import_profiles(self):
        """
        Profile importer
        """

        # Load(), StaticGenerator(), Generator(), Battery(), Shunt()

        dev_type_text = self.ui.profile_device_type_comboBox.currentText()
        magnitudes, mag_types = self.circuit.profile_magnitudes[dev_type_text]

        idx = self.ui.device_type_magnitude_comboBox.currentIndex()
        magnitude = magnitudes[idx]

        dev_type = self.circuit.device_type_name_dict[dev_type_text]
        objects = self.circuit.get_elements_by_type(dev_type)

        if len(objects) > 0:
            self.profile_input_dialogue = ProfileInputGUI(parent=self,
                                                          list_of_objects=objects,
                                                          magnitudes=[magnitude])

            self.profile_input_dialogue.resize(int(1.61 * 600.0), 550)  # golden ratio
            self.profile_input_dialogue.exec_()  # exec leaves the parent on hold

            if self.profile_input_dialogue.time is not None:

                # if there are no profiles:
                if self.circuit.time_profile is None:
                    self.circuit.format_profiles(self.profile_input_dialogue.time)

                elif len(self.profile_input_dialogue.time) != len(self.circuit.time_profile):
                    warning_msg("The imported profile length does not match the existing one.\n"
                                "Delete the existing profiles before continuing.\n"
                                "The import action will not be performed")
                    return False

                # Assign profiles
                for i, elm in enumerate(objects):
                    if not self.profile_input_dialogue.zeroed[i]:

                        if self.profile_input_dialogue.normalized:
                            base_value = getattr(elm, magnitude)
                            data = self.profile_input_dialogue.data[:, i] * base_value
                        else:
                            data = self.profile_input_dialogue.data[:, i]

                        # assign the profile to the object
                        prof_attr = elm.properties_with_profile[magnitude]
                        setattr(elm, prof_attr, data)
                        # elm.profile_f[magnitude](dialogue.time, dialogue.data[:, i], dialogue.normalized)
                    else:
                        print(elm.name, 'skipped')

                # set up sliders
                self.set_up_profile_sliders()
                self.update_date_dependent_combos()
                self.display_profiles()

                # ask to update active profile when magnitude is P for generators and loads
                if len(objects) > 0:
                    if magnitude == 'P':
                        if objects[0].device_type == dev.DeviceType.GeneratorDevice:
                            ok = yes_no_question(
                                "Do you want to correct the generators active profile based on the active power profile?",
                                "Match")
                            if ok:
                                self.fix_generators_active_based_on_the_power(ask_before=False)
                        elif objects[0].device_type == dev.DeviceType.LoadDevice:
                            ok = yes_no_question(
                                "Do you want to correct the loads active profile based on the active power profile?",
                                "Match")
                            if ok:
                                self.fix_loads_active_based_on_the_power(ask_before=False)

            else:
                pass  # the dialogue was closed

        else:
            warning_msg("There are no objects to which to assign a profile. \nYou need to load or create a grid!")

    def modify_profiles(self, operation='+'):
        """
        Edit profiles with a linear combination
        Args:
            operation: '+', '-', '*', '/'

        Returns: Nothing
        """
        value = self.ui.profile_factor_doubleSpinBox.value()

        dev_type_text = self.ui.profile_device_type_comboBox.currentText()
        magnitudes, mag_types = self.circuit.profile_magnitudes[dev_type_text]
        idx = self.ui.device_type_magnitude_comboBox.currentIndex()
        magnitude = magnitudes[idx]

        dev_type = self.circuit.device_type_name_dict[dev_type_text]
        objects = self.circuit.get_elements_by_type(dev_type)
        # Assign profiles
        if len(objects) > 0:

            indices = self.ui.profiles_tableView.selectedIndexes()

            attr = objects[0].properties_with_profile[magnitude]

            model = self.ui.profiles_tableView.model()

            mod_cols = list()

            if len(indices) == 0:
                # no index was selected
                for i, elm in enumerate(objects):

                    tpe = getattr(elm, attr).dtype

                    if operation == '+':
                        setattr(elm, attr, (getattr(elm, attr) + value).astype(tpe))
                        mod_cols.append(i)

                    elif operation == '-':
                        setattr(elm, attr, (getattr(elm, attr) - value).astype(tpe))
                        mod_cols.append(i)

                    elif operation == '*':
                        setattr(elm, attr, (getattr(elm, attr) * value).astype(tpe))
                        mod_cols.append(i)

                    elif operation == '/':
                        setattr(elm, attr, (getattr(elm, attr) / value).astype(tpe))
                        mod_cols.append(i)

                    elif operation == 'set':
                        arr = getattr(elm, attr)
                        setattr(elm, attr, (np.ones(len(arr)) * value).astype(tpe))
                        mod_cols.append(i)

                    else:
                        raise Exception('Operation not supported: ' + str(operation))

            else:
                # indices were selected ...

                for idx in indices:

                    elm = objects[idx.column()]
                    tpe = type(getattr(elm, attr))

                    if operation == '+':
                        getattr(elm, attr)[idx.row()] += value
                        mod_cols.append(idx.column())

                    elif operation == '-':
                        getattr(elm, attr)[idx.row()] -= value
                        mod_cols.append(idx.column())

                    elif operation == '*':
                        getattr(elm, attr)[idx.row()] *= value
                        mod_cols.append(idx.column())

                    elif operation == '/':
                        getattr(elm, attr)[idx.row()] /= value
                        mod_cols.append(idx.column())

                    elif operation == 'set':
                        getattr(elm, attr)[idx.row()] = value
                        mod_cols.append(idx.column())

                    else:
                        raise Exception('Operation not supported: ' + str(operation))

            model.add_state(mod_cols, 'linear combinations')
            model.update()

    def set_profile_as_linear_combination(self):
        """
        Edit profiles with a linear combination
        Returns: Nothing
        """

        # value = self.ui.profile_factor_doubleSpinBox.value()

        dev_type_text = self.ui.profile_device_type_comboBox.currentText()
        magnitudes, mag_types = self.circuit.profile_magnitudes[dev_type_text]
        idx_from = self.ui.device_type_magnitude_comboBox.currentIndex()
        magnitude_from = magnitudes[idx_from]

        idx_to = self.ui.device_type_magnitude_comboBox_2.currentIndex()
        magnitude_to = magnitudes[idx_to]

        if len(self.circuit.buses) > 0 and magnitude_from != magnitude_to:

            msg = "Are you sure that you want to overwrite the values " + magnitude_to + \
                  " with the values of " + magnitude_from + "?"

            reply = QtWidgets.QMessageBox.question(self, 'Message', msg,
                                                   QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:

                dev_type = self.circuit.device_type_name_dict[dev_type_text]
                objects = self.circuit.get_elements_by_type(dev_type)

                # Assign profiles
                if len(objects) > 0:
                    attr_from = objects[0].properties_with_profile[magnitude_from]
                    attr_to = objects[0].properties_with_profile[magnitude_to]

                    for i, elm in enumerate(objects):
                        setattr(elm, attr_to, getattr(elm, attr_from) * 1.0)

                    self.display_profiles()

            else:
                # rejected the operation
                pass

        else:
            # no buses or no actual change
            pass

    def load_results_driver(self):
        """
        Load a driver from disk
        """
        idx = self.ui.diskSessionsTreeView.selectedIndexes()
        if len(idx) > 0:
            tree_mdl = self.ui.diskSessionsTreeView.model()
            item = tree_mdl.itemFromIndex(idx[0])
            path = gf.get_tree_item_path(item)

            if len(path) > 1:
                session_name = path[0]
                study_name = path[1]
                if self.last_file_driver is not None:
                    data_dict, json_files = self.last_file_driver.load_session_objects(session_name=session_name,
                                                                                       study_name=study_name)

                    self.session.register_driver_from_disk_data(self.circuit, study_name, data_dict)

                    self.update_available_results()
                else:
                    error_msg('No file driver declared :/')
            else:
                info_msg('Select a driver inside a session', 'Driver load from disk')

    def delete_results_driver(self):
        """
        Delete the driver
        :return:
        """
        idx = self.ui.results_treeView.selectedIndexes()
        if len(idx) > 0:
            tree_mdl = self.ui.results_treeView.model()
            item = tree_mdl.itemFromIndex(idx[0])
            path = gf.get_tree_item_path(item)

            if len(path) > 0:
                study_name = path[0]
                study_type = self.available_results_dict[study_name]

                quit_msg = "Do you want to delete the results driver " + study_name + "?"
                reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                                       QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

                if reply == QtWidgets.QMessageBox.Yes:
                    self.session.delete_driver_by_name(study_name)
                    self.update_available_results()

    def plot_profiles(self):
        """
        Plot profiles from the time events
        """
        value = self.ui.profile_factor_doubleSpinBox.value()

        dev_type_text = self.ui.profile_device_type_comboBox.currentText()
        magnitudes, mag_types = self.circuit.profile_magnitudes[dev_type_text]
        idx = self.ui.device_type_magnitude_comboBox.currentIndex()
        magnitude = magnitudes[idx]

        dev_type = self.circuit.device_type_name_dict[dev_type_text]
        objects = self.circuit.get_elements_by_type(dev_type)

        # get the selected element
        obj_idx = self.ui.profiles_tableView.selectedIndexes()

        t = self.circuit.time_profile

        # Assign profiles
        if len(obj_idx):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            k = obj_idx[0].column()
            units_dict = {attr: pair.units for attr, pair in objects[k].editable_headers.items()}

            unit = units_dict[magnitude]
            ax.set_ylabel(unit)

            # get the unique columns in the selected cells
            cols = set()
            for i in range(len(obj_idx)):
                cols.add(obj_idx[i].column())

            # plot every column
            dta = dict()
            for k in cols:
                attr = objects[k].properties_with_profile[magnitude]
                dta[objects[k].name] = getattr(objects[k], attr)
            df = pd.DataFrame(data=dta, index=t)
            df.plot(ax=ax)

            plt.show()

    def display_profiles(self):
        """
        Display profile
        """
        if self.circuit.time_profile is not None:

            dev_type_text = self.ui.profile_device_type_comboBox.currentText()

            magnitudes, mag_types = self.circuit.profile_magnitudes[dev_type_text]

            if len(magnitudes) > 0:
                # get the enumeration univoque association with he device text
                dev_type = self.circuit.device_type_name_dict[dev_type_text]

                idx = self.ui.device_type_magnitude_comboBox.currentIndex()
                magnitude = magnitudes[idx]
                mtype = mag_types[idx]

                mdl = gf.ProfilesModel(multi_circuit=self.circuit,
                                       device_type=dev_type,
                                       magnitude=magnitude,
                                       format=mtype,
                                       parent=self.ui.profiles_tableView)
            else:
                mdl = None

            self.ui.profiles_tableView.setModel(mdl)

    def get_clustering_results(self) -> Union[sim.ClusteringResults, None]:
        """
        Get the clustering results if available
        :return: ClusteringResults or None
        """
        if self.ui.actionUse_clustering.isChecked():
            _, clustering_results = self.session.get_driver_results(sim.SimulationTypes.ClusteringAnalysis_run)

            if clustering_results is not None:
                n = len(clustering_results.time_indices)

                if n != self.ui.cluster_number_spinBox.value():
                    error_msg("The number of clusters in the stored results is different from the specified :(\n"
                              "Run another clustering analysis.")

                    return None
                else:
                    # all ok
                    return clustering_results
            else:
                # no results ...
                warning_msg("There are no clustering results.")
                self.ui.actionUse_clustering.setChecked(False)
                return None

        else:
            # not marked ...
            return None

    def run_clustering(self):
        """
        Run a clustering analysis
        """
        if self.circuit.get_bus_number() > 0 and self.circuit.get_time_number() > 0:

            if not self.session.is_this_running(sim.SimulationTypes.ClusteringAnalysis_run):

                self.add_simulation(sim.SimulationTypes.ClusteringAnalysis_run)

                self.LOCK()

                # get the power flow options from the GUI
                options = sim.ClusteringAnalysisOptions(n_points=self.ui.cluster_number_spinBox.value())

                drv = sim.ClusteringDriver(grid=self.circuit,
                                           options=options)
                self.session.run(drv,
                                 post_func=self.post_clustering,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            else:
                warning_msg('Another clustering is being executed now...')
        else:
            pass

    def post_clustering(self):
        """
        Action performed after the short circuit.
        Returns:

        """
        # update the results in the circuit structures
        drv, results = self.session.get_driver_results(sim.SimulationTypes.ClusteringAnalysis_run)
        self.remove_simulation(sim.SimulationTypes.ClusteringAnalysis_run)
        if results is not None:

            self.update_available_results()
        else:
            error_msg('Something went wrong, There are no power short circuit results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def get_selected_power_flow_options(self):
        """
        Gather power flow run options
        :return:
        """
        solver_type = self.solvers_dict[self.ui.solver_comboBox.currentText()]

        q_control_mode = self.q_control_modes_dict[self.ui.reactive_power_control_mode_comboBox.currentText()]
        q_steepness_factor = 1.0
        taps_control_mode = self.taps_control_modes_dict[self.ui.taps_control_mode_comboBox.currentText()]

        verbose = self.ui.verbositySpinBox.value()

        exponent = self.ui.tolerance_spinBox.value()
        tolerance = 1.0 / (10.0 ** exponent)

        max_iter = self.ui.max_iterations_spinBox.value()

        max_outer_iter = 1000  # not used anymore

        dispatch_storage = False
        mu = self.ui.muSpinBox.value()

        if self.ui.helm_retry_checkBox.isChecked():
            retry_with_other_methods = True  # to set a value
        else:
            retry_with_other_methods = False

        if self.ui.apply_impedance_tolerances_checkBox.isChecked():
            branch_impedance_tolerance_mode = bs.BranchImpedanceMode.Upper
        else:
            branch_impedance_tolerance_mode = bs.BranchImpedanceMode.Specified

        temp_correction = self.ui.temperature_correction_checkBox.isChecked()

        distributed_slack = self.ui.distributed_slack_checkBox.isChecked()

        ignore_single_node_islands = self.ui.ignore_single_node_islands_checkBox.isChecked()

        use_stored_guess = self.ui.use_voltage_guess_checkBox.isChecked()

        override_branch_controls = self.ui.override_branch_controls_checkBox.isChecked()

        ops = sim.PowerFlowOptions(solver_type=solver_type,
                                   retry_with_other_methods=retry_with_other_methods,
                                   verbose=verbose,
                                   initialize_with_existing_solution=use_stored_guess,
                                   tolerance=tolerance,
                                   max_iter=max_iter,
                                   max_outer_loop_iter=max_outer_iter,
                                   control_q=q_control_mode,
                                   multi_core=False,
                                   dispatch_storage=dispatch_storage,
                                   control_taps=taps_control_mode,
                                   apply_temperature_correction=temp_correction,
                                   branch_impedance_tolerance_mode=branch_impedance_tolerance_mode,
                                   q_steepness_factor=q_steepness_factor,
                                   distributed_slack=distributed_slack,
                                   ignore_single_node_islands=ignore_single_node_islands,
                                   mu=mu,
                                   use_stored_guess=use_stored_guess,
                                   override_branch_controls=override_branch_controls)

        return ops

    def run_power_flow(self):
        """
        Run a power flow simulation
        :return:
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.PowerFlow_run):

                self.LOCK()

                self.add_simulation(sim.SimulationTypes.PowerFlow_run)

                self.ui.progress_label.setText('Compiling the grid...')
                QtGui.QGuiApplication.processEvents()

                # get the power flow options from the GUI
                options = self.get_selected_power_flow_options()

                # compute the automatic precision
                if self.ui.auto_precision_checkBox.isChecked():

                    options.tolerance, tol_idx = self.circuit.get_automatic_precision()

                    if tol_idx > 12:
                        tol_idx = 12

                    self.ui.tolerance_spinBox.setValue(tol_idx)

                use_opf = self.ui.actionOpf_to_Power_flow.isChecked()

                if use_opf:

                    drv, results = self.session.get_driver_results(sim.SimulationTypes.OPF_run)

                    if drv is not None:
                        if results is not None:
                            opf_results = results
                        else:
                            warning_msg('There are no OPF results, '
                                        'therefore this operation will not use OPF information.')
                            self.ui.actionOpf_to_Power_flow.setChecked(False)
                            opf_results = None
                    else:

                        # try the OPF-NTC...
                        drv, results = self.session.get_driver_results(sim.SimulationTypes.OPF_NTC_run)

                        if drv is not None:
                            if results is not None:
                                opf_results = results
                            else:
                                warning_msg('There are no OPF-NTC results, '
                                            'therefore this operation will not use OPF information.')
                                self.ui.actionOpf_to_Power_flow.setChecked(False)
                                opf_results = None
                        else:
                            warning_msg('There are no OPF results, '
                                        'therefore this operation will not use OPF information.')
                            self.ui.actionOpf_to_Power_flow.setChecked(False)
                            opf_results = None
                else:
                    opf_results = None

                self.ui.progress_label.setText('Running power flow...')
                QtGui.QGuiApplication.processEvents()
                # set power flow object instance
                engine = self.get_preferred_engine()
                drv = sim.PowerFlowDriver(self.circuit, options, opf_results, engine=engine)

                self.session.run(drv,
                                 post_func=self.post_power_flow,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            else:
                warning_msg('Another simulation of the same type is running...')
        else:
            pass

    def post_power_flow(self):
        """
        Action performed after the power flow.
        Returns:

        """
        # update the results in the circuit structures

        drv, results = self.session.get_driver_results(sim.SimulationTypes.PowerFlow_run)

        if results is not None:
            self.ui.progress_label.setText('Colouring power flow results in the grid...')
            # QtGui.QGuiApplication.processEvents()

            self.remove_simulation(sim.SimulationTypes.PowerFlow_run)

            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked() or len(self.bus_viewer_windows) > 0:
                self.colour_schematic()

            # print convergence reports on the console
            for report in drv.convergence_reports:
                msg_ = 'Power flow converged: \n' + report.to_dataframe().__str__() + '\n\n'
                self.console_msg(msg_)

        else:
            warning_msg('There are no power flow results.\nIs there any slack bus or generator?', 'Power flow')
            # QtGui.QGuiApplication.processEvents()

        if drv is not None:
            if len(drv.logger) > 0:
                dlg = LogsDialogue('Power flow', drv.logger)
                dlg.exec_()
            if len(drv.logger.debug_entries):
                self.console_msg(drv.logger.debug_entries)

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_short_circuit(self):
        """
        Run a short circuit simulation
        The short circuit simulation must be performed after a power flow simulation
        without any load or topology change
        :return:
        """
        if len(self.circuit.buses) > 0:
            if not self.session.is_this_running(sim.SimulationTypes.ShortCircuit_run):

                pf_drv, pf_results = self.session.get_driver_results(sim.SimulationTypes.PowerFlow_run)

                if pf_results is not None:

                    # Since we must run this study in the same conditions as
                    # the last power flow, no compilation is needed

                    # get the short circuit selected buses
                    sel_buses = list()
                    self_short_circuit_types = list()
                    for i, bus in enumerate(self.circuit.buses):
                        if bus.graphic_obj is not None:
                            if bus.graphic_obj.any_short_circuit():
                                sel_buses.append(i)
                                self_short_circuit_types.append(bus.graphic_obj.sc_type)

                    if len(sel_buses) > 1:
                        error_msg("GridCal only supports one short circuit bus at the time", "Short circuit")
                        return

                    if len(sel_buses) == 0:
                        warning_msg('You need to enable some buses for short circuit.'
                                    + '\nEnable them by right click, and selecting on the context menu.')
                    else:
                        self.add_simulation(sim.SimulationTypes.ShortCircuit_run)

                        self.LOCK()

                        if self.ui.apply_impedance_tolerances_checkBox.isChecked():
                            branch_impedance_tolerance_mode = bs.BranchImpedanceMode.Lower
                        else:
                            branch_impedance_tolerance_mode = bs.BranchImpedanceMode.Specified

                        # get the power flow options from the GUI
                        sc_options = sim.ShortCircuitOptions(bus_index=sel_buses[0],
                                                             fault_type=self_short_circuit_types[0],
                                                             branch_impedance_tolerance_mode=branch_impedance_tolerance_mode)

                        pf_options = self.get_selected_power_flow_options()

                        drv = sim.ShortCircuitDriver(grid=self.circuit,
                                                     options=sc_options,
                                                     pf_options=pf_options,
                                                     pf_results=pf_results)
                        self.session.run(drv,
                                         post_func=self.post_short_circuit,
                                         prog_func=self.ui.progressBar.setValue,
                                         text_func=self.ui.progress_label.setText)

                else:
                    info_msg('Run a power flow simulation first.\n'
                             'The results are needed to initialize this simulation.')
            else:
                warning_msg('Another short circuit is being executed now...')
        else:
            pass

    def post_short_circuit(self):
        """
        Action performed after the short circuit.
        Returns:

        """
        # update the results in the circuit structures
        drv, results = self.session.get_driver_results(sim.SimulationTypes.ShortCircuit_run)
        self.remove_simulation(sim.SimulationTypes.ShortCircuit_run)
        if results is not None:

            self.ui.progress_label.setText('Colouring short circuit results in the grid...')
            QtGui.QGuiApplication.processEvents()

            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()

        else:
            error_msg('Something went wrong, There are no power short circuit results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_linear_analysis(self):
        """
        Run a Power Transfer Distribution Factors analysis
        :return:
        """
        if len(self.circuit.buses) > 0:
            if not self.session.is_this_running(sim.SimulationTypes.LinearAnalysis_run):

                self.add_simulation(sim.SimulationTypes.LinearAnalysis_run)

                self.LOCK()

                options = sim.LinearAnalysisOptions(
                    distribute_slack=self.ui.ptdf_distributed_slack_checkBox.isChecked(),
                    correct_values=self.ui.ptdf_correct_nonsense_values_checkBox.isChecked())

                engine = self.get_preferred_engine()
                drv = sim.LinearAnalysisDriver(grid=self.circuit, options=options, engine=engine)

                self.session.run(drv,
                                 post_func=self.post_linear_analysis,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)
            else:
                warning_msg('Another PTDF is being executed now...')
        else:
            pass

    def post_linear_analysis(self):
        """
        Action performed after the short circuit.
        Returns:

        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.LinearAnalysis_run)

        self.remove_simulation(sim.SimulationTypes.LinearAnalysis_run)

        # update the results in the circuit structures
        # if not drv.__cancel__:
        if results is not None:

            self.ui.progress_label.setText('Colouring PTDF results in the grid...')
            QtGui.QGuiApplication.processEvents()

            self.update_available_results()
            self.colour_schematic()
        else:
            error_msg('Something went wrong, There are no PTDF results.')

        if len(drv.logger) > 0:
            dlg = LogsDialogue('PTDF', drv.logger)
            dlg.exec_()

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_linear_analysis_ts(self):
        """
        Run PTDF time series simulation
        """
        if len(self.circuit.buses) > 0:
            if self.valid_time_series():
                if not self.session.is_this_running(sim.SimulationTypes.LinearAnalysis_TS_run):

                    self.add_simulation(sim.SimulationTypes.LinearAnalysis_TS_run)
                    self.LOCK()

                    options = sim.LinearAnalysisOptions(distribute_slack=self.ui.distributed_slack_checkBox.isChecked())

                    drv = sim.LinearAnalysisTimeSeriesDriver(grid=self.circuit,
                                                             options=options,
                                                             time_indices=self.get_time_indices(),
                                                             clustering_results=self.get_clustering_results())

                    self.session.run(drv,
                                     post_func=self.post_linear_analysis_ts,
                                     prog_func=self.ui.progressBar.setValue,
                                     text_func=self.ui.progress_label.setText)
                else:
                    warning_msg('Another PTDF time series is being executed now...')
            else:
                warning_msg('There are no time series...')

    def post_linear_analysis_ts(self):
        """
        Action performed after the short circuit.
        Returns:

        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.LinearAnalysis_TS_run)
        self.remove_simulation(sim.SimulationTypes.LinearAnalysis_TS_run)

        # update the results in the circuit structures
        # if not drv.__cancel__:
        if results is not None:

            self.ui.progress_label.setText('Colouring PTDF results in the grid...')
            QtGui.QGuiApplication.processEvents()

            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked():
                if results.S.shape[0] > 0:
                    self.colour_schematic()
                else:
                    info_msg('Cannot colour because the PTDF results have zero time steps :/')

        else:
            error_msg('Something went wrong, There are no PTDF Time series results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_contingency_analysis(self):
        """
        Run a Power Transfer Distribution Factors analysis
        :return:
        """
        if len(self.circuit.buses) > 0:

            if len(self.circuit.contingency_groups) > 0:

                if not self.session.is_this_running(sim.SimulationTypes.ContingencyAnalysis_run):

                    self.add_simulation(sim.SimulationTypes.ContingencyAnalysis_run)

                    self.LOCK()

                    pf_options = self.get_selected_power_flow_options()

                    options = sim.ContingencyAnalysisOptions(
                        distributed_slack=self.ui.distributed_slack_checkBox.isChecked(),
                        use_provided_flows=False,
                        Pf=None,
                        pf_options=pf_options,
                        engine=self.contingency_engines_dict[self.ui.contingencyEngineComboBox.currentText()]
                    )

                    drv = sim.ContingencyAnalysisDriver(grid=self.circuit, options=options)

                    self.session.run(drv,
                                     post_func=self.post_contingency_analysis,
                                     prog_func=self.ui.progressBar.setValue,
                                     text_func=self.ui.progress_label.setText)
                else:
                    warning_msg('Another contingency analysis is being executed now...')

            else:
                warning_msg('There are no contingency groups declared...')
        else:
            pass

    def post_contingency_analysis(self):
        """
        Action performed after the short circuit.
        Returns:

        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.ContingencyAnalysis_run)
        self.remove_simulation(sim.SimulationTypes.ContingencyAnalysis_run)

        # update the results in the circuit structures
        # if not drv.__cancel__:
        if results is not None:

            self.ui.progress_label.setText('Colouring contingency analysis results in the grid...')
            QtGui.QGuiApplication.processEvents()

            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()
        else:
            error_msg('Something went wrong, There are no contingency analysis results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_contingency_analysis_ts(self) -> None:
        """
        Run a Power Transfer Distribution Factors analysis
        :return:
        """
        if len(self.circuit.buses) > 0:

            if len(self.circuit.contingency_groups) > 0:

                if self.valid_time_series():
                    if not self.session.is_this_running(sim.SimulationTypes.ContingencyAnalysisTS_run):

                        self.add_simulation(sim.SimulationTypes.ContingencyAnalysisTS_run)

                        self.LOCK()

                        pf_options = self.get_selected_power_flow_options()

                        options = sim.ContingencyAnalysisOptions(
                            distributed_slack=self.ui.distributed_slack_checkBox.isChecked(),
                            use_provided_flows=False,
                            Pf=None,
                            pf_options=pf_options,
                            engine=self.contingency_engines_dict[self.ui.contingencyEngineComboBox.currentText()]
                        )

                        drv = sim.ContingencyAnalysisTimeSeries(grid=self.circuit,
                                                                options=options,
                                                                time_indices=self.get_time_indices(),
                                                                clustering_results=self.get_clustering_results(),
                                                                engine=self.get_preferred_engine())

                        self.session.run(drv,
                                         post_func=self.post_contingency_analysis_ts,
                                         prog_func=self.ui.progressBar.setValue,
                                         text_func=self.ui.progress_label.setText)
                    else:
                        warning_msg('Another LODF is being executed now...')
                else:
                    warning_msg('There are no time series...')

            else:
                warning_msg('There are no contingency groups declared...')

        else:
            pass

    def post_contingency_analysis_ts(self) -> None:
        """
        Action performed after the short circuit.
        Returns:

        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.ContingencyAnalysisTS_run)
        self.remove_simulation(sim.SimulationTypes.ContingencyAnalysisTS_run)

        # update the results in the circuit structures
        # if not drv.__cancel__:
        if results is not None:

            self.ui.progress_label.setText('Colouring results in the grid...')
            QtGui.QGuiApplication.processEvents()

            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()
        else:
            error_msg('Something went wrong, There are no contingency time series results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_available_transfer_capacity(self):
        """
        Run a Power Transfer Distribution Factors analysis
        :return:
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.NetTransferCapacity_run):
                distributed_slack = self.ui.distributed_slack_checkBox.isChecked()
                dT = self.ui.atcPerturbanceSpinBox.value()
                threshold = self.ui.atcThresholdSpinBox.value()
                max_report_elements = 5  # TODO: self.ui.ntcReportLimitingElementsSpinBox.value()
                # available transfer capacity inter areas
                compatible_areas, lst_from, lst_to, lst_br, lst_hvdc_br, areas_from, areas_to = self.get_compatible_areas_from_to()

                if not compatible_areas:
                    return

                idx_from = np.array([i for i, bus in lst_from])
                idx_to = np.array([i for i, bus in lst_to])
                idx_br = np.array([i for i, bus, sense in lst_br])
                sense_br = np.array([sense for i, bus, sense in lst_br])

                # HVDC
                idx_hvdc_br = np.array([i for i, bus, sense in lst_hvdc_br])
                sense_hvdc_br = np.array([sense for i, bus, sense in lst_hvdc_br])

                if self.ui.usePfValuesForAtcCheckBox.isChecked():
                    pf_drv, pf_results = self.session.get_driver_results(sim.SimulationTypes.PowerFlow_run)
                    if pf_results is not None:
                        Pf = pf_results.Sf.real
                        Pf_hvdc = pf_results.hvdc_Pf.real
                        use_provided_flows = True
                    else:
                        warning_msg('There were no power flow values available. Linear flows will be used.')
                        use_provided_flows = False
                        Pf_hvdc = None
                        Pf = None
                else:
                    use_provided_flows = False
                    Pf = None
                    Pf_hvdc = None

                if len(idx_from) == 0:
                    error_msg('The area "from" has no buses!')
                    return

                if len(idx_to) == 0:
                    error_msg('The area "to" has no buses!')
                    return

                if len(idx_br) == 0:
                    error_msg('There are no inter-area Branches!')
                    return

                mode = self.transfer_modes_dict[self.ui.transferMethodComboBox.currentText()]

                options = sim.AvailableTransferCapacityOptions(distributed_slack=distributed_slack,
                                                               use_provided_flows=use_provided_flows,
                                                               bus_idx_from=idx_from,
                                                               bus_idx_to=idx_to,
                                                               idx_br=idx_br,
                                                               sense_br=sense_br,
                                                               Pf=Pf,
                                                               idx_hvdc_br=idx_hvdc_br,
                                                               sense_hvdc_br=sense_hvdc_br,
                                                               Pf_hvdc=Pf_hvdc,
                                                               dT=dT,
                                                               threshold=threshold,
                                                               mode=mode,
                                                               max_report_elements=max_report_elements)

                drv = sim.AvailableTransferCapacityDriver(grid=self.circuit,
                                                          options=options)

                self.session.run(drv,
                                 post_func=self.post_available_transfer_capacity,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)
                self.add_simulation(sim.SimulationTypes.NetTransferCapacity_run)
                self.LOCK()

            else:
                warning_msg('Another contingency analysis is being executed now...')

        else:
            pass

    def post_available_transfer_capacity(self):
        """
        Action performed after the short circuit.
        Returns:

        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.NetTransferCapacity_run)
        self.remove_simulation(sim.SimulationTypes.NetTransferCapacity_run)

        # update the results in the circuit structures
        # if not drv.__cancel__:
        if results is not None:

            self.ui.progress_label.setText('Colouring ATC results in the grid...')
            QtGui.QGuiApplication.processEvents()

            self.update_available_results()
            self.colour_schematic()
        else:
            error_msg('Something went wrong, There are no ATC results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_available_transfer_capacity_ts(self, use_clustering=False):
        """
        Run a Power Transfer Distribution Factors analysis
        :return:
        """
        if len(self.circuit.buses) > 0:

            if self.valid_time_series():
                if not self.session.is_this_running(sim.SimulationTypes.NetTransferCapacity_run):

                    distributed_slack = self.ui.distributed_slack_checkBox.isChecked()
                    dT = self.ui.atcPerturbanceSpinBox.value()
                    threshold = self.ui.atcThresholdSpinBox.value()
                    max_report_elements = 5  # TODO: self.ui.ntcReportLimitingElementsSpinBox.value()

                    # available transfer capacity inter areas
                    compatible_areas, lst_from, lst_to, lst_br, lst_hvdc_br, areas_from, areas_to = self.get_compatible_areas_from_to()

                    if not compatible_areas:
                        return

                    idx_from = np.array([i for i, bus in lst_from])
                    idx_to = np.array([i for i, bus in lst_to])
                    idx_br = np.array([i for i, bus, sense in lst_br])
                    sense_br = np.array([sense for i, bus, sense in lst_br])

                    # HVDC
                    idx_hvdc_br = np.array([i for i, bus, sense in lst_hvdc_br])
                    sense_hvdc_br = np.array([sense for i, bus, sense in lst_hvdc_br])

                    if self.ui.usePfValuesForAtcCheckBox.isChecked():
                        pf_drv, pf_results = self.session.get_driver_results(sim.SimulationTypes.TimeSeries_run)
                        if pf_results is not None:
                            Pf = pf_results.Sf.real
                            Pf_hvdc = pf_results.hvdc_Pf.real
                            use_provided_flows = True
                        else:
                            warning_msg('There were no power flow values available. Linear flows will be used.')
                            use_provided_flows = False
                            Pf_hvdc = None
                            Pf = None
                    else:
                        use_provided_flows = False
                        Pf_hvdc = None
                        Pf = None

                    if len(idx_from) == 0:
                        error_msg('The area "from" has no buses!')
                        return

                    if len(idx_to) == 0:
                        error_msg('The area "to" has no buses!')
                        return

                    if len(idx_br) == 0:
                        error_msg('There are no inter-area Branches!')
                        return

                    mode = self.transfer_modes_dict[self.ui.transferMethodComboBox.currentText()]
                    cluster_number = self.ui.cluster_number_spinBox.value()
                    options = sim.AvailableTransferCapacityOptions(distributed_slack=distributed_slack,
                                                                   use_provided_flows=use_provided_flows,
                                                                   bus_idx_from=idx_from,
                                                                   bus_idx_to=idx_to,
                                                                   idx_br=idx_br,
                                                                   sense_br=sense_br,
                                                                   Pf=Pf,
                                                                   idx_hvdc_br=idx_hvdc_br,
                                                                   sense_hvdc_br=sense_hvdc_br,
                                                                   Pf_hvdc=Pf_hvdc,
                                                                   dT=dT,
                                                                   threshold=threshold,
                                                                   mode=mode,
                                                                   max_report_elements=max_report_elements,
                                                                   use_clustering=use_clustering,
                                                                   cluster_number=cluster_number)

                    drv = sim.AvailableTransferCapacityTimeSeriesDriver(grid=self.circuit,
                                                                        options=options,
                                                                        time_indices=self.get_time_indices(),
                                                                        clustering_results=self.get_clustering_results())

                    self.session.run(drv,
                                     post_func=self.post_available_transfer_capacity_ts,
                                     prog_func=self.ui.progressBar.setValue,
                                     text_func=self.ui.progress_label.setText)
                    self.add_simulation(sim.SimulationTypes.NetTransferCapacityTS_run)
                    self.LOCK()

                else:
                    warning_msg('Another ATC time series is being executed now...')
            else:
                error_msg('There are no time series!')
        else:
            pass

    def run_available_transfer_capacity_clustering(self):
        """
        Run the ATC time series using clustering
        :return:
        """
        self.run_available_transfer_capacity_ts(use_clustering=True)

    def post_available_transfer_capacity_ts(self):
        """
        Action performed after the short circuit.
        Returns:

        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.NetTransferCapacityTS_run)
        self.remove_simulation(sim.SimulationTypes.NetTransferCapacityTS_run)

        # update the results in the circuit structures
        # if not drv.__cancel__:
        if results is not None:

            self.ui.progress_label.setText('Colouring ATC time series results in the grid...')
            QtGui.QGuiApplication.processEvents()

            self.update_available_results()
            self.colour_schematic()
        else:
            error_msg('Something went wrong, There are no ATC time series results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_continuation_power_flow(self):
        """
        Run voltage stability (voltage collapse) in a separated thread
        :return:
        """

        if len(self.circuit.buses) > 0:

            pf_drv, pf_results = self.session.get_driver_results(sim.SimulationTypes.PowerFlow_run)

            if pf_results is not None:

                if not self.session.is_this_running(sim.SimulationTypes.ContinuationPowerFlow_run):

                    # get the selected UI options
                    use_alpha = self.ui.start_vs_from_default_radioButton.isChecked()

                    # direction vector
                    alpha = self.ui.alpha_doubleSpinBox.value()
                    n = len(self.circuit.buses)

                    # vector that multiplies the target power: The continuation direction
                    alpha_vec = np.ones(n)

                    if self.ui.atcRadioButton.isChecked():
                        use_alpha = True
                        compatible_areas, lst_from, lst_to, lst_br, lst_hvdc_br, areas_from, areas_to = self.get_compatible_areas_from_to()

                        if compatible_areas:
                            idx_from = [i for i, bus in lst_from]
                            idx_to = [i for i, bus in lst_to]

                            alpha_vec[idx_from] *= 2
                            alpha_vec[idx_to] *= -2
                            sel_bus_idx = np.zeros(0, dtype=int)  # for completeness

                            # HVDC
                            idx_hvdc_br = np.array([i for i, bus, sense in lst_hvdc_br])
                            sense_hvdc_br = np.array([sense for i, bus, sense in lst_hvdc_br])
                        else:
                            sel_bus_idx = np.zeros(0, dtype=int)  # for completeness
                            # incompatible areas...exit
                            return
                    else:
                        sel_buses = self.get_selected_buses()
                        if len(sel_buses) == 0:
                            # all nodes
                            alpha_vec *= alpha
                            sel_bus_idx = np.zeros(0, dtype=int)  # for completeness
                        else:
                            # pick the selected nodes
                            sel_bus_idx = np.array([k for k, bus in sel_buses])
                            alpha_vec[sel_bus_idx] = alpha_vec[sel_bus_idx] * alpha

                    use_profiles = self.ui.start_vs_from_selected_radioButton.isChecked()
                    start_idx = self.ui.vs_departure_comboBox.currentIndex()
                    end_idx = self.ui.vs_target_comboBox.currentIndex()

                    if len(sel_bus_idx) > 0:
                        if sum([self.circuit.buses[i].get_device_number() for i in sel_bus_idx]) == 0:
                            warning_msg('You have selected a group of buses with no power injection.\n'
                                        'this will result in an infinite continuation, since the loading variation '
                                        'of buses with zero injection will be infinite.', 'Continuation Power Flow')
                            return

                    mode = self.ui.vc_stop_at_comboBox.currentText()

                    vc_stop_at_dict = {sim.CpfStopAt.Nose.value: sim.CpfStopAt.Nose,
                                       sim.CpfStopAt.Full.value: sim.CpfStopAt.Full,
                                       sim.CpfStopAt.ExtraOverloads.value: sim.CpfStopAt.ExtraOverloads}

                    pf_options = self.get_selected_power_flow_options()

                    # declare voltage collapse options
                    vc_options = sim.ContinuationPowerFlowOptions(step=0.0001,
                                                                  approximation_order=sim.CpfParametrization.Natural,
                                                                  adapt_step=True,
                                                                  step_min=0.00001,
                                                                  step_max=0.2,
                                                                  error_tol=1e-3,
                                                                  tol=pf_options.tolerance,
                                                                  max_it=pf_options.max_iter,
                                                                  stop_at=vc_stop_at_dict[mode],
                                                                  verbose=False)

                    if use_alpha:
                        '''
                        use the current power situation as start
                        and a linear combination of the current situation as target
                        '''
                        # lock the UI
                        self.LOCK()

                        self.ui.progress_label.setText('Compiling the grid...')
                        QtGui.QGuiApplication.processEvents()

                        #  compose the base power
                        Sbase = pf_results.Sbus / self.circuit.Sbase

                        base_overload_number = len(np.where(np.abs(pf_results.loading) > 1)[0])

                        vc_inputs = sim.ContinuationPowerFlowInput(Sbase=Sbase,
                                                                   Vbase=pf_results.voltage,
                                                                   Starget=Sbase * alpha,
                                                                   base_overload_number=base_overload_number)

                        pf_options = self.get_selected_power_flow_options()

                        # create object
                        drv = sim.ContinuationPowerFlowDriver(circuit=self.circuit,
                                                              options=vc_options,
                                                              inputs=vc_inputs,
                                                              pf_options=pf_options)
                        self.session.run(drv,
                                         post_func=self.post_continuation_power_flow,
                                         prog_func=self.ui.progressBar.setValue,
                                         text_func=self.ui.progress_label.setText)

                    elif use_profiles:
                        '''
                        Here the start and finish power states are taken from the profiles
                        '''
                        if start_idx > -1 and end_idx > -1:

                            # lock the UI
                            self.LOCK()

                            pf_drv.run_at(start_idx)

                            # get the power Injections array to get the initial and end points
                            nc = core.compile_time_circuit(circuit=self.circuit)
                            Sprof = nc.Sbus
                            vc_inputs = sim.ContinuationPowerFlowInput(Sbase=Sprof[:, start_idx],
                                                                       Vbase=pf_results.voltage,
                                                                       Starget=Sprof[:, end_idx])

                            pf_options = self.get_selected_power_flow_options()

                            # create object
                            drv = sim.ContinuationPowerFlowDriver(circuit=self.circuit,
                                                                  options=vc_options,
                                                                  inputs=vc_inputs,
                                                                  pf_options=pf_options)
                            self.session.run(drv,
                                             post_func=self.post_continuation_power_flow,
                                             prog_func=self.ui.progressBar.setValue,
                                             text_func=self.ui.progress_label.setText)
                        else:
                            info_msg('Check the selected start and finnish time series indices.')
                else:
                    warning_msg('Another voltage collapse simulation is running...')
            else:
                info_msg('Run a power flow simulation first.\n'
                         'The results are needed to initialize this simulation.')
        else:
            pass

    def post_continuation_power_flow(self):
        """
        Actions performed after the voltage stability. Launched by the thread after its execution
        :return:
        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.ContinuationPowerFlow_run)

        if results is not None:

            self.remove_simulation(sim.SimulationTypes.ContinuationPowerFlow_run)

            if results.voltages is not None:
                if results.voltages.shape[0] > 0:

                    self.update_available_results()

                    if self.ui.draw_schematic_checkBox.isChecked():
                        self.colour_schematic()
            else:
                info_msg('The voltage stability did not converge.\nIs this case already at the collapse limit?')
        else:
            error_msg('Something went wrong, There are no voltage stability results.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_power_flow_time_series(self):
        """
        Run a time series power flow simulation in a separated thread from the gui
        @return:
        """
        if len(self.circuit.buses) > 0:
            if not self.session.is_this_running(sim.SimulationTypes.TimeSeries_run):
                if self.valid_time_series():
                    self.LOCK()

                    self.add_simulation(sim.SimulationTypes.TimeSeries_run)

                    self.ui.progress_label.setText('Compiling the grid...')
                    QtGui.QGuiApplication.processEvents()

                    use_opf_vals = self.ui.actionOpf_to_Power_flow.isChecked()

                    if use_opf_vals:

                        drv, opf_time_series_results = self.session.get_driver_results(
                            SimulationTypes.OPFTimeSeries_run)

                        if opf_time_series_results is None:
                            if use_opf_vals:
                                info_msg('There are no OPF time series, '
                                         'therefore this operation will not use OPF information.')
                                self.ui.actionOpf_to_Power_flow.setChecked(False)

                    else:
                        opf_time_series_results = None

                    options = self.get_selected_power_flow_options()

                    drv = sim.PowerFlowTimeSeriesDriver(grid=self.circuit,
                                                        options=options,
                                                        time_indices=self.get_time_indices(),
                                                        opf_time_series_results=opf_time_series_results,
                                                        clustering_results=self.get_clustering_results(),
                                                        engine=self.get_preferred_engine())

                    self.session.run(drv,
                                     post_func=self.post_power_flow_time_series,
                                     prog_func=self.ui.progressBar.setValue,
                                     text_func=self.ui.progress_label.setText)

                else:
                    warning_msg('There are no time series.', 'Time series')
            else:
                warning_msg('Another time series power flow is being executed now...')
        else:
            pass

    def post_power_flow_time_series(self):
        """
        Events to do when the time series simulation has finished
        @return:
        """

        drv, results = self.session.get_driver_results(sim.SimulationTypes.TimeSeries_run)

        if results is not None:

            self.remove_simulation(sim.SimulationTypes.TimeSeries_run)

            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()

        else:
            warning_msg('No results for the time series simulation.')

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_stochastic(self):
        """
        Run a Monte Carlo simulation
        @return:
        """

        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.MonteCarlo_run):

                if self.circuit.time_profile is not None:

                    self.LOCK()

                    self.add_simulation(sim.SimulationTypes.StochasticPowerFlow)

                    self.ui.progress_label.setText('Compiling the grid...')
                    QtGui.QGuiApplication.processEvents()

                    pf_options = self.get_selected_power_flow_options()

                    simulation_type = self.stochastic_pf_methods_dict[
                        self.ui.stochastic_pf_method_comboBox.currentText()]

                    tol = 10 ** (-1 * self.ui.tolerance_stochastic_spinBox.value())
                    max_iter = self.ui.max_iterations_stochastic_spinBox.value()
                    drv = sim.StochasticPowerFlowDriver(self.circuit,
                                                        pf_options,
                                                        mc_tol=tol,
                                                        batch_size=100,
                                                        sampling_points=max_iter,
                                                        simulation_type=simulation_type)
                    self.session.run(drv,
                                     post_func=self.post_stochastic,
                                     prog_func=self.ui.progressBar.setValue,
                                     text_func=self.ui.progress_label.setText)
                else:
                    warning_msg('There are no time series.')

            else:
                warning_msg('Another Monte Carlo simulation is running...')

        else:
            pass

    def post_stochastic(self):
        """
        Actions to perform after the Monte Carlo simulation is finished
        @return:
        """

        drv, results = self.session.get_driver_results(sim.SimulationTypes.StochasticPowerFlow)

        if results is not None:

            self.remove_simulation(sim.SimulationTypes.StochasticPowerFlow)

            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()


        else:
            pass

        if not self.session.is_anything_running():
            self.UNLOCK()

    def clear_cascade(self):
        """
        Clear cascade simulation
        """
        # self.cascade = None
        self.ui.cascade_tableView.setModel(None)

    def run_cascade_step(self):
        """
        Run cascade step
        """
        if len(self.circuit.buses) > 0:

            self.LOCK()
            if self.session.exists(sim.SimulationTypes.Cascade_run):
                options = self.get_selected_power_flow_options()
                options.solver_type = bs.SolverType.LM
                max_isl = self.ui.cascading_islands_spinBox.value()
                drv = sim.Cascading(self.circuit.copy(), options, max_additional_islands=max_isl)

                self.session.run(drv,
                                 post_func=self.post_cascade,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            self.cascade.perform_step_run()

            self.post_cascade()

            self.UNLOCK()

    def run_cascade(self):
        """
        Run a cascading to blackout simulation
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.Cascade_run):

                self.add_simulation(sim.SimulationTypes.Cascade_run)

                self.LOCK()

                self.ui.progress_label.setText('Compiling the grid...')
                QtGui.QGuiApplication.processEvents()

                options = self.get_selected_power_flow_options()
                options.solver_type = bs.SolverType.LM

                max_isl = self.ui.cascading_islands_spinBox.value()
                n_lsh_samples = self.ui.max_iterations_stochastic_spinBox.value()

                drv = sim.Cascading(self.circuit.copy(), options,
                                    max_additional_islands=max_isl,
                                    n_lhs_samples_=n_lsh_samples)

                self.session.run(drv,
                                 post_func=self.post_cascade,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

                # run
                drv.start()

            else:
                warning_msg('Another cascade is running...')
        else:
            pass

    def post_cascade(self, idx=None):
        """
        Actions to perform after the cascade simulation is finished
        """

        # update the results in the circuit structures
        drv, results = self.session.get_driver_results(sim.SimulationTypes.Cascade_run)

        self.remove_simulation(sim.SimulationTypes.Cascade_run)

        n = len(results.events)

        if n > 0:

            # display the last event, if none is selected
            if idx is None:
                idx = n - 1

            # Accumulate all the failed Branches
            br_idx = np.zeros(0, dtype=int)
            for i in range(idx):
                br_idx = np.r_[br_idx, results.events[i].removed_idx]

            # pick the results at the designated cascade step
            results = results.events[idx].pf_results  # StochasticPowerFlowResults object

            # Update results
            self.update_available_results()

            # print grid
            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()

            # Set cascade table
            self.ui.cascade_tableView.setModel(gf.PandasModel(drv.get_table()))

        if not self.session.is_anything_running():
            self.UNLOCK()

    def cascade_table_click(self):
        """
        Display cascade upon cascade scenario click
        Returns:

        """

        idx = self.ui.cascade_tableView.currentIndex()
        if idx.row() > -1:
            self.post_cascade(idx=idx.row())

    def run_opf(self):
        """
        Run OPF simulation
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.OPF_run):

                self.remove_simulation(sim.SimulationTypes.OPF_run)

                # get the power flow options from the GUI
                solver = self.lp_solvers_dict[self.ui.lpf_solver_comboBox.currentText()]
                mip_solver = self.mip_solvers_dict[self.ui.mip_solver_comboBox.currentText()]
                time_grouping = self.opf_time_groups[self.ui.opf_time_grouping_comboBox.currentText()]
                zonal_grouping = self.opf_zonal_groups[self.ui.opfZonalGroupByComboBox.currentText()]
                pf_options = self.get_selected_power_flow_options()
                consider_contingencies = self.ui.considerContingenciesOpfCheckBox.isChecked()
                skip_generation_limits = self.ui.skipOpfGenerationLimitsCheckBox.isChecked()
                lodf_tolerance = self.ui.opfContingencyToleranceSpinBox.value()
                maximize_flows = self.ui.opfMaximizeExcahngeCheckBox.isChecked()
                unit_commitment = self.ui.opfUnitCommitmentCheckBox.isChecked()

                # available transfer capacity inter areas
                if maximize_flows:
                    compatible_areas, lst_from, lst_to, lst_br, lst_hvdc_br, areas_from, areas_to = self.get_compatible_areas_from_to()
                    idx_from = np.array([i for i, bus in lst_from])
                    idx_to = np.array([i for i, bus in lst_to])

                    if len(idx_from) == 0:
                        error_msg('The area "from" has no buses!')
                        return

                    if len(idx_to) == 0:
                        error_msg('The area "to" has no buses!')
                        return
                else:
                    idx_from = None
                    idx_to = None
                    areas_from = None
                    areas_to = None

                # try to acquire the linear results
                linear_results = self.session.linear_power_flow
                if linear_results is not None:
                    LODF = linear_results.LODF
                else:
                    LODF = None
                    if consider_contingencies and self.get_preferred_engine() == bs.EngineType.GridCal:
                        warning_msg("To consider contingencies, the LODF matrix is required.\n"
                                    "Run a linear simulation first", "OPF time series")
                        return

                options = sim.OptimalPowerFlowOptions(solver=solver,
                                                      time_grouping=time_grouping,
                                                      zonal_grouping=zonal_grouping,
                                                      mip_solver=mip_solver,
                                                      power_flow_options=pf_options,
                                                      consider_contingencies=consider_contingencies,
                                                      skip_generation_limits=skip_generation_limits,
                                                      LODF=LODF,
                                                      lodf_tolerance=lodf_tolerance,
                                                      maximize_flows=maximize_flows,
                                                      area_from_bus_idx=idx_from,
                                                      area_to_bus_idx=idx_to,
                                                      areas_from=areas_from,
                                                      areas_to=areas_to,
                                                      unit_commitment=unit_commitment)

                options.max_vm = self.ui.maxVoltageModuleStepSpinBox.value()
                options.max_va = self.ui.maxVoltageAngleStepSpinBox.value()

                self.ui.progress_label.setText('Running optimal power flow...')
                QtGui.QGuiApplication.processEvents()
                pf_options = self.get_selected_power_flow_options()

                self.LOCK()

                # set power flow object instance
                drv = sim.OptimalPowerFlowDriver(grid=self.circuit,
                                                 options=options,
                                                 engine=self.get_preferred_engine())

                self.session.run(drv,
                                 post_func=self.post_opf,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            else:
                warning_msg('Another OPF is being run...')
        else:
            pass

    def post_opf(self):
        """
        Actions to run after the OPF simulation
        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.OPF_run)

        if results is not None:

            self.remove_simulation(sim.SimulationTypes.OPF_run)

            if not results.converged:
                warning_msg('Some islands did not solve.\n'
                            'Check that all Branches have rating and \n'
                            'that the generator bounds are ok.\n'
                            'You may also use the diagnostic tool (F8)', 'OPF')

            self.update_available_results()
            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_opf_time_series(self):
        """
        OPF Time Series run
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.OPFTimeSeries_run):

                if self.circuit.time_profile is not None:

                    self.add_simulation(sim.SimulationTypes.OPFTimeSeries_run)

                    self.LOCK()

                    # Compile the grid
                    self.ui.progress_label.setText('Compiling the grid...')
                    QtGui.QGuiApplication.processEvents()

                    # get the power flow options from the GUI
                    solver = self.lp_solvers_dict[self.ui.lpf_solver_comboBox.currentText()]
                    mip_solver = self.mip_solvers_dict[self.ui.mip_solver_comboBox.currentText()]
                    time_grouping = self.opf_time_groups[self.ui.opf_time_grouping_comboBox.currentText()]
                    zonal_grouping = self.opf_zonal_groups[self.ui.opfZonalGroupByComboBox.currentText()]
                    pf_options = self.get_selected_power_flow_options()
                    consider_contingencies = self.ui.considerContingenciesOpfCheckBox.isChecked()
                    skip_generation_limits = self.ui.skipOpfGenerationLimitsCheckBox.isChecked()

                    lodf_tolerance = self.ui.opfContingencyToleranceSpinBox.value()
                    maximize_flows = self.ui.opfMaximizeExcahngeCheckBox.isChecked()
                    unit_commitment = self.ui.opfUnitCommitmentCheckBox.isChecked()

                    # available transfer capacity inter areas
                    if maximize_flows:
                        compatible_areas, lst_from, lst_to, lst_br, lst_hvdc_br, areas_from, areas_to = self.get_compatible_areas_from_to()
                        idx_from = np.array([i for i, bus in lst_from])
                        idx_to = np.array([i for i, bus in lst_to])

                        if len(idx_from) == 0:
                            error_msg('The area "from" has no buses!')
                            return

                        if len(idx_to) == 0:
                            error_msg('The area "to" has no buses!')
                            return
                    else:
                        idx_from = None
                        idx_to = None
                        areas_from = None
                        areas_to = None

                    # try to acquire the linear results
                    linear_results = self.session.linear_power_flow
                    if linear_results is not None:
                        LODF = linear_results.LODF
                    else:
                        LODF = None
                        if consider_contingencies and self.get_preferred_engine() == bs.EngineType.GridCal:
                            warning_msg("To consider contingencies, the LODF matrix is required.\n"
                                        "Run a linear simulation first", "OPF time series")
                            return

                    options = sim.OptimalPowerFlowOptions(solver=solver,
                                                          time_grouping=time_grouping,
                                                          zonal_grouping=zonal_grouping,
                                                          mip_solver=mip_solver,
                                                          power_flow_options=pf_options,
                                                          consider_contingencies=consider_contingencies,
                                                          skip_generation_limits=skip_generation_limits,
                                                          LODF=LODF,
                                                          lodf_tolerance=lodf_tolerance,
                                                          maximize_flows=maximize_flows,
                                                          area_from_bus_idx=idx_from,
                                                          area_to_bus_idx=idx_to,
                                                          areas_from=areas_from,
                                                          areas_to=areas_to,
                                                          unit_commitment=unit_commitment
                                                          )

                    # create the OPF time series instance
                    # if non_sequential:
                    drv = sim.OptimalPowerFlowTimeSeriesDriver(grid=self.circuit,
                                                               options=options,
                                                               time_indices=self.get_time_indices(),
                                                               clustering_results=self.get_clustering_results())

                    drv.engine = self.get_preferred_engine()

                    self.session.run(drv,
                                     post_func=self.post_opf_time_series,
                                     prog_func=self.ui.progressBar.setValue,
                                     text_func=self.ui.progress_label.setText)

                else:
                    warning_msg('There are no time series.\nLoad time series are needed for this simulation.')

            else:
                warning_msg('Another OPF time series is running already...')

        else:
            pass

    def post_opf_time_series(self):
        """
        Post OPF Time Series
        """

        drv, results = self.session.get_driver_results(sim.SimulationTypes.OPFTimeSeries_run)

        if results is not None:

            if len(drv.logger) > 0:
                dlg = LogsDialogue('logger', drv.logger)
                dlg.exec_()

            # remove from the current simulations
            self.remove_simulation(sim.SimulationTypes.OPFTimeSeries_run)

            if results is not None:

                self.update_available_results()

                if self.ui.draw_schematic_checkBox.isChecked():
                    self.colour_schematic()

                msg = 'OPF time series elapsed ' + str(drv.elapsed) + ' s'
                self.console_msg(msg)

        else:
            pass

        if not self.session.is_anything_running():
            self.UNLOCK()

    def copy_opf_to_time_series(self):
        """
        Copy the OPF generation values to the Time series object and execute a time series simulation
        """
        if len(self.circuit.buses) > 0:

            if self.circuit.time_profile is not None:

                drv, results = self.session.get_driver_results(sim.SimulationTypes.OPFTimeSeries_run)

                if results is not None:

                    quit_msg = "Are you sure that you want overwrite the time events " \
                               "with the simulated by the OPF time series?"
                    reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                                           QtWidgets.QMessageBox.Yes,
                                                           QtWidgets.QMessageBox.No)

                    if reply == QtWidgets.QMessageBox.Yes:

                        self.circuit.apply_lp_profiles(results)

                    else:
                        pass

                else:
                    info_msg('There are no OPF time series execution.'
                             '\nRun OPF time series to be able to copy the value to the time series object.')

            else:
                warning_msg('There are no time series.\nLoad time series are needed for this simulation.')
        else:
            pass

    def default_options_opf_ntc_optimal(self):
        """
        Set the default options for the NTC optimization in the optimal setting
        :return:
        """
        self.ui.skipNtcGenerationLimitsCheckBox.setChecked(False)
        self.ui.considerContingenciesNtcOpfCheckBox.setChecked(True)
        self.ui.ntcDispatchAllAreasCheckBox.setChecked(False)
        self.ui.ntcFeasibilityCheckCheckBox.setChecked(False)
        self.ui.weightPowerShiftSpinBox.setValue(0)
        self.ui.weightGenCostSpinBox.setValue(0)
        self.ui.weightsOverloadsSpinBox.setValue(0)

    def default_options_opf_ntc_proportional(self):
        """
        Set the default options for the NTC optimization in the proportional setting
        :return:
        """
        self.ui.skipNtcGenerationLimitsCheckBox.setChecked(True)
        self.ui.considerContingenciesNtcOpfCheckBox.setChecked(True)
        self.ui.ntcDispatchAllAreasCheckBox.setChecked(False)
        self.ui.ntcFeasibilityCheckCheckBox.setChecked(False)
        self.ui.weightPowerShiftSpinBox.setValue(5)
        self.ui.weightGenCostSpinBox.setValue(2)
        self.ui.weightsOverloadsSpinBox.setValue(3)

    def run_opf_ntc(self):
        """
        Run OPF simulation
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.OPF_NTC_run):

                self.remove_simulation(sim.SimulationTypes.OPF_NTC_run)

                # available transfer capacity inter areas
                compatible_areas, lst_from, lst_to, lst_br, lst_hvdc_br, areas_from, areas_to = self.get_compatible_areas_from_to()

                if not compatible_areas:
                    return

                idx_from = np.array([i for i, bus in lst_from])
                idx_to = np.array([i for i, bus in lst_to])
                idx_br = np.array([i for i, bus, sense in lst_br])

                if len(idx_from) == 0:
                    error_msg('The area "from" has no buses!')
                    return

                if len(idx_to) == 0:
                    error_msg('The area "to" has no buses!')
                    return

                if len(idx_br) == 0:
                    error_msg('There are no inter-area Branches!')
                    return

                mip_solver = self.mip_solvers_dict[self.ui.mip_solver_comboBox.currentText()]

                if self.ui.optimalRedispatchRadioButton.isChecked():
                    generation_formulation = dev.GenerationNtcFormulation.Optimal
                    # perform_previous_checks = False
                elif self.ui.proportionalRedispatchRadioButton.isChecked():
                    generation_formulation = dev.GenerationNtcFormulation.Proportional
                    # perform_previous_checks = True
                else:
                    generation_formulation = dev.GenerationNtcFormulation.Optimal
                    # perform_previous_checks = False

                monitor_only_sensitive_branches = self.ui.ntcSelectBasedOnExchangeSensitivityCheckBox.isChecked()
                monitor_only_ntc_rule_branches = self.ui.ntcSelectBasedOnAcerCriteriaCheckBox.isChecked()
                skip_generation_limits = self.ui.skipNtcGenerationLimitsCheckBox.isChecked()
                branch_sensitivity_threshold = self.ui.ntcAlphaSpinBox.value() / 100.0
                dT = self.ui.atcPerturbanceSpinBox.value()
                mode = self.transfer_modes_dict[self.ui.transferMethodComboBox.currentText()]
                tolerance = 10.0 ** self.ui.ntcOpfTolSpinBox.value()

                perform_previous_checks = self.ui.ntcFeasibilityCheckCheckBox.isChecked()

                dispatch_all_areas = self.ui.ntcDispatchAllAreasCheckBox.isChecked()

                weight_power_shift = 10.0 ** self.ui.weightPowerShiftSpinBox.value()
                weight_generation_cost = 10.0 ** self.ui.weightGenCostSpinBox.value()

                # todo: add consider_nx_contingencies to gui if necessary
                consider_contingencies = self.ui.considerContingenciesNtcOpfCheckBox.isChecked()
                consider_nx_contingencies = self.ui.considerContingenciesNtcOpfCheckBox.isChecked()
                consider_hvdc_contingencies = self.ui.considerContingenciesHvdcOpfCheckBox.isChecked()
                consider_gen_contingencies = self.ui.considerContingenciesGeneratorOpfCheckBox.isChecked()
                generation_contingency_threshold = self.ui.contingencyGenerationThresholdDoubleSpinBox.value()

                trm = self.ui.trmSpinBox.value()
                ntc_load_rule = self.ui.ntcLoadRuleSpinBox.value() / 100.0
                loading_threshold_to_report = self.ui.ntcReportLoadingThresholdSpinBox.value()
                n1_consideration = self.ui.n1ConsiderationCheckBox.isChecked()

                options = sim.OptimalNetTransferCapacityOptions(
                    area_from_bus_idx=idx_from,
                    area_to_bus_idx=idx_to,
                    mip_solver=mip_solver,
                    generation_formulation=generation_formulation,
                    monitor_only_sensitive_branches=monitor_only_sensitive_branches,
                    monitor_only_ntc_rule_branches=monitor_only_ntc_rule_branches,
                    branch_sensitivity_threshold=branch_sensitivity_threshold,
                    skip_generation_limits=skip_generation_limits,
                    dispatch_all_areas=dispatch_all_areas,
                    tolerance=tolerance,
                    sensitivity_dT=dT,
                    transfer_method=mode,
                    perform_previous_checks=perform_previous_checks,
                    weight_power_shift=weight_power_shift,
                    weight_generation_cost=weight_generation_cost,
                    consider_contingencies=consider_contingencies,
                    consider_hvdc_contingencies=consider_hvdc_contingencies,
                    consider_gen_contingencies=consider_gen_contingencies,
                    consider_nx_contingencies=consider_nx_contingencies,
                    generation_contingency_threshold=generation_contingency_threshold,
                    loading_threshold_to_report=loading_threshold_to_report,
                    trm=trm,
                    ntc_load_rule=ntc_load_rule,
                    n1_consideration=n1_consideration,
                )

                self.ui.progress_label.setText('Running optimal net transfer capacity...')
                QtGui.QGuiApplication.processEvents()
                pf_options = self.get_selected_power_flow_options()

                # set power flow object instance
                drv = sim.OptimalNetTransferCapacityDriver(
                    grid=self.circuit,
                    options=options,
                    pf_options=pf_options)

                self.LOCK()
                self.session.run(drv,
                                 post_func=self.post_opf_ntc,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            else:
                warning_msg('Another OPF is being run...')
        else:
            pass

    def post_opf_ntc(self):
        """
        Actions to run after the OPF simulation
        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.OPF_NTC_run)

        if results is not None:

            self.remove_simulation(sim.SimulationTypes.OPF_NTC_run)
            self.update_available_results()

            if self.ui.draw_schematic_checkBox.isChecked():
                self.colour_schematic()

        if drv.logger is not None:
            if len(drv.logger) > 0:
                dlg = LogsDialogue(drv.name, drv.logger)
                dlg.setModal(True)
                dlg.exec_()

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_opf_ntc_ts(self, with_clustering=False):
        """
        Run OPF time series simulation
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.OPF_NTC_TS_run):

                self.remove_simulation(sim.SimulationTypes.OPF_NTC_TS_run)

                # available transfer capacity inter areas
                compatible_areas, lst_from, lst_to, lst_br, lst_hvdc_br, areas_from, areas_to = self.get_compatible_areas_from_to()

                if not compatible_areas:
                    return

                idx_from = np.array([i for i, bus in lst_from])
                idx_to = np.array([i for i, bus in lst_to])
                idx_br = np.array([i for i, bus, sense in lst_br])

                if len(idx_from) == 0:
                    error_msg('The area "from" has no buses!')
                    return

                if len(idx_to) == 0:
                    error_msg('The area "to" has no buses!')
                    return

                if len(idx_br) == 0:
                    error_msg('There are no inter-area Branches!')
                    return

                mip_solver = self.mip_solvers_dict[self.ui.mip_solver_comboBox.currentText()]

                if self.ui.optimalRedispatchRadioButton.isChecked():
                    generation_formulation = dev.GenerationNtcFormulation.Optimal
                elif self.ui.proportionalRedispatchRadioButton.isChecked():
                    generation_formulation = dev.GenerationNtcFormulation.Proportional
                else:
                    generation_formulation = dev.GenerationNtcFormulation.Optimal

                monitor_only_sensitive_branches = self.ui.ntcSelectBasedOnExchangeSensitivityCheckBox.isChecked()
                monitor_only_ntc_rule_branches = self.ui.ntcSelectBasedOnAcerCriteriaCheckBox.isChecked()
                skip_generation_limits = self.ui.skipNtcGenerationLimitsCheckBox.isChecked()
                branch_sensitivity_threshold = self.ui.atcThresholdSpinBox.value()
                dT = self.ui.atcPerturbanceSpinBox.value()
                mode = self.transfer_modes_dict[self.ui.transferMethodComboBox.currentText()]
                tolerance = 10.0 ** self.ui.ntcOpfTolSpinBox.value()

                perform_previous_checks = self.ui.ntcFeasibilityCheckCheckBox.isChecked()

                dispatch_all_areas = self.ui.ntcDispatchAllAreasCheckBox.isChecked()

                weight_power_shift = 10.0 ** self.ui.weightPowerShiftSpinBox.value()
                weight_generation_cost = 10.0 ** self.ui.weightGenCostSpinBox.value()

                # todo: add consider_nx_contingencies to gui if necessary
                consider_contingencies = self.ui.considerContingenciesNtcOpfCheckBox.isChecked()
                consider_nx_contingencies = self.ui.considerContingenciesNtcOpfCheckBox.isChecked()
                consider_hvdc_contingencies = self.ui.considerContingenciesHvdcOpfCheckBox.isChecked()
                consider_gen_contingencies = self.ui.considerContingenciesGeneratorOpfCheckBox.isChecked()
                generation_contingency_threshold = self.ui.contingencyGenerationThresholdDoubleSpinBox.value()

                trm = self.ui.trmSpinBox.value()
                loading_threshold_to_report = self.ui.ntcReportLoadingThresholdSpinBox.value()
                ntcLoadRule = self.ui.ntcLoadRuleSpinBox.value() / 100
                n1Consideration = self.ui.n1ConsiderationCheckBox.isChecked()

                options = sim.OptimalNetTransferCapacityOptions(
                    area_from_bus_idx=idx_from,
                    area_to_bus_idx=idx_to,
                    mip_solver=mip_solver,
                    generation_formulation=generation_formulation,
                    monitor_only_sensitive_branches=monitor_only_sensitive_branches,
                    monitor_only_ntc_rule_branches=monitor_only_ntc_rule_branches,
                    branch_sensitivity_threshold=branch_sensitivity_threshold,
                    skip_generation_limits=skip_generation_limits,
                    dispatch_all_areas=dispatch_all_areas,
                    tolerance=tolerance,
                    sensitivity_dT=dT,
                    transfer_method=mode,
                    perform_previous_checks=perform_previous_checks,
                    weight_power_shift=weight_power_shift,
                    weight_generation_cost=weight_generation_cost,
                    consider_contingencies=consider_contingencies,
                    consider_hvdc_contingencies=consider_hvdc_contingencies,
                    consider_gen_contingencies=consider_gen_contingencies,
                    consider_nx_contingencies=consider_nx_contingencies,
                    generation_contingency_threshold=generation_contingency_threshold,
                    trm=trm,
                    loading_threshold_to_report=loading_threshold_to_report,
                    ntc_load_rule=ntcLoadRule,
                    n1_consideration=n1Consideration)

                self.ui.progress_label.setText('Running optimal net transfer capacity time series...')
                QtGui.QGuiApplication.processEvents()

                # set optimal net transfer capacity driver instance
                drv = sim.OptimalNetTransferCapacityTimeSeriesDriver(grid=self.circuit,
                                                                     options=options,
                                                                     time_indices=self.get_time_indices(),
                                                                     clustering_results=self.get_clustering_results())

                self.LOCK()
                self.session.run(drv,
                                 post_func=self.post_opf_ntc_ts,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            else:
                warning_msg('Another Optimal NCT time series is being run...')
        else:
            pass

    def post_opf_ntc_ts(self):
        """
        Actions to run after the optimal net transfer capacity time series simulation
        """

        drv, results = self.session.get_driver_results(sim.SimulationTypes.OPF_NTC_TS_run)

        if results is not None:

            if len(drv.logger) > 0:
                dlg = LogsDialogue('logger', drv.logger)
                dlg.exec_()

            # remove from the current simulations
            self.remove_simulation(sim.SimulationTypes.OPF_NTC_TS_run)

            if results is not None:
                self.update_available_results()

                self.colour_schematic()

                msg = 'Optimal NTC time series elapsed ' + str(drv.elapsed) + ' s'
                self.console_msg(msg)

        else:
            pass

        if not self.session.is_anything_running():
            self.UNLOCK()

    def reduce_grid(self):
        """
        Reduce grid by removing Branches and nodes according to the selected options
        """

        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.TopologyReduction_run):

                # compute the options
                rx_criteria = self.ui.rxThresholdCheckBox.isChecked()
                exponent = self.ui.rxThresholdSpinBox.value()
                rx_threshold = 1.0 / (10.0 ** exponent)

                # get the selected indices
                checked = gf.get_checked_indices(self.ui.removeByTypeListView.model())

                if len(checked) > 0:

                    selected_types = list()
                    for i in checked:
                        selected_type_txt = self.ui.removeByTypeListView.model().item(i).text()
                        selected_type = dev.BranchType(selected_type_txt)
                        selected_types.append(selected_type)

                    # compose options
                    options = sim.TopologyReductionOptions(rx_criteria=rx_criteria,
                                                           rx_threshold=rx_threshold,
                                                           selected_types=selected_types)

                    # find which Branches to remove
                    br_to_remove = sim.select_branches_to_reduce(circuit=self.circuit,
                                                                 rx_criteria=options.rx_criteria,
                                                                 rx_threshold=options.rx_threshold,
                                                                 selected_types=options.selected_type)
                    if len(br_to_remove) > 0:
                        # raise dialogue
                        branches = self.circuit.get_branches()
                        elms = [branches[i] for i in br_to_remove]
                        diag = ElementsDialogue('Elements to be reduced', elms)
                        diag.show()
                        diag.exec_()

                        if diag.accepted:

                            self.LOCK()

                            self.add_simulation(sim.SimulationTypes.TopologyReduction_run)

                            # reduce the grid
                            self.topology_reduction = sim.TopologyReduction(grid=self.circuit,
                                                                            branch_indices=br_to_remove)

                            # Set the time series run options
                            self.topology_reduction.progress_signal.connect(self.ui.progressBar.setValue)
                            self.topology_reduction.progress_text.connect(self.ui.progress_label.setText)
                            self.topology_reduction.done_signal.connect(self.post_reduce_grid)

                            self.topology_reduction.start()
                        else:
                            pass
                    else:
                        info_msg('There were no Branches identified', 'Topological grid reduction')
                else:
                    warning_msg('Select at least one reduction option in the topology settings',
                                'Topological grid reduction')
            else:
                warning_msg('Another topological reduction is being conducted...', 'Topological grid reduction')
        else:
            pass

    def post_reduce_grid(self):
        """
        Actions after reducing
        """

        self.remove_simulation(sim.SimulationTypes.TopologyReduction_run)

        self.create_schematic_from_api(explode_factor=1)

        self.clear_results()

        if not self.session.is_anything_running():
            self.UNLOCK()

    def run_find_node_groups(self):
        """
        Run the node groups algorithm
        """
        if self.ui.actionFind_node_groups.isChecked():

            drv, ptdf_results = self.session.get_driver_results(sim.SimulationTypes.LinearAnalysis_run)

            if ptdf_results is not None:

                self.LOCK()
                sigmas = self.ui.node_distances_sigma_doubleSpinBox.value()
                min_group_size = self.ui.node_distances_elements_spinBox.value()
                drv = sim.NodeGroupsDriver(grid=self.circuit,
                                           sigmas=sigmas,
                                           min_group_size=min_group_size,
                                           ptdf_results=ptdf_results)

                self.session.run(drv,
                                 post_func=self.post_run_find_node_groups,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            else:
                error_msg('There are no PTDF results :/')

        else:
            # delete the markers
            for bus in self.circuit.buses:
                if bus.graphic_obj is not None:
                    bus.graphic_obj.delete_big_marker()

    def post_run_find_node_groups(self):
        """
        Colour the grid after running the node grouping
        :return:
        """
        self.UNLOCK()
        print('\nGroups:')

        drv, results = self.session.get_driver_results(sim.SimulationTypes.NodeGrouping_run)

        if drv is not None:

            for group in drv.groups_by_name:
                print(group)

            colours = viz.get_n_colours(n=len(drv.groups_by_index))

            for c, group in enumerate(drv.groups_by_index):
                for i in group:
                    bus = self.circuit.buses[i]
                    if bus.active:
                        if bus.graphic_obj is not None:
                            r, g, b, a = colours[c]
                            color = QtGui.QColor(r * 255, g * 255, b * 255, a * 255)
                            bus.graphic_obj.add_big_marker(color=color, tool_tip_text='Group ' + str(c))

    def run_inputs_analysis(self):
        """

        :return:
        """
        if len(self.circuit.buses) > 0:

            if not self.session.is_this_running(sim.SimulationTypes.InputsAnalysis_run):

                self.remove_simulation(sim.SimulationTypes.InputsAnalysis_run)

                # set power flow object instance
                drv = sim.InputsAnalysisDriver(self.circuit)

                self.LOCK()
                self.session.run(drv,
                                 post_func=self.post_inputs_analysis,
                                 prog_func=self.ui.progressBar.setValue,
                                 text_func=self.ui.progress_label.setText)

            else:
                warning_msg('Another inputs analysis is being run...')
        else:
            pass

    def post_inputs_analysis(self):
        """

        :return:
        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.InputsAnalysis_run)

        if results is not None:
            self.remove_simulation(sim.SimulationTypes.InputsAnalysis_run)
            self.update_available_results()
            self.colour_schematic()

        if len(drv.logger) > 0:
            dlg = LogsDialogue(drv.name, drv.logger)
            dlg.exec_()

        if not self.session.is_anything_running():
            self.UNLOCK()

    def storage_location(self):
        """
        Add storage markers to the schematic
        """

        if len(self.circuit.buses) > 0:

            if self.ui.actionStorage_location_suggestion.isChecked():

                ts_drv, ts_results = self.session.get_driver_results(sim.SimulationTypes.TimeSeries_run)

                if ts_results is not None:

                    # get the numerical object of the circuit
                    numeric_circuit = core.compile_time_circuit(self.circuit)

                    # perform a time series analysis
                    ts_analysis = grid_analysis.TimeSeriesResultsAnalysis(numeric_circuit, ts_results)

                    # get the indices of the buses selected for storage
                    idx = np.where(ts_analysis.buses_selected_for_storage_frequency > 0)[0]

                    if len(idx) > 0:

                        frequencies = ts_analysis.buses_selected_for_storage_frequency[idx]

                        fmax = np.max(frequencies)

                        # prepare the color map
                        seq = [(0, 'green'),
                               (0.6, 'orange'),
                               (1.0, 'red')]
                        cmap = LinearSegmentedColormap.from_list('vcolors', seq)

                        self.buses_for_storage = list()

                        for i, freq in zip(idx, frequencies):

                            bus = self.circuit.buses[i]
                            self.buses_for_storage.append(bus)

                            # add a marker to the bus if there are no batteries in it
                            if bus.graphic_obj.big_marker is None and len(bus.batteries) == 0:
                                r, g, b, a = cmap(freq / fmax)
                                color = QtGui.QColor(r * 255, g * 255, b * 255, a * 255)
                                bus.graphic_obj.add_big_marker(color=color)
                    else:

                        info_msg('No problems were detected, therefore no storage is suggested',
                                 'Storage location')

                else:
                    warning_msg('There is no time series simulation.\n It is needed for this functionality.',
                                'Storage location')

            else:

                # delete the red dots
                if self.buses_for_storage is not None:

                    for bus in self.buses_for_storage:
                        # add a marker to the bus...
                        if bus.graphic_obj.big_marker is not None:
                            bus.graphic_obj.delete_big_marker()
                else:
                    pass
        else:
            pass

    def run_sigma_analysis(self):
        """
        Run the sigma analysis
        """
        if len(self.circuit.buses) > 0:
            options = self.get_selected_power_flow_options()
            bus_names = np.array([b.name for b in self.circuit.buses])
            sigma_driver = sim.SigmaAnalysisDriver(grid=self.circuit, options=options)
            sigma_driver.run()

            if not sigma_driver.results.converged:
                error_msg("Sigma coefficients did not converge :(")

            self.sigma_dialogue = SigmaAnalysisGUI(parent=self,
                                                   results=sigma_driver.results,
                                                   bus_names=bus_names,
                                                   good_coefficients=sigma_driver.results.converged)
            self.sigma_dialogue.resize(int(1.61 * 600.0), 550)  # golden ratio
            self.sigma_dialogue.show()  # exec leaves the parent on hold

    def grid_generator(self):
        """
        Open the grid generator window
        """
        self.grid_generator_dialogue = GridGeneratorGUI(parent=self)
        self.grid_generator_dialogue.resize(int(1.61 * 600.0), 550)  # golden ratio
        self.grid_generator_dialogue.exec_()

        if self.grid_generator_dialogue.applied:

            if len(self.circuit.buses) > 0:
                reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                       'Are you sure that you want to delete the current grid and replace it?',
                                                       QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

                if reply == QtWidgets.QMessageBox.No:
                    return

            self.circuit = self.grid_generator_dialogue.circuit

            # create schematic
            self.create_schematic_from_api(explode_factor=1)

            # set circuit name
            self.grid_editor.name_label.setText("Random grid " + str(len(self.circuit.buses)) + ' buses')

            # set base magnitudes
            self.ui.sbase_doubleSpinBox.setValue(self.circuit.Sbase)
            self.ui.fbase_doubleSpinBox.setValue(self.circuit.fBase)
            self.ui.model_version_label.setText('Model v. ' + str(self.circuit.model_version))

            # set circuit comments
            self.ui.comments_textEdit.setText("Grid generated randomly using the RPGM algorithm.")

            # update the drop down menus that display dates
            self.update_date_dependent_combos()
            self.update_area_combos()

            # clear the results
            self.clear_results()

    def import_bus_coordinates(self):
        """

        :return:
        """
        self.coordinates_window = CoordinatesInputGUI(self, self.circuit.buses)
        self.coordinates_window.exec_()

        self.create_schematic_from_api()

    def set_selected_bus_property(self, prop):
        """

        :param prop:
        :return:
        """
        if prop == 'area':
            self.object_select_window = ObjectSelectWindow('Area', self.circuit.areas, parent=self)
            self.object_select_window.setModal(True)
            self.object_select_window.exec_()

            if self.object_select_window.selected_object is not None:
                for k, bus in self.get_selected_buses():
                    bus.area = self.object_select_window.selected_object
                    print('Set {0} into bus {1}'.format(self.object_select_window.selected_object.name, bus.name))

        elif prop == 'country':
            self.object_select_window = ObjectSelectWindow('country', self.circuit.countries, parent=self)
            self.object_select_window.setModal(True)
            self.object_select_window.exec_()

            if self.object_select_window.selected_object is not None:
                for k, bus in self.get_selected_buses():
                    bus.country = self.object_select_window.selected_object
                    print('Set {0} into bus {1}'.format(self.object_select_window.selected_object.name, bus.name))

        elif prop == 'zone':
            self.object_select_window = ObjectSelectWindow('Zones', self.circuit.zones)
            self.object_select_window.setModal(True)
            self.object_select_window.exec_()

            if self.object_select_window.selected_object is not None:
                for k, bus in self.get_selected_buses():
                    bus.zone = self.object_select_window.selected_object
                    print('Set {0} into bus {1}'.format(self.object_select_window.selected_object.name, bus.name))
        else:
            error_msg('Unrecognized option' + str(prop))
            return

    def set_cancel_state(self) -> None:
        """
        Cancel what ever's going on that can be cancelled
        @return:
        """

        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                               'Are you sure that you want to cancel the simulation?',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            # send the cancel state to whatever it is being executed

            for drv in self.get_all_threads():
                if drv is not None:
                    if hasattr(drv, 'cancel'):
                        drv.cancel()
        else:
            pass

    def get_available_results(self):
        """
        Get a list of all the available results' objects
        :return: list[object]
        """
        lst = list()

        for drv in self.get_simulations():
            if drv is not None:
                if hasattr(drv, 'results'):
                    if drv.results is not None:
                        lst.append(drv)

        return lst

    def update_available_results(self) -> None:
        """
        Update the results that are displayed in the results tab
        """

        self.available_results_dict = dict()
        self.available_results_steps_dict = dict()

        # clear results lists
        self.ui.results_treeView.setModel(None)

        available_results = self.get_available_results()
        max_steps = 0
        d = dict()
        lst = list()
        for driver in available_results:
            name = driver.tpe.value
            lst.append(name)
            d[name] = driver.results.get_name_tree()
            self.available_results_dict[name] = driver.results.get_name_to_results_type_dict()
            steps = driver.get_steps()
            self.available_results_steps_dict[name] = steps
            if len(steps) > max_steps:
                max_steps = len(steps)

        icons = {SimulationTypes.PowerFlow_run.value: ':/Icons/icons/pf',
                 SimulationTypes.TimeSeries_run.value: ':/Icons/icons/pf_ts.svg',
                 SimulationTypes.ClusteringTimeSeries_run.value: ':/Icons/icons/pf_ts_cluster.svg',
                 SimulationTypes.OPF_run.value: ':/Icons/icons/dcopf.svg',
                 SimulationTypes.OPFTimeSeries_run.value: ':/Icons/icons/dcopf_ts.svg',
                 SimulationTypes.ShortCircuit_run.value: ':/Icons/icons/short_circuit.svg',
                 SimulationTypes.LinearAnalysis_run.value: ':/Icons/icons/ptdf.svg',
                 SimulationTypes.LinearAnalysis_TS_run.value: ':/Icons/icons/ptdf_ts.svg',
                 SimulationTypes.SigmaAnalysis_run.value: ':/Icons/icons/sigma.svg',
                 SimulationTypes.StochasticPowerFlow.value: ':/Icons/icons/stochastic_power_flow.svg',
                 SimulationTypes.ContingencyAnalysis_run.value: ':/Icons/icons/otdf.svg',
                 SimulationTypes.ContingencyAnalysisTS_run.value: ':/Icons/icons/otdf_ts.svg',
                 SimulationTypes.NetTransferCapacity_run.value: ':/Icons/icons/atc.svg',
                 SimulationTypes.NetTransferCapacityTS_run.value: ':/Icons/icons/atc_ts.svg',
                 SimulationTypes.OptimalNetTransferCapacityTimeSeries_run.value: ':/Icons/icons/ntc_opf_ts.svg',
                 SimulationTypes.InputsAnalysis_run.value: ':/Icons/icons/stats.svg',
                 SimulationTypes.NodeGrouping_run.value: ':/Icons/icons/ml.svg',
                 SimulationTypes.ContinuationPowerFlow_run.value: ':/Icons/icons/continuation_power_flow.svg',
                 SimulationTypes.ClusteringAnalysis_run.value: ':/Icons/icons/clustering.svg', }

        self.ui.results_treeView.setModel(gf.get_tree_model(d, 'Results', icons=icons))
        lst.reverse()  # this is to show the latest simulation first
        mdl = gf.get_list_model(lst)
        self.ui.available_results_to_color_comboBox.setModel(mdl)
        self.ui.available_results_to_color_map_comboBox.setModel(mdl)
        self.ui.resultsTableView.setModel(None)

    def clear_results(self):
        """
        Clear the results tab
        """
        self.session.clear()

        self.buses_for_storage = None

        self.calculation_inputs_to_display = None
        self.ui.simulation_data_island_comboBox.clear()

        self.available_results_dict = dict()
        self.ui.resultsTableView.setModel(None)
        self.ui.available_results_to_color_comboBox.model().clear()
        self.ui.results_treeView.setModel(None)

        self.ui.schematic_step_label.setText("")
        self.ui.simulation_results_step_slider.setMinimum(0)
        self.ui.simulation_results_step_slider.setMaximum(0)

        self.ui.map_time_label.setText("")
        self.ui.map_time_horizontalSlider.setMinimum(0)
        self.ui.map_time_horizontalSlider.setMaximum(0)

        self.ui.simulationDataStructureTableView.setModel(None)
        self.ui.profiles_tableView.setModel(None)
        self.ui.resultsTableView.setModel(None)
        self.ui.dataStructureTableView.setModel(None)

        self.ui.sbase_doubleSpinBox.setValue(self.circuit.Sbase)
        self.ui.fbase_doubleSpinBox.setValue(self.circuit.fBase)
        self.ui.model_version_label.setText('Model v. ' + str(self.circuit.model_version))
        self.ui.user_name_label.setText('User: ' + str(self.circuit.user_name))
        if self.open_file_thread_object is not None:
            self.ui.file_information_label.setText(self.open_file_thread_object.file_name)

        self.ui.units_label.setText("")

    def grid_colour_function(self, plot_function, current_study: str, current_step: int):

        use_flow_based_width = self.ui.branch_width_based_on_flow_checkBox.isChecked()
        min_branch_width = self.ui.min_branch_size_spinBox.value()
        max_branch_width = self.ui.max_branch_size_spinBox.value()
        min_bus_width = self.ui.min_node_size_spinBox.value()
        max_bus_width = self.ui.max_node_size_spinBox.value()
        cmap_text = self.ui.palette_comboBox.currentText()

        cmap = self.cmap_dict[cmap_text]

        if current_study == sim.PowerFlowDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.PowerFlow_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.Sbus,
                                 Sf=results.Sf,
                                 St=results.St,
                                 voltages=results.voltage,
                                 loadings=np.abs(results.loading),
                                 types=results.bus_types,
                                 losses=results.losses,
                                 failed_br_idx=None,
                                 hvdc_Pf=results.hvdc_Pf,
                                 hvdc_Pt=results.hvdc_Pt,
                                 hvdc_losses=results.hvdc_losses,
                                 hvdc_loading=results.hvdc_loading,
                                 use_flow_based_width=use_flow_based_width,
                                 ma=results.tap_module,
                                 theta=results.tap_angle,
                                 Beq=results.Beq,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.PowerFlowTimeSeriesDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.TimeSeries_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.S[current_step, :],
                                 Sf=results.Sf[current_step, :],
                                 St=results.St[current_step, :],
                                 voltages=results.voltage[current_step, :],
                                 loadings=np.abs(results.loading[current_step, :]),
                                 types=results.bus_types,
                                 losses=results.losses[current_step, :],
                                 failed_br_idx=None,
                                 hvdc_Pf=results.hvdc_Pf[current_step, :],
                                 hvdc_Pt=results.hvdc_Pt[current_step, :],
                                 hvdc_losses=results.hvdc_losses[current_step, :],
                                 hvdc_loading=results.hvdc_loading[current_step, :],
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.ContinuationPowerFlowDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.ContinuationPowerFlow_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.Sbus[current_step, :],
                                 Sf=results.Sf[current_step, :],
                                 voltages=results.voltages[current_step, :],
                                 loadings=np.abs(results.loading[current_step, :]),
                                 types=results.bus_types,
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.StochasticPowerFlowDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.StochasticPowerFlow)

            return plot_function(circuit=self.circuit,
                                 voltages=results.V_points[current_step, :],
                                 loadings=np.abs(results.loading_points[current_step, :]),
                                 Sf=results.Sbr_points[current_step, :],
                                 types=results.bus_types,
                                 Sbus=results.S_points[current_step, :],
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.ShortCircuitDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.ShortCircuit_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.Sbus1,
                                 Sf=results.Sf1,
                                 voltages=results.voltage1,
                                 types=results.bus_types,
                                 loadings=results.loading1,
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.OptimalPowerFlowDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.OPF_run)

            return plot_function(circuit=self.circuit,
                                 voltages=results.voltage,
                                 loadings=results.loading,
                                 types=results.bus_types,
                                 Sf=results.Sf,
                                 Sbus=results.Sbus,
                                 hvdc_Pf=results.hvdc_Pf,
                                 hvdc_Pt=-results.hvdc_Pf,
                                 hvdc_loading=results.hvdc_loading,
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.OptimalPowerFlowTimeSeriesDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.OPFTimeSeries_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.Sbus[current_step, :],
                                 Sf=results.Sf[current_step, :],
                                 voltages=results.voltage[current_step, :],
                                 loadings=np.abs(results.loading[current_step, :]),
                                 types=results.bus_types,
                                 hvdc_Pf=results.hvdc_Pf[current_step, :],
                                 hvdc_Pt=-results.hvdc_Pf[current_step, :],
                                 hvdc_loading=results.hvdc_loading[current_step, :],
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.LinearAnalysisDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.LinearAnalysis_run)
            voltage = np.ones(self.circuit.get_bus_number())

            return plot_function(circuit=self.circuit,
                                 Sbus=results.Sbus,
                                 Sf=results.Sf,
                                 St=-results.Sf,
                                 voltages=voltage,
                                 loadings=results.loading,
                                 types=results.bus_types,
                                 loading_label='Loading',
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.LinearAnalysisTimeSeriesDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.LinearAnalysis_TS_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.S[current_step],
                                 Sf=results.Sf[current_step],
                                 voltages=results.voltage[current_step],
                                 loadings=np.abs(results.loading[current_step]),
                                 types=results.bus_types,
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.ContingencyAnalysisDriver.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.ContingencyAnalysis_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.S[current_step, :],
                                 Sf=results.Sf[current_step, :],
                                 voltages=results.voltage[current_step, :],
                                 loadings=np.abs(results.loading[current_step, :]),
                                 types=results.bus_types,
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.ContingencyAnalysisTimeSeries.tpe.value:
            drv, results = self.session.get_driver_results(sim.SimulationTypes.ContingencyAnalysisTS_run)

            return plot_function(circuit=self.circuit,
                                 Sbus=results.S[current_step, :],
                                 Sf=results.worst_flows[current_step, :],
                                 voltages=np.ones(results.nbus, dtype=complex),
                                 loadings=np.abs(results.worst_loading[current_step]),
                                 types=results.bus_types,
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.InputsAnalysisDriver.tpe.value:

            drv, results = self.session.get_driver_results(sim.SimulationTypes.InputsAnalysis_run)
            nbus = self.circuit.get_bus_number()
            nbr = self.circuit.get_branch_number()
            # empty
            return plot_function(circuit=self.circuit,
                                 Sbus=np.zeros(nbus, dtype=complex),
                                 Sf=np.zeros(nbr, dtype=complex),
                                 loadings=np.zeros(nbr, dtype=complex),
                                 voltages=np.ones(nbus, dtype=complex),
                                 use_flow_based_width=use_flow_based_width,
                                 min_branch_width=min_branch_width,
                                 max_branch_width=max_branch_width,
                                 min_bus_width=min_bus_width,
                                 max_bus_width=max_bus_width,
                                 cmap=cmap)

        elif current_study == sim.AvailableTransferCapacityTimeSeriesDriver.tpe.value:
            pass

        elif current_study == sim.AvailableTransferCapacityDriver.tpe.value:
            pass

        elif current_study == 'Transient stability':
            raise Exception('Not implemented :(')

        else:
            print('<' + current_study + '> Not implemented :(')

    def colour_schematic(self) -> None:
        """
        Color the grid now
        """

        if not self.ui.draw_schematic_checkBox.isChecked():
            # The schematic drawing is disabled
            return None

        if self.ui.available_results_to_color_comboBox.currentIndex() > -1:
            self.grid_colour_function(plot_function=viz.colour_the_schematic,
                                      current_study=self.ui.available_results_to_color_comboBox.currentText(),
                                      current_step=self.ui.simulation_results_step_slider.value())

    def colour_next_simulation_step(self):
        """
        Next colour step
        """
        current_step = self.ui.simulation_results_step_slider.value()
        count = self.ui.simulation_results_step_slider.maximum() + 1

        if count > 0:
            nxt = current_step + 1

            if nxt >= count:
                nxt = count - 1

            self.ui.simulation_results_step_slider.setValue(nxt)

            self.colour_schematic()

    def colour_previous_simulation_step(self):
        """
        Prev colour step
        """
        current_step = self.ui.simulation_results_step_slider.value()
        count = self.ui.simulation_results_step_slider.maximum() + 1

        if count > 0:
            prv = current_step - 1

            if prv < 0:
                prv = 0

            self.ui.simulation_results_step_slider.setValue(prv)

            self.colour_schematic()

    def colour_next_simulation_step_map(self):
        """
        Next colour step
        """
        current_step = self.ui.map_time_horizontalSlider.value()
        count = self.ui.map_time_horizontalSlider.maximum() + 1

        if count > 0:
            nxt = current_step + 1

            if nxt >= count:
                nxt = count - 1

            self.ui.map_time_horizontalSlider.setValue(nxt)

            self.colour_map()

    def colour_previous_simulation_step_map(self):
        """
        Prev colour step
        """
        current_step = self.ui.map_time_horizontalSlider.value()
        count = self.ui.map_time_horizontalSlider.maximum() + 1

        if count > 0:
            prv = current_step - 1

            if prv < 0:
                prv = 0

            self.ui.map_time_horizontalSlider.setValue(prv)

            self.colour_map()

    def update_available_steps_to_color(self):
        """
        Update the available simulation steps in the combo box
        """
        if self.ui.available_results_to_color_comboBox.currentIndex() > -1:
            current_study = self.ui.available_results_to_color_comboBox.currentText()

            self.schematic_list_steps = self.available_results_steps_dict[current_study]

            if len(self.schematic_list_steps) > 0:
                self.ui.simulation_results_step_slider.setMinimum(0)
                self.ui.simulation_results_step_slider.setMaximum(len(self.schematic_list_steps) - 1)
                self.ui.simulation_results_step_slider.setSliderPosition(0)
                self.ui.schematic_step_label.setText(self.schematic_list_steps[0])
            else:
                self.ui.simulation_results_step_slider.setMinimum(0)
                self.ui.simulation_results_step_slider.setMaximum(0)
                self.ui.simulation_results_step_slider.setSliderPosition(0)
                self.ui.schematic_step_label.setText("No steps")

    def update_available_steps_to_color_map(self):
        """
        Update the available simulation steps in the combo box
        """
        if self.ui.available_results_to_color_map_comboBox.currentIndex() > -1:
            current_study = self.ui.available_results_to_color_map_comboBox.currentText()

            self.map_list_steps = self.available_results_steps_dict[current_study]

            if len(self.map_list_steps) > 0:
                self.ui.map_time_horizontalSlider.setMinimum(0)
                self.ui.map_time_horizontalSlider.setMaximum(len(self.map_list_steps) - 1)
                self.ui.map_time_horizontalSlider.setSliderPosition(0)
                self.ui.map_time_label.setText(self.map_list_steps[0])
            else:
                self.ui.map_time_horizontalSlider.setMinimum(0)
                self.ui.map_time_horizontalSlider.setMaximum(0)
                self.ui.map_time_horizontalSlider.setSliderPosition(0)
                self.ui.map_time_label.setText("No steps")

    def map_time_slider_change(self):
        """

        :return:
        """
        idx = self.ui.map_time_horizontalSlider.value()
        if idx > -1:
            self.ui.map_time_label.setText(self.map_list_steps[idx])

    def schematic_time_slider_change(self):
        idx = self.ui.simulation_results_step_slider.value()

        if len(self.schematic_list_steps):
            if idx > -1:
                self.ui.schematic_step_label.setText(self.schematic_list_steps[idx])
        else:
            self.ui.schematic_step_label.setText("No steps")

    def results_tree_view_click(self, index):
        """
        Display the simulation results on the results table
        """
        tree_mdl = self.ui.results_treeView.model()
        item = tree_mdl.itemFromIndex(index)
        path = gf.get_tree_item_path(item)

        if len(path) > 1:

            if len(path) == 2:
                study_name = path[0]
                result_name = path[1]
            elif len(path) == 3:
                study_name = path[0]
                result_name = path[2]
            else:
                raise Exception('Path len ' + str(len(path)) + ' not supported')

            if study_name in self.available_results_dict.keys():
                if result_name in self.available_results_dict[study_name].keys():

                    study_type = self.available_results_dict[study_name][result_name]

                    self.results_mdl = None

                    self.results_mdl = self.session.get_results_model_by_name(study_name=study_name,
                                                                              study_type=study_type)

                    if self.results_mdl is not None:

                        if self.ui.results_as_abs_checkBox.isChecked():
                            self.results_mdl.convert_to_abs()

                        if self.ui.results_as_cdf_checkBox.isChecked():
                            self.results_mdl.convert_to_cdf()

                        # set the table model
                        self.ui.resultsTableView.setModel(self.results_mdl)
                        self.ui.units_label.setText(self.results_mdl.units)
                    else:
                        self.ui.resultsTableView.setModel(None)
                        self.ui.units_label.setText("")

                else:
                    self.ui.resultsTableView.setModel(None)
                    self.ui.units_label.setText("")
            else:
                self.ui.resultsTableView.setModel(None)
                self.ui.units_label.setText("")

    def plot_results(self):
        """
        Plot the results
        """
        mdl: ResultsModel = self.ui.resultsTableView.model()

        if mdl is not None:

            plt.rcParams["date.autoformatter.minute"] = "%Y-%m-%d %H:%M:%S"

            # get the selected element
            obj_idx = self.ui.resultsTableView.selectedIndexes()

            # create figure to plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            if len(obj_idx):

                # get the unique columns in the selected cells
                cols = np.zeros(len(obj_idx), dtype=int)
                rows = np.zeros(len(obj_idx), dtype=int)

                for i in range(len(obj_idx)):
                    cols[i] = obj_idx[i].column()
                    rows[i] = obj_idx[i].row()

                cols = np.unique(cols)
                rows = np.unique(rows)

            else:
                # plot all
                cols = None
                rows = None

            # none selected, plot all
            mdl.plot(ax=ax, selected_col_idx=cols, selected_rows=rows, stacked=False)

            plt.show()

    def save_results_df(self):
        """
        Save the data displayed at the results as excel
        """
        mdl: ResultsModel = self.ui.resultsTableView.model()

        if mdl is not None:
            file, filter = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", '',
                                                                 filter="CSV (*.csv);;Excel files (*.xlsx)")

            if file != '':
                if 'xlsx' in filter:
                    f = file
                    if not f.endswith('.xlsx'):
                        f += '.xlsx'
                    mdl.save_to_excel(f)
                    print('Saved!')
                if 'csv' in filter:
                    f = file
                    if not f.endswith('.csv'):
                        f += '.csv'
                    mdl.save_to_csv(f)
                    print('Saved!')
                else:
                    error_msg(file[0] + ' is not valid :(')
        else:
            warning_msg('There is no profile displayed, please display one', 'Copy profile to clipboard')

    def copy_results_data(self):
        """
        Copy the current displayed profiles to the clipboard
        """
        mdl = self.ui.resultsTableView.model()
        if mdl is not None:
            mdl.copy_to_clipboard()
            print('Copied!')
        else:
            warning_msg('There is no profile displayed, please display one', 'Copy profile to clipboard')

    def copy_objects_data(self):
        """
        Copy the current displayed objects table to the clipboard
        """
        mdl = self.ui.dataStructureTableView.model()
        if mdl is not None:
            mdl.copy_to_clipboard()
            print('Copied!')
        else:
            warning_msg('There is no data displayed, please display one', 'Copy profile to clipboard')

    def copy_results_data_as_numpy(self):
        """
        Copy the current displayed profiles to the clipboard
        """
        mdl = self.ui.resultsTableView.model()
        if mdl is not None:
            mdl.copy_numpy_to_clipboard()
            print('Copied!')
        else:
            warning_msg('There is no profile displayed, please display one', 'Copy profile to clipboard')

    def set_state(self):
        """
        Set the selected profiles state in the grid
        """
        idx = self.ui.profile_time_selection_comboBox.currentIndex()

        if idx > -1:
            self.circuit.set_state(t=idx)
        else:
            info_msg('No time state selected', 'Set state')

    def set_value_to_column(self):
        """
        Set the value to all the column
        :return: Nothing
        """
        idx = self.ui.dataStructureTableView.currentIndex()
        mdl = self.ui.dataStructureTableView.model()  # is of type ObjectsModel
        col = idx.column()
        if mdl is not None:
            if col > -1:
                mdl.copy_to_column(idx)
                # update the view
                self.view_objects_data()
            else:
                info_msg('Select some element to serve as source to copy', 'Set value to column')
        else:
            pass

    def display_grid_analysis(self):
        """
        Display the grid analysis GUI
        """

        self.analysis_dialogue = GridAnalysisGUI(parent=self,
                                                 object_types=self.grid_editor.object_types,
                                                 circuit=self.circuit)

        self.analysis_dialogue.resize(int(1.61 * 600.0), 600)
        self.analysis_dialogue.show()

    def adjust_all_node_width(self):
        """
        Adapt the width of all the nodes to their names
        """
        for bus in self.circuit.buses:
            if bus.graphic_obj is not None:
                bus.graphic_obj.adapt()

    def set_up_profile_sliders(self):
        """
        Set up profiles
        """
        if self.circuit.time_profile is not None:
            t = len(self.circuit.time_profile) - 1

            self.ui.profile_start_slider.setMinimum(0)
            self.ui.profile_start_slider.setMaximum(t)
            self.ui.profile_start_slider.setValue(0)

            self.ui.profile_end_slider.setMinimum(0)
            self.ui.profile_end_slider.setMaximum(t)
            self.ui.profile_end_slider.setValue(t)
        else:
            pass

    def change_circuit_base(self):
        """
        Update the circuit base values from the UI
        """

        Sbase_new = self.ui.sbase_doubleSpinBox.value()
        self.circuit.change_base(Sbase_new)

        self.circuit.fBase = self.ui.fbase_doubleSpinBox.value()

    def explosion_factor_change(self):
        """
        Change the node explosion factor
        """
        if self.grid_editor is not None:
            self.grid_editor.expand_factor = self.ui.explosion_factor_doubleSpinBox.value()

    def zoom_in(self):
        """
        Zoom the diagram in
        """
        if self.grid_editor is not None:
            self.grid_editor.diagramView.zoom_in()

    def zoom_out(self):
        """
        Zoom the diagram out
        """
        if self.grid_editor is not None:
            self.grid_editor.diagramView.zoom_out()

    def profile_sliders_changed(self):
        """
        Correct sliders if they change
        """
        start = self.ui.profile_start_slider.value()
        end = self.ui.profile_end_slider.value()

        if start > end:
            self.ui.profile_end_slider.setValue(start)
            end = start

        if self.circuit.time_profile is not None:
            if len(self.circuit.time_profile) > 0:
                t1 = self.circuit.time_profile[start]
                t2 = self.circuit.time_profile[end]
                t1 = pd.to_datetime(t1).strftime('%d/%m/%Y %H:%M')
                t2 = pd.to_datetime(t2).strftime('%d/%m/%Y %H:%M')
                self.ui.profile_label.setText(str(t1) + ' -> ' + str(t2))

    def edit_from_catalogue(self):
        """
        Edit catalogue element
        """
        model = self.ui.dataStructureTableView.model()
        sel_item = self.ui.dataStructuresTreeView.selectedIndexes()[0]
        elm_type = sel_item.data(role=QtCore.Qt.DisplayRole)

        if model is not None:

            # get the selected index
            idx = self.ui.dataStructureTableView.currentIndex().row()

            if idx > -1:
                if elm_type == dev.DeviceType.OverheadLineTypeDevice.value:

                    # pick the object
                    tower = self.circuit.overhead_line_types[idx]

                    # launch editor
                    self.tower_builder_window = TowerBuilderGUI(parent=self,
                                                                tower=tower,
                                                                wires_catalogue=self.circuit.wire_types)
                    self.tower_builder_window.resize(int(1.81 * 700.0), 700)
                    self.tower_builder_window.exec()
                    self.collect_memory()

                else:

                    warning_msg('No editor available.\nThe values can be changes from within the table.',
                                'Transformers')
            else:
                info_msg('Choose an element from the table')
        else:
            info_msg('Select a catalogue element and then a catalogue object')

    def process_templates(self):
        """
        Process all Branches templates
        """
        if self.circuit is not None:
            logger = self.circuit.apply_all_branch_types()

            if len(logger) > 0:
                dlg = LogsDialogue('Process templates', logger)
                dlg.exec_()

    def recompile_circuits_for_display(self):
        """
        Recompile the circuits available to display
        :return:
        """
        if self.circuit is not None:

            engine = self.get_preferred_engine()

            if engine == bs.EngineType.GridCal:
                numerical_circuit = core.compile_numerical_circuit_at(circuit=self.circuit, t_idx=None)
                calculation_inputs = numerical_circuit.split_into_islands()
                self.calculation_inputs_to_display = calculation_inputs

            elif engine == bs.EngineType.Bentayga:
                import GridCal.Engine.Core.Compilers.circuit_to_bentayga as ben
                self.calculation_inputs_to_display = ben.get_snapshots_from_bentayga(self.circuit)

            elif engine == bs.EngineType.NewtonPA:
                import GridCal.Engine.Core.Compilers.circuit_to_newton_pa as ne
                self.calculation_inputs_to_display = ne.get_snapshots_from_newtonpa(self.circuit)

            else:
                # fallback to gridcal
                numerical_circuit = core.compile_numerical_circuit_at(circuit=self.circuit, t_idx=None)
                calculation_inputs = numerical_circuit.split_into_islands()
                self.calculation_inputs_to_display = calculation_inputs

            return True
        else:
            self.calculation_inputs_to_display = None
            return False

    def update_islands_to_display(self):
        """
        Compile the circuit and allow the display of the calculation objects
        :return:
        """
        self.recompile_circuits_for_display()
        self.ui.simulation_data_island_comboBox.clear()
        lst = ['Island ' + str(i) for i, circuit in enumerate(self.calculation_inputs_to_display)]
        self.ui.simulation_data_island_comboBox.addItems(lst)
        if len(self.calculation_inputs_to_display) > 0:
            self.ui.simulation_data_island_comboBox.setCurrentIndex(0)

    def plot_style_change(self):
        """
        Change the style
        """
        style = self.ui.plt_style_comboBox.currentText()
        plt.style.use(style)

    def copy_profiles(self):
        """
        Copy the current displayed profiles to the clipboard
        """

        mdl = self.ui.profiles_tableView.model()
        if mdl is not None:
            mdl.copy_to_clipboard()
            print('Copied!')
        else:
            warning_msg('There is no profile displayed, please display one', 'Copy profile to clipboard')

    def paste_profiles(self):
        """
        Paste clipboard data into the profile
        """

        mdl = self.ui.profiles_tableView.model()
        if mdl is not None:

            if len(self.ui.profiles_tableView.selectedIndexes()) > 0:
                index = self.ui.profiles_tableView.selectedIndexes()[0]
                row_idx = index.row()
                col_idx = index.column()
            else:
                row_idx = 0
                col_idx = 0

            mdl.paste_from_clipboard(row_idx=row_idx, col_idx=col_idx)
        else:
            warning_msg('There is no profile displayed, please display one', 'Paste profile to clipboard')

    def undo(self):
        """
        Undo table changes
        """

        model = self.ui.profiles_tableView.model()
        if model is not None:
            model.undo()
        else:
            pass

    def redo(self):
        """
        redo table changes
        """
        model = self.ui.profiles_tableView.model()
        if model is not None:
            model.redo()
        else:
            pass

    def smart_search(self):
        """
        Filter
        """

        if len(self.type_objects_list) > 0:
            command = self.ui.smart_search_lineEdit.text().lower()
            attr = self.ui.property_comboBox.currentText()

            elm = self.type_objects_list[0]
            tpe = elm.editable_headers[attr].tpe

            filtered_objects = list()

            if command.startswith('>') and not command.startswith('>='):
                # greater than selection
                args = command.replace('>', '').strip()

                try:
                    args = tpe(args)
                except TypeError:
                    error_msg('Could not parse the argument for the data type')
                    return

                filtered_objects = [x for x in self.type_objects_list if getattr(x, attr) > args]

            elif command.startswith('<') and not command.startswith('<='):
                # "less than" selection
                args = command.replace('<', '').strip()

                try:
                    args = tpe(args)
                except TypeError:
                    error_msg('Could not parse the argument for the data type')
                    return

                filtered_objects = [x for x in self.type_objects_list if getattr(x, attr) < args]

            elif command.startswith('>='):
                # greater or equal than selection
                args = command.replace('>=', '').strip()

                try:
                    args = tpe(args)
                except TypeError:
                    error_msg('Could not parse the argument for the data type')
                    return

                filtered_objects = [x for x in self.type_objects_list if getattr(x, attr) >= args]

            elif command.startswith('<='):
                # "less or equal than" selection
                args = command.replace('<=', '').strip()

                try:
                    args = tpe(args)
                except TypeError:
                    error_msg('Could not parse the argument for the data type')
                    return

                filtered_objects = [x for x in self.type_objects_list if getattr(x, attr) <= args]

            elif command.startswith('*'):
                # "like" selection
                args = command.replace('*', '').strip()

                if tpe == str:

                    try:
                        args = tpe(args)
                    except TypeError:
                        error_msg('Could not parse the argument for the data type')
                        return

                    filtered_objects = [x for x in self.type_objects_list if args in getattr(x, attr).lower()]

                elif elm.device_type == dev.DeviceType.BusDevice:
                    filtered_objects = [x for x in self.type_objects_list if args in getattr(x, attr).name.lower()]

                else:
                    info_msg('This filter type is only valid for strings')

            elif command.startswith('='):
                # Exact match
                args = command.replace('=', '').strip()

                if tpe == str:

                    try:
                        args = tpe(args)
                    except TypeError:
                        error_msg('Could not parse the argument for the data type')
                        return

                    filtered_objects = [x for x in self.type_objects_list if getattr(x, attr).lower() == args]

                elif tpe == bool:

                    if args.lower() == 'true':
                        args = True
                    elif args.lower() == 'false':
                        args = False
                    else:
                        args = False

                    filtered_objects = [x for x in self.type_objects_list if getattr(x, attr) == args]

                elif elm.device_type == dev.DeviceType.BusDevice:
                    filtered_objects = [x for x in self.type_objects_list if args == getattr(x, attr).name.lower()]

                else:
                    try:
                        filtered_objects = [x for x in self.type_objects_list if getattr(x, attr).name.lower() == args]
                    except:
                        filtered_objects = [x for x in self.type_objects_list if getattr(x, attr) == args]

            elif command.startswith('!='):
                # Exact match
                args = command.replace('==', '').strip()

                if tpe == str:

                    try:
                        args = tpe(args)
                    except TypeError:
                        error_msg('Could not parse the argument for the data type')
                        return

                    filtered_objects = [x for x in self.type_objects_list if getattr(x, attr).lower() != args]

                elif elm.device_type == dev.DeviceType.BusDevice:
                    filtered_objects = [x for x in self.type_objects_list if args != getattr(x, attr).name.lower()]

                else:
                    filtered_objects = [x for x in self.type_objects_list if getattr(x, attr) != args]

            else:
                filtered_objects = self.type_objects_list

            self.display_filter(filtered_objects)

        else:
            # nothing to search
            pass

    def delete_and_reduce_selected_objects(self):
        """
        Delete and reduce the buses
        This function removes the buses but whenever a bus is removed, the devices connected to it
        are inherited by the bus of higher voltage that is connected.
        If the bus is isolated, those devices are lost.
        """
        model = self.ui.dataStructureTableView.model()

        if model is not None:
            sel_idx = self.ui.dataStructureTableView.selectedIndexes()
            objects = model.objects

            if len(objects) > 0:

                if objects[0].device_type == dev.DeviceType.BusDevice:

                    if len(sel_idx) > 0:

                        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                               'Do you want to reduce and delete the selected elements?',
                                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

                        if reply == QtWidgets.QMessageBox.Yes:

                            self.LOCK()

                            self.add_simulation(sim.SimulationTypes.Delete_and_reduce_run)

                            self.delete_and_reduce_driver = sim.DeleteAndReduce(grid=self.circuit,
                                                                                objects=objects,
                                                                                sel_idx=sel_idx)

                            self.delete_and_reduce_driver.progress_signal.connect(self.ui.progressBar.setValue)
                            self.delete_and_reduce_driver.progress_text.connect(self.ui.progress_label.setText)
                            self.delete_and_reduce_driver.done_signal.connect(self.UNLOCK)
                            self.delete_and_reduce_driver.done_signal.connect(
                                self.post_delete_and_reduce_selected_objects)

                            self.delete_and_reduce_driver.start()

                        else:
                            # selected QMessageBox.No
                            pass

                    else:
                        # no selection
                        pass

                else:
                    info_msg('This function is only applicable to buses')

            else:
                # no objects
                pass
        else:
            pass

    def post_delete_and_reduce_selected_objects(self):
        """
        POst delete and merge buses
        """
        if self.delete_and_reduce_driver is not None:

            for bus in self.delete_and_reduce_driver.buses_merged:
                if bus.graphic_obj is not None:
                    bus.graphic_obj.create_children_icons()
                    bus.graphic_obj.arrange_children()

            self.create_schematic_from_api(explode_factor=1)

            self.clear_results()

            self.remove_simulation(sim.SimulationTypes.Delete_and_reduce_run)

            self.UNLOCK()

    def delete_selected_objects(self):
        """
        Delete selection
        """

        model = self.ui.dataStructureTableView.model()

        if model is not None:
            sel_idx = self.ui.dataStructureTableView.selectedIndexes()
            objects = model.objects

            if len(sel_idx) > 0:

                ok = yes_no_question('Are you sure that you want to delete the selected elements?', 'Delete')
                if ok:

                    # get the unique rows
                    unique = set()
                    for idx in sel_idx:
                        unique.add(idx.row())

                    unique = list(unique)
                    unique.sort(reverse=True)
                    for r in unique:

                        if objects[r].graphic_obj is not None:
                            # this is a more complete function than the circuit one because it removes the
                            # graphical items too, and for loads and generators it deletes them properly
                            objects[r].graphic_obj.remove(ask=False)
                        else:
                            # objects.pop(r)
                            self.circuit.delete_elements_by_type(obj=objects[r])

                    # update the view
                    self.display_filter(objects)
                    self.update_area_combos()
                    self.update_date_dependent_combos()
                else:
                    pass
            else:
                info_msg('Select some cells')
        else:
            pass

    def delete_shit(self, min_island=1):
        """
        Delete small islands, disconnected stuff and other garbage
        """
        numerical_circuit_ = core.compile_numerical_circuit_at(circuit=self.circuit, )
        islands = numerical_circuit_.split_into_islands()
        logger = bs.Logger()
        buses_to_delete = list()
        buses_to_delete_idx = list()
        for island in islands:
            if island.nbus <= min_island:
                for r in island.original_bus_idx:
                    buses_to_delete.append(self.circuit.buses[r])
                    buses_to_delete_idx.append(r)

        for r, bus in enumerate(self.circuit.buses):
            if not bus.active and not np.any(bus.active_prof):
                if r not in buses_to_delete_idx:
                    buses_to_delete.append(bus)
                    buses_to_delete_idx.append(r)

        for elm in buses_to_delete:
            if elm.graphic_obj is not None:
                # this is a more complete function than the circuit one because it removes the
                # graphical items too, and for loads and generators it deletes them properly
                print('Deleted ', elm.device_type.value, elm.name)
                logger.add_info("Deleted " + str(elm.device_type.value), elm.name)
                elm.graphic_obj.remove(ask=False)

        # search other elements to delete
        for dev_lst in [self.circuit.lines,
                        self.circuit.dc_lines,
                        self.circuit.vsc_devices,
                        self.circuit.hvdc_lines,
                        self.circuit.transformers2w,
                        self.circuit.get_generators(),
                        self.circuit.get_loads(),
                        self.circuit.get_shunts(),
                        self.circuit.get_batteries(),
                        self.circuit.get_static_generators(),
                        ]:
            for elm in dev_lst:
                if not elm.active and not np.any(elm.active_prof):
                    if elm.graphic_obj is not None:
                        # this is a more complete function than the circuit one because it removes the
                        # graphical items too, and for loads and generators it deletes them properly
                        print('Deleted ', elm.device_type.value, elm.name)
                        logger.add_info("Deleted " + str(elm.device_type.value), elm.name)
                        elm.graphic_obj.remove(ask=False)

        return logger

    def correct_branch_monitoring(self, max_loading=1.0):
        """
        The NTC optimization and other algorithms will not work if we have overloaded Branches in DC monitored
        We can try to not monitor those to try to get it working
        """
        res = self.session.power_flow

        if res is None:
            self.console_msg('No power flow results.\n')
        else:
            branches = self.circuit.get_branches_wo_hvdc()
            for elm, loading in zip(branches, res.loading):
                if loading >= max_loading:
                    elm.monitor_loading = False
                    self.console_msg('Disabled loading monitoring for {0}, loading: {1}'.format(elm.name, loading))

    def add_objects(self):
        """
        Add default objects objects
        """
        model = self.ui.dataStructureTableView.model()
        elm_type = self.ui.dataStructuresTreeView.selectedIndexes()[0].data(role=QtCore.Qt.DisplayRole)

        if model is not None:

            if elm_type == dev.DeviceType.SubstationDevice.value:
                self.circuit.add_substation(dev.Substation('Default'))
                self.update_area_combos()

            elif elm_type == dev.DeviceType.ZoneDevice.value:
                self.circuit.add_zone(dev.Zone('Default'))
                self.update_area_combos()

            elif elm_type == dev.DeviceType.AreaDevice.value:
                self.circuit.add_area(dev.Area('Default'))
                self.update_area_combos()

            elif elm_type == dev.DeviceType.CountryDevice.value:
                self.circuit.add_country(dev.Country('Default'))
                self.update_area_combos()

            elif elm_type == dev.DeviceType.BusDevice.value:
                self.circuit.add_bus(dev.Bus(name='Bus ' + str(len(self.circuit.buses) + 1),
                                             area=self.circuit.areas[0],
                                             zone=self.circuit.zones[0],
                                             substation=self.circuit.substations[0],
                                             country=self.circuit.countries[0]))

            elif elm_type == dev.DeviceType.ContingencyGroupDevice.value:
                group = dev.ContingencyGroup(name="Contingency group " + str(len(self.circuit.contingency_groups) + 1))
                self.circuit.add_contingency_group(group)

            elif elm_type == dev.DeviceType.InvestmentsGroupDevice.value:
                group = dev.InvestmentsGroup(name="Investments group " + str(len(self.circuit.contingency_groups) + 1))
                self.circuit.add_investments_group(group)

            elif elm_type == dev.DeviceType.Technology.value:
                tech = dev.Technology(name="Technology " + str(len(self.circuit.technologies) + 1))
                self.circuit.add_technology(tech)

            elif elm_type == dev.DeviceType.OverheadLineTypeDevice.value:

                obj = dev.OverheadLineType()
                obj.frequency = self.circuit.fBase
                obj.tower_name = 'Tower ' + str(len(self.circuit.overhead_line_types))
                self.circuit.add_overhead_line(obj)

            elif elm_type == dev.DeviceType.UnderGroundLineDevice.value:

                name = 'Cable ' + str(len(self.circuit.underground_cable_types))
                obj = dev.UndergroundLineType(name=name)
                self.circuit.add_underground_line(obj)

            elif elm_type == dev.DeviceType.SequenceLineDevice.value:

                name = 'Sequence line ' + str(len(self.circuit.sequence_line_types))
                obj = dev.SequenceLineType(name=name)
                self.circuit.add_sequence_line(obj)

            elif elm_type == dev.DeviceType.WireDevice.value:

                name = 'Wire ' + str(len(self.circuit.wire_types))
                obj = dev.Wire(name=name, gmr=0.01, r=0.01, x=0)
                self.circuit.add_wire(obj)

            elif elm_type == dev.DeviceType.TransformerTypeDevice.value:

                name = 'Transformer type ' + str(len(self.circuit.transformer_types))
                obj = dev.TransformerType(hv_nominal_voltage=10, lv_nominal_voltage=0.4, nominal_power=2,
                                          copper_losses=0.8, iron_losses=0.1, no_load_current=0.1,
                                          short_circuit_voltage=0.1,
                                          gr_hv1=0.5, gx_hv1=0.5, name=name)
                self.circuit.add_transformer_type(obj)

            elif elm_type == dev.DeviceType.FuelDevice.value:

                name = 'Fuel ' + str(len(self.circuit.fuels))
                obj = dev.Fuel(name=name)
                self.circuit.add_fuel(obj)

            elif elm_type == dev.DeviceType.EmissionGasDevice.value:

                name = 'Gas ' + str(len(self.circuit.emission_gases))
                obj = dev.EmissionGas(name=name)
                self.circuit.add_emission_gas(obj)

            elif elm_type == dev.DeviceType.GeneratorTechnologyAssociation.value:

                obj = dev.GeneratorTechnology()
                self.circuit.add_generator_technology(obj)

            elif elm_type == dev.DeviceType.GeneratorFuelAssociation.value:

                obj = dev.GeneratorFuel()
                self.circuit.add_generator_fuel(obj)

            elif elm_type == dev.DeviceType.GeneratorEmissionAssociation.value:

                obj = dev.GeneratorEmission()
                self.circuit.add_generator_emission(obj)

            else:
                info_msg("This object does not support table-like addition.\nUse the schematic instead.")
                return

            # update the view
            self.view_objects_data()

    def clear_big_bus_markers(self):
        """
        clear all the buses' "big marker"
        """
        for bus in self.circuit.buses:
            if bus.graphic_obj is not None:
                bus.graphic_obj.delete_big_marker()

    def set_big_bus_marker(self, buses, color: QtGui.QColor):
        """
        Set a big marker at the selected buses
        :param buses: list of Bus objects
        :param color: colour to use
        """
        for bus in buses:
            if bus.graphic_obj is not None:
                bus.graphic_obj.add_big_marker(color=color)
                bus.graphic_obj.setSelected(True)

    def highlight_selection_buses(self):
        """
        Highlight and select the buses of the selected objects
        """

        model = self.ui.dataStructureTableView.model()

        if model is not None:

            sel_idx = self.ui.dataStructureTableView.selectedIndexes()
            objects = model.objects

            if len(objects) > 0:

                if len(sel_idx) > 0:

                    unique = set()
                    for idx in sel_idx:
                        unique.add(idx.row())
                    sel_obj = list()
                    for idx in unique:
                        sel_obj.append(objects[idx])

                    elm = objects[0]

                    self.clear_big_bus_markers()
                    color = QtGui.QColor(55, 200, 171, 180)

                    if elm.device_type == dev.DeviceType.BusDevice:

                        self.set_big_bus_marker(buses=sel_obj, color=color)

                    elif elm.device_type in [dev.DeviceType.BranchDevice,
                                             dev.DeviceType.LineDevice,
                                             dev.DeviceType.Transformer2WDevice,
                                             dev.DeviceType.HVDCLineDevice,
                                             dev.DeviceType.VscDevice,
                                             dev.DeviceType.DCLineDevice]:
                        buses = list()
                        for br in sel_obj:
                            buses.append(br.bus_from)
                            buses.append(br.bus_to)
                        self.set_big_bus_marker(buses=buses, color=color)

                    else:
                        buses = list()
                        for elm in sel_obj:
                            buses.append(elm.bus)
                        self.set_big_bus_marker(buses=buses, color=color)

                else:
                    info_msg('Select some elements to highlight', 'Highlight')
            else:
                pass

    def highlight_based_on_property(self):
        """
        Highlight and select the buses of the selected objects
        """

        model = self.ui.dataStructureTableView.model()

        if model is not None:
            objects = model.objects

            if len(objects) > 0:

                elm = objects[0]
                attr = self.ui.property_comboBox.currentText()
                tpe = elm.editable_headers[attr].tpe

                if tpe in [float, int]:

                    self.clear_big_bus_markers()

                    if elm.device_type == dev.DeviceType.BusDevice:
                        # buses
                        buses = objects
                        values = [getattr(elm, attr) for elm in objects]

                    elif elm.device_type in [dev.DeviceType.BranchDevice,
                                             dev.DeviceType.LineDevice,
                                             dev.DeviceType.DCLineDevice,
                                             dev.DeviceType.HVDCLineDevice,
                                             dev.DeviceType.Transformer2WDevice,
                                             dev.DeviceType.SwitchDevice,
                                             dev.DeviceType.VscDevice,
                                             dev.DeviceType.UpfcDevice]:
                        # Branches
                        buses = list()
                        values = list()
                        for br in objects:
                            buses.append(br.bus_from)
                            buses.append(br.bus_to)
                            val = getattr(br, attr)
                            values.append(val)
                            values.append(val)

                    else:
                        # loads, generators, etc...
                        buses = [elm.bus for elm in objects]
                        values = [getattr(elm, attr) for elm in objects]

                    # build the color map
                    seq = [(0.0, 'gray'),
                           (0.5, 'orange'),
                           (1, 'red')]
                    cmap = LinearSegmentedColormap.from_list('lcolors', seq)
                    mx = max(values)

                    if mx != 0:
                        # color based on the value
                        for bus, value in zip(buses, values):
                            if bus.graphic_obj is not None:
                                r, g, b, a = cmap(value / mx)
                                color = QtGui.QColor(r * 255, g * 255, b * 255, a * 255)
                                bus.graphic_obj.add_big_marker(color=color)
                    else:
                        info_msg('The maximum value is 0, so the coloring cannot be applied',
                                 'Highlight based on property')
                else:
                    info_msg('The selected property must be of a numeric type',
                             'Highlight based on property')

            else:
                pass

    def get_selected_buses(self) -> List[Tuple[int, dev.Bus]]:
        """
        Get the selected buses
        :return:
        """
        lst: List[Tuple[int, dev.Bus]] = list()
        for k, bus in enumerate(self.circuit.buses):
            if bus.graphic_obj is not None:
                if bus.graphic_obj.isSelected():
                    lst.append((k, bus))
        return lst

    def get_selected_contingency_devices(self) -> List[dev.EditableDevice]:
        """
        Get the selected buses
        :return:
        """
        lst: List[dev.EditableDevice] = list()
        for k, elm in enumerate(self.circuit.get_contingency_devices()):
            if elm.graphic_obj is not None:
                if elm.graphic_obj.isSelected():
                    lst.append(elm)
        return lst

    def get_selected_investment_devices(self) -> List[dev.EditableDevice]:
        """
        Get the selected buses
        :return:
        """
        lst: List[dev.EditableDevice] = list()
        for k, elm in enumerate(self.circuit.get_investment_devices()):
            if elm.graphic_obj is not None:
                if elm.graphic_obj.isSelected():
                    lst.append(elm)
        return lst

    def delete_selected_from_the_schematic(self):
        """
        Prompt to delete the selected buses from the schematic
        """
        if len(self.circuit.buses) > 0:

            # get the selected buses
            selected = self.get_selected_buses()

            if len(selected) > 0:
                reply = QtWidgets.QMessageBox.question(self, 'Delete',
                                                       'Are you sure that you want to delete the selected elements?',
                                                       QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

                if reply == QtWidgets.QMessageBox.Yes:

                    # remove the buses (from the schematic and the circuit)
                    for k, bus in selected:
                        if bus.graphic_obj is not None:
                            # this is a more complete function than the circuit one because it removes the
                            # graphical items too, and for loads and generators it deletes them properly
                            bus.graphic_obj.remove(ask=False)
                else:
                    pass
            else:
                info_msg('Choose some elements from the schematic', 'Delete buses')
        else:
            pass

    def try_to_fix_buses_location(self):
        """
        Try to fix the location of the buses
        """

        selected_buses = self.get_selected_buses()
        if len(selected_buses) > 0:
            self.circuit.try_to_fix_buses_location(buses_selection=selected_buses)
            for k, bus in selected_buses:
                if bus.graphic_obj is not None:
                    bus.graphic_obj.set_position(x=bus.x, y=bus.y)
        else:
            info_msg('Choose some elements from the schematic', 'Fix buses locations')

    def copy_opf_to_profiles(self):
        """
        Copy the results from the OPF time series to the profiles
        """
        drv, results = self.session.get_driver_results(sim.SimulationTypes.OPFTimeSeries_run)

        if results is not None:

            reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                   'Are you sure that you want to overwrite '
                                                   'the generation profiles with the OPF results?',
                                                   QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                for i, gen in enumerate(self.circuit.get_generators()):
                    gen.P_prof = results.generator_power[:, i]

        else:
            warning_msg('The OPF time series has no results :(')

    def valid_time_series(self):
        """
        Check if there are valid time series
        """
        if len(self.circuit.buses) > 0:
            if self.circuit.time_profile is not None:
                if len(self.circuit.time_profile) > 0:
                    return True
        return False

    def delete_created_files(self):
        """
        Delete the files created by GridCal as temporary files
        """
        for f in self.files_to_delete_at_exit:
            if os.path.exists(f):
                os.remove(f)

    def enable_manual_file_operations(self, val=True):
        """
        Enable / disable manual operations
        :param val: True/False
        """
        self.ui.actionSave.setEnabled(val)
        self.ui.actionNew_project.setEnabled(val)
        self.ui.actionOpen_file.setEnabled(val)

    # def file_sync_toggle(self):
    #     """
    #     Toggle file sync on/off
    #     """
    #     if self.ui.actionSync.isChecked():
    #
    #         # attempt to start synchronizing
    #         if os.path.exists(self.file_name):
    #             sleep_time = self.ui.sync_interval_spinBox.value()  # seconds to sleep
    #             self.file_sync_thread = syncdrv.FileSyncThread(self.circuit, file_name=self.file_name,
    #                                                            sleep_time=sleep_time)
    #
    #             # upon sync check (call the gui dialogue)
    #             self.file_sync_thread.sync_event.connect(self.post_file_sync)
    #
    #             # upon sync gui check
    #             self.file_sync_thread.items_processed_event.connect(self.post_file_sync_items_processed)
    #
    #             self.file_sync_thread.start()
    #
    #             # disable the regular save so that you cannot override the synchronization process
    #             self.enable_manual_file_operations(False)
    #
    #         else:
    #             warning_msg('Cannot sync because the file does not exist.\nDid you save the model?')
    #             self.ui.actionSync.setChecked(False)
    #
    #             # enable the regular save button
    #             self.enable_manual_file_operations(True)
    #     else:
    #         # attempt to stop the synchronization
    #         if self.file_sync_thread.isRunning():
    #             self.file_sync_thread.cancel()
    #             self.file_sync_thread.quit()
    #
    #             # enable the regular save button
    #             self.enable_manual_file_operations(True)
    #
    #         self.UNLOCK()

    # def post_file_sync(self):
    #     """
    #     Actions to perform upon synchronization
    #     """
    #
    #     if self.file_sync_thread.version_conflict:
    #         # version conflict and changes
    #         if len(self.file_sync_thread.issues) > 0:
    #
    #             if self.ui.accept_newer_changes_checkBox.isChecked():
    #                 if self.file_sync_thread.highest_version > self.circuit.model_version:
    #                     # there are newer changes and we want to automatically accept them
    #                     self.post_file_sync_items_processed()
    #                 else:
    #                     # there are newer changes but we do not want to automatically accept them
    #                     self.file_sync_window = SyncDialogueWindow(parent=self,
    #                                                                file_sync_thread=self.file_sync_thread)  # will pause the thread
    #                     self.file_sync_window.setModal(True)
    #                     self.file_sync_window.show()
    #             else:
    #                 # we want to check all the conflicts
    #                 self.file_sync_window = SyncDialogueWindow(parent=self,
    #                                                            file_sync_thread=self.file_sync_thread)  # will pause the thread
    #                 self.file_sync_window.setModal(True)
    #                 self.file_sync_window.show()
    #         else:
    #             # just read the file because there were no changes but the version was upgraded
    #             self.circuit.model_version = self.file_sync_thread.highest_version
    #             self.ui.model_version_label.setText('Model v. ' + str(self.circuit.model_version))
    #
    #     else:
    #         # no version conflict, and there were changes on my side
    #         if len(self.file_sync_thread.issues) > 0:
    #             self.save_file()

    # def post_file_sync_items_processed(self):
    #     """
    #     Modify, Add or delete objects after the sync acceptation
    #     This is done here because it concerns the GUI thread
    #     """
    #
    #     # first add any bus that has been created
    #     for issue in self.file_sync_thread.issues:
    #         if issue.issue_type == bs.SyncIssueType.Added and issue.device_type == DeviceType.BusDevice:
    #             # add the bus directly with all the device it may contain
    #             issue.their_elm.delete_children()
    #             if issue.their_elm.graphic_obj is not None:
    #                 issue.their_elm.graphic_obj = self.grid_editor.add_api_bus(issue.their_elm)
    #             self.circuit.add_bus(issue.their_elm)
    #
    #     # create dictionary of buses
    #     bus_dict = self.circuit.get_bus_dict()
    #
    #     # add the rest of the devices
    #     for issue in self.file_sync_thread.issues:
    #
    #         if issue.issue_type == bs.SyncIssueType.Conflict:
    #             if issue.accepted():
    #                 issue.accept_change()
    #
    #         elif issue.issue_type == bs.SyncIssueType.Added:
    #
    #             if issue.device_type == DeviceType.BranchDevice:
    #                 # re_map the buses
    #                 name_f = issue.their_elm.bus_from.name
    #                 issue.their_elm.bus_from = bus_dict[name_f]
    #                 name_t = issue.their_elm.bus_to.name
    #                 issue.their_elm.bus_to = bus_dict[name_t]
    #
    #                 # add the device
    #                 if issue.their_elm.graphic_obj is not None:
    #                     issue.their_elm.graphic_obj = self.grid_editor.add_api_branch(issue.their_elm)
    #                     issue.their_elm.bus_from.graphic_obj.update()
    #                     issue.their_elm.bus_to.graphic_obj.update()
    #                     issue.their_elm.graphic_obj.redraw()
    #                 self.circuit.add_branch(issue.their_elm)
    #
    #             elif issue.device_type == DeviceType.BusDevice:
    #                 # we already added the buses, but we need to exclude them from the list
    #                 continue
    #
    #             else:
    #                 # re_map the buses
    #                 name_f = issue.their_elm.bus.name
    #                 bus = bus_dict[name_f]
    #                 issue.their_elm.bus = bus
    #
    #                 # add the device
    #                 bus.add_device(issue.their_elm)
    #                 if issue.their_elm.graphic_obj is not None:
    #                     bus.graphic_obj.create_children_icons()
    #
    #         elif issue.issue_type == bs.SyncIssueType.Deleted:
    #             if issue.their_elm.graphic_obj is not None:
    #                 issue.my_elm.graphic_obj.remove(ask=False)
    #
    #     # center nodes
    #     self.grid_editor.align_schematic()

    def snapshot_balance(self):
        """
        Snapshot balance report
        """
        df = self.circuit.snapshot_balance()
        self.console_msg('\n' + str(df))

    def add_default_catalogue(self):
        """
        Add default catalogue to circuit
        """
        self.circuit.transformer_types += dev.get_transformer_catalogue()
        self.circuit.underground_cable_types += dev.get_cables_catalogue()
        self.circuit.wire_types += dev.get_wires_catalogue()

    def bus_viewer(self):
        """
        Launch bus viewer
        """
        model = self.ui.dataStructureTableView.model()

        if model is not None:

            sel_idx = self.ui.dataStructureTableView.selectedIndexes()
            objects = model.objects

            if len(objects) > 0:

                if len(sel_idx) > 0:

                    unique = {idx.row() for idx in sel_idx}
                    sel_obj = [objects[idx] for idx in unique][0]
                    root_bus = None
                    if isinstance(sel_obj, dev.Bus):
                        root_bus = sel_obj

                    elif isinstance(sel_obj, dev.Generator):
                        root_bus = sel_obj.bus

                    elif isinstance(sel_obj, dev.Battery):
                        root_bus = sel_obj.bus

                    elif isinstance(sel_obj, dev.Load):
                        root_bus = sel_obj.bus

                    elif isinstance(sel_obj, dev.Shunt):
                        root_bus = sel_obj.bus

                    elif isinstance(sel_obj, dev.Line):
                        root_bus = sel_obj.bus_from

                    elif isinstance(sel_obj, dev.Transformer2W):
                        root_bus = sel_obj.bus_from

                    elif isinstance(sel_obj, dev.DcLine):
                        root_bus = sel_obj.bus_from

                    elif isinstance(sel_obj, dev.HvdcLine):
                        root_bus = sel_obj.bus_from

                    elif isinstance(sel_obj, dev.VSC):
                        root_bus = sel_obj.bus_from

                    elif isinstance(sel_obj, dev.UPFC):
                        root_bus = sel_obj.bus_from

                    if root_bus is not None:
                        window = BusViewerGUI(self.circuit, root_bus)
                        self.bus_viewer_windows.append(window)
                        window.show()

    def import_plexos_node_load(self):
        """
        Open and parse Plexos load file
        """
        fname = self.select_csv_file('Open node load')

        if fname:
            df = pd.read_csv(fname, index_col=0)
            logger = self.circuit.import_plexos_load_profiles(df=df)
            self.update_date_dependent_combos()

            if len(logger) > 0:
                dlg = LogsDialogue('Plexos load import', logger)
                dlg.exec_()

    def import_plexos_generator_generation(self):
        """
        Open and parse Plexos generation file
        """
        fname = self.select_csv_file('Open generation')

        if fname:
            df = pd.read_csv(fname, index_col=0)
            logger = self.circuit.import_plexos_generation_profiles(df=df)
            self.update_date_dependent_combos()

            if len(logger) > 0:
                dlg = LogsDialogue('Plexos generation import', logger)
                dlg.exec_()

    def import_plexos_branch_rates(self):
        """
        Open and parse Plexos load file
        """
        fname = self.select_csv_file('Open branch rates')

        if fname:
            df = pd.read_csv(fname, index_col=0)

            if self.circuit.get_time_number() != df.shape[0]:
                error_msg('The data has a different number of rows than the existing profiles')
            else:

                logger = self.circuit.import_branch_rates_profiles(df=df)
                self.update_date_dependent_combos()

                if len(logger) > 0:
                    dlg = LogsDialogue('Plexos branch rates import', logger)
                    dlg.exec_()

    def search_in_results(self):
        """
        Search in the results model
        """

        if self.results_mdl is not None:
            text = self.ui.sear_results_lineEdit.text().strip()

            if text != '':
                mdl = self.results_mdl.search(text)
            else:
                mdl = None

            self.ui.resultsTableView.setModel(mdl)

    def get_snapshot_circuit(self):
        """
        Get a snapshot compilation
        :return: SnapshotData instance
        """
        return core.compile_numerical_circuit_at(circuit=self.circuit)

    def get_compatible_areas_from_to(self) -> Tuple[bool,
    List[Tuple[int, dev.Bus]], List[Tuple[int, dev.Bus]],
    List[Tuple[int, object, float]], List[Tuple[int, object, float]],
    List[dev.Area], List[dev.Area]]:
        """
        Get the lists that help defining the inter area objects
        :return: success?,
                 list of tuples bus idx, Bus in the areas from,
                 list of tuples bus idx, Bus in the areas to,
                 List of inter area Branches (branch index, branch object, flow sense w.r.t the area exchange),
                 List of inter area HVDC (branch index, branch object, flow sense w.r.t the area exchange),
                 List of areas from,
                 List of areas to
        """
        areas_from_idx = gf.get_checked_indices(self.ui.areaFromListView.model())
        areas_to_idx = gf.get_checked_indices(self.ui.areaToListView.model())
        areas_from = [self.circuit.areas[i] for i in areas_from_idx]
        areas_to = [self.circuit.areas[i] for i in areas_to_idx]

        for a1 in areas_from:
            if a1 in areas_to:
                error_msg("The area from '{0}' is in the list of areas to. This cannot be.".format(a1.name),
                          'Incompatible areas')
                return False, [], [], [], [], [], []
        for a2 in areas_to:
            if a2 in areas_from:
                error_msg("The area to '{0}' is in the list of areas from. This cannot be.".format(a2.name),
                          'Incompatible areas')
                return False, [], [], [], [], [], []

        lst_from = self.circuit.get_areas_buses(areas_from)
        lst_to = self.circuit.get_areas_buses(areas_to)
        lst_br = self.circuit.get_inter_areas_branches(areas_from, areas_to)
        lst_br_hvdc = self.circuit.get_inter_areas_hvdc_branches(areas_from, areas_to)
        return True, lst_from, lst_to, lst_br, lst_br_hvdc, areas_from, areas_to

    @property
    def numerical_circuit(self):
        return self.get_snapshot_circuit()

    @property
    def islands(self):
        numerical_circuit = core.compile_numerical_circuit_at(circuit=self.circuit)
        calculation_inputs = numerical_circuit.split_into_islands()
        return calculation_inputs

    def fuse_devices(self):
        """
        Fuse the devices per node into a single device per category
        """
        ok = yes_no_question("This action will fuse all the devices per node and per category. Are you sure?",
                             "Fuse devices")

        if ok:
            self.circuit.fuse_devices()
            self.create_schematic_from_api()

    # def correct_inconsistencies(self):
    #     """
    #     Call correct inconsistencies
    #     :return:
    #     """
    #     dlg = CorrectInconsistenciesDialogue()
    #     dlg.setModal(True)
    #     dlg.exec_()
    #
    #     if dlg.accepted:
    #         logger = Logger()
    #
    #         self.circuit.correct_inconsistencies(logger=logger,
    #                                              maximum_difference=dlg.max_virtual_tap.value(),
    #                                              min_vset=dlg.min_voltage.value(),
    #                                              max_vset=dlg.max_voltage.value())
    #
    #         if len(logger) > 0:
    #             dlg = LogsDialogue("correct inconsistencies", logger)
    #             dlg.setModal(True)
    #             dlg.exec_()

    def delete_inconsistencies(self):
        """
        Call delete shit
        :return:
        """
        ok = yes_no_question(
            "This action removes all disconnected devices with no active profile and remove all small islands",
            "Delete inconsistencies")

        if ok:
            logger = self.delete_shit()

            if len(logger) > 0:
                dlg = LogsDialogue("Delete inconsistencies", logger)
                dlg.setModal(True)
                dlg.exec_()

    def re_index_time(self):
        """
        Re-index time
        :return:
        """

        dlg = TimeReIndexDialogue()
        dlg.setModal(True)
        dlg.exec_()

        if dlg.accepted:
            self.circuit.re_index_time2(t0=dlg.date_time_editor.dateTime().toPython(),
                                        step_size=dlg.step_length.value(),
                                        step_unit=dlg.units.currentText())

            self.update_date_dependent_combos()

    def fix_generators_active_based_on_the_power(self, ask_before=True):
        """
        set the generators active based on the active power values
        :return:
        """

        if ask_before:
            ok = yes_no_question("This action sets the generation active profile based on the active power profile "
                                 "such that ig a generator active power is zero, the active value is false",
                                 "Set generation active profile")
        else:
            ok = True

        if ok:
            self.circuit.set_generators_active_profile_from_their_active_power()
            self.circuit.set_batteries_active_profile_from_their_active_power()

    def fix_loads_active_based_on_the_power(self, ask_before=True):
        """
        set the loads active based on the active power values
        :return:
        """

        if ask_before:
            ok = yes_no_question("This action sets the generation active profile based on the active power profile "
                                 "such that ig a generator active power is zero, the active value is false",
                                 "Set generation active profile")
        else:
            ok = True

        if ok:
            self.circuit.set_loads_active_profile_from_their_active_power()

    def get_all_objects_in_memory(self):
        objects = []
        # for name, obj in globals().items():
        #     objects.append([name, sys.getsizeof(obj)])

        traverse_objects('MainGUI', self, objects)

        df = pd.DataFrame(data=objects, columns=['Name', 'Size (kb)'])
        df.sort_values(by='Size (kb)', inplace=True, ascending=False)
        return df

    def structure_analysis_plot(self):

        if len(self.ui.dataStructuresTreeView.selectedIndexes()) > 0:
            elm_type = self.ui.dataStructuresTreeView.selectedIndexes()[0].data(role=QtCore.Qt.DisplayRole)

            object_histogram_analysis(circuit=self.circuit, object_type=elm_type, fig=None)
            plt.show()
        else:
            info_msg('Select a data structure')

    def import_profiles_from_models(self):
        """
        Open the dialogue to load profile data from models
        """

        if len(self.circuit.buses) == 0:
            warning_msg("There are no objects to which to assign a profile. \n"
                        "You need to load or create a grid!")
            return

        if self.circuit.time_profile is None:
            self.new_profiles_structure()

        # if there are no profiles:
        if self.circuit.time_profile is not None:
            self.models_input_dialogue = ModelsInputGUI(parent=self,
                                                        time_array=self.circuit.time_profile)

            self.models_input_dialogue.resize(int(1.61 * 600.0), 550)  # golden ratio
            self.models_input_dialogue.exec_()  # exec leaves the parent on hold

            if self.models_input_dialogue.grids_model is not None:
                self.models_input_dialogue.process(main_grid=self.circuit)

                # set up sliders
                self.set_up_profile_sliders()
                self.update_date_dependent_combos()
                self.display_profiles()

        else:
            warning_msg("You need to declare a time profile first.\n\n"
                        "Then, this button will show the dialogue to\n"
                        "load the data from the models at the time steps\n"
                        "that you prefer.\n\n"
                        "Use the 'Create profiles button'.")

    def import_contingencies(self):
        """
        Open file to import contingencies file
        """

        files_types = "Formats (*.json)"

        # call dialog to select the file

        filenames, type_selected = QtWidgets.QFileDialog.getOpenFileNames(parent=self,
                                                                          caption='Open file',
                                                                          dir=self.project_directory,
                                                                          filter=files_types)

        if len(filenames) == 1:
            contingencies = import_contingencies_from_json(filenames[0])
            logger = self.circuit.set_contingencies(contingencies=contingencies)

            if len(logger) > 0:
                dlg = LogsDialogue('Contingencies import', logger)
                dlg.exec_()

    def initialize_contingencies(self):
        """
        Launch the contingency planner to initialize the contingencies
        :return:
        """
        self.contingency_planner_dialogue = ContingencyPlannerGUI(parent=self, grid=self.circuit)
        # self.contingency_planner_dialogue.resize(int(1.61 * 600.0), 550)  # golden ratio
        self.contingency_planner_dialogue.exec_()

        # gather results
        if self.contingency_planner_dialogue.generated_results:
            self.circuit.contingency_groups = self.contingency_planner_dialogue.contingency_groups
            self.circuit.contingencies = self.contingency_planner_dialogue.contingencies

    def add_selected_to_contingency(self):
        """
        Add contingencies from the schematic selection
        """
        if len(self.circuit.buses) > 0:

            # get the selected buses
            selected = self.get_selected_contingency_devices()

            if len(selected) > 0:
                group = dev.ContingencyGroup(idtag=None,
                                             name="Contingency " + str(len(self.circuit.contingency_groups)),
                                             category="single" if len(selected) == 1 else "multiple")
                self.circuit.add_contingency_group(group)

                for elm in selected:
                    con = dev.Contingency(device_idtag=elm.idtag,
                                          code=elm.code,
                                          name=elm.name,
                                          prop="active",
                                          value=0,
                                          group=group)
                    self.circuit.add_contingency(con)
            else:
                info_msg("Select some elements in the schematic first", "Add selected to contingency")

    def export_contingencies(self):
        """
        Export contingencies
        :return:
        """
        if len(self.circuit.contingencies) > 0:

            # declare the allowed file types
            files_types = "JSON file (*.json)"

            # call dialog to select the file
            filename, type_selected = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '', files_types)

            if not (filename.endswith('.json')):
                filename += ".json"

            if filename != "":
                # save file
                export_contingencies_json_file(circuit=self.circuit, file_path=filename)

    def add_selected_to_investment(self):
        """
        Add contingencies from the schematic selection
        """
        if len(self.circuit.buses) > 0:

            # get the selected buses
            selected = self.get_selected_investment_devices()

            if len(selected) > 0:
                group = dev.InvestmentsGroup(idtag=None,
                                             name="Investment " + str(len(self.circuit.contingency_groups)),
                                             category="single" if len(selected) == 1 else "multiple")
                self.circuit.add_investments_group(group)

                for elm in selected:
                    con = dev.Investment(device_idtag=elm.idtag,
                                         code=elm.code,
                                         name=elm.name,
                                         CAPEX=0.0,
                                         OPEX=0.0,
                                         group=group)
                    self.circuit.add_investment(con)
            else:
                info_msg("Select some elements in the schematic first", "Add selected to investment")

    def colour_map(self):
        """
        Draw lines
        :return:
        """
        current_study = self.ui.available_results_to_color_map_comboBox.currentText()

        if current_study != '':
            poly = self.grid_colour_function(plot_function=viz.get_map_polylines,
                                             current_study=current_study,
                                             current_step=self.ui.map_time_horizontalSlider.value())

            # # delete the previous layer
            # self.map_widget.DeleteLayer(self.polyline_layer_id)
            #
            # # draw again
            # self.polyline_layer_id = self.map_widget.AddPolylineLayer(data=poly,
            #                                                           map_rel=True,
            #                                                           visible=True,
            #                                                           delta=40,
            #                                                           show_levels=list(range(15)),
            #                                                           # levels at which to show the polylines
            #                                                           name='<polyline_layer>')

            self.map_widget.getLayer(self.polyline_layer_id).data = poly
            self.map_widget.update()

            # self.map_widget.setLayerSelectable(self.polyline_layer_id, True)

    def config_file_path(self) -> str:
        """

        :return:
        """
        return os.path.join(get_create_gridcal_folder(), 'gui_config.json')

    def config_file_exists(self) -> bool:
        """
        Check if the config file exists
        :return: True / False
        """
        return os.path.exists(self.config_file_path())

    def get_config_structure(self) -> Dict[str, Dict[str, any]]:
        """
        Get the settings configuration dictionary
        This serves to collect automatically the settings
        and apply the incomming setting automatically as well
        :return: Dict[name, Dict[name, QtWidget]
        """
        return {"graphics": {
            "dark_mode": self.ui.dark_mode_checkBox,
            "palette": self.ui.palette_comboBox,
            "min_node_size": self.ui.min_node_size_spinBox,
            "max_node_size": self.ui.max_node_size_spinBox,
            "min_branch_size": self.ui.min_branch_size_spinBox,
            "max_branch_size": self.ui.max_branch_size_spinBox,
            "width_based_flow": self.ui.branch_width_based_on_flow_checkBox,
            "map_tile_provider": self.ui.tile_provider_comboBox,
            "plotting_style": self.ui.plt_style_comboBox
            },
            "machine_learning": {
                "clustering": {
                    "cluster_number": self.ui.cluster_number_spinBox,
                },
                "node_grouping": {
                    "sigma": self.ui.node_distances_sigma_doubleSpinBox,
                    "n_elements": self.ui.node_distances_elements_spinBox,
                }
            },
            "linear": {
                "perturbance_power": self.ui.atcPerturbanceSpinBox,
                "transfer_sensitivity_threshold": self.ui.atcThresholdSpinBox,
                "transfer_method": self.ui.transferMethodComboBox,
                "Loading_threshold_to_report": self.ui.ntcReportLoadingThresholdSpinBox,
                "consider_contingencies": self.ui.n1ConsiderationCheckBox
            },
            "stochastic": {
                "method": self.ui.stochastic_pf_method_comboBox,
                "voltage_variance": self.ui.tolerance_stochastic_spinBox,
                "number_of_samples": self.ui.max_iterations_stochastic_spinBox
            },
            "cascading": {
                "additional_islands": self.ui.cascading_islands_spinBox
            },
            "power_flow": {
                "solver": self.ui.solver_comboBox,
                "retry": self.ui.helm_retry_checkBox,
                "distributed_slack": self.ui.distributed_slack_checkBox,
                "ignore_single_node_islands": self.ui.ignore_single_node_islands_checkBox,
                "automatic_precision": self.ui.auto_precision_checkBox,
                "use_voltage_guess": self.ui.use_voltage_guess_checkBox,
                "precision": self.ui.tolerance_spinBox,
                "acceleration": self.ui.muSpinBox,
                "max_iterations": self.ui.max_iterations_spinBox,
                "verbosity": self.ui.verbositySpinBox,
                "reactive_power_control_mode": self.ui.reactive_power_control_mode_comboBox,
                "transformer_taps_control_mode": self.ui.taps_control_mode_comboBox,
                "apply_temperature_correction": self.ui.temperature_correction_checkBox,
                "apply_impedance_tolerances": self.ui.apply_impedance_tolerances_checkBox,
                "override_branch_controls": self.ui.override_branch_controls_checkBox,
            },
            "optimal_power_flow": {
                "method": self.ui.lpf_solver_comboBox,
                "time_grouping": self.ui.opf_time_grouping_comboBox,
                "zone_grouping": self.ui.opfZonalGroupByComboBox,
                "mip_solver": self.ui.mip_solver_comboBox,
                "contingency_tolerance": self.ui.opfContingencyToleranceSpinBox,
                "max_module_step": self.ui.maxVoltageModuleStepSpinBox,
                "max_angle_step": self.ui.maxVoltageAngleStepSpinBox,
                "skip_generation_limits": self.ui.skipOpfGenerationLimitsCheckBox,
                "consider_contingencies": self.ui.considerContingenciesOpfCheckBox,
                "maximize_area_exchange": self.ui.opfMaximizeExcahngeCheckBox,
                "unit_commitment": self.ui.opfUnitCommitmentCheckBox,
            },
            "continuation_power_flow": {
                "max_iterations": self.ui.vs_max_iterations_spinBox,
                "stop_at": self.ui.vc_stop_at_comboBox,
                "increase_system_loading": self.ui.start_vs_from_default_radioButton,
                "lambda_factor": self.ui.alpha_doubleSpinBox,
                "points_from_time_series": self.ui.start_vs_from_selected_radioButton,
                "now": self.ui.vs_departure_comboBox,
                "target": self.ui.vs_target_comboBox,
                "available_transfer_capacity": self.ui.atcRadioButton,
            },
            "net_transfer_capacity": {
                "proportional_redispatch": self.ui.proportionalRedispatchRadioButton,
                "optimal_redispatch": self.ui.optimalRedispatchRadioButton,
                "skip_generation_limits": self.ui.skipNtcGenerationLimitsCheckBox,
                "dispatch_all_areas": self.ui.ntcDispatchAllAreasCheckBox,
                "check_feasibility": self.ui.ntcFeasibilityCheckCheckBox,

                "slack_variables_tolerance": self.ui.ntcOpfTolSpinBox,
                "transmission_reliability_margin": self.ui.trmSpinBox,

                "branch_exchange_sensitivity": self.ui.ntcAlphaSpinBox,
                "branch_rating_contribution": self.ui.ntcLoadRuleSpinBox,

                "consider_branch_contingencies": self.ui.considerContingenciesNtcOpfCheckBox,
                "consider_hvdc_contingencies": self.ui.considerContingenciesHvdcOpfCheckBox,
                "consider_generator_contingencies": self.ui.considerContingenciesGeneratorOpfCheckBox,
                "generator_contingency_power": self.ui.contingencyGenerationThresholdDoubleSpinBox,

                "power_shift_weight": self.ui.weightPowerShiftSpinBox,
                "generation_cost_weight": self.ui.weightGenCostSpinBox,
                "branch_overload_weight": self.ui.weightsOverloadsSpinBox,
            },
            "general": {
                "base_power": self.ui.sbase_doubleSpinBox,
                "frequency": self.ui.fbase_doubleSpinBox,
                "engine": self.ui.engineComboBox
            },
            "contingencies": {
                "contingencies_engine": self.ui.contingencyEngineComboBox
            },
            "file": {
                "store_results_in_file": self.ui.saveResultsCheckBox
            }
        }

    def get_gui_config_data(self) -> Dict[str, Dict[str, Union[float, int, str, bool]]]:
        """
        Get a dictionary with the GUI configuration data
        :return:
        """

        def struct_to_data(data_: Dict[str, Dict[str, Union[float, int, str, bool, Dict]]],
                           struct_: Dict[str, Dict[str, any]]):
            """
            Recursive function to get the config dictionary from the GUI values
            :param data_: Dictionary to fill
            :param struct_: result of self.get_config_structure()
            """
            for key, value in struct_.items():
                if isinstance(value, dict):
                    data_[key] = dict()
                    struct_to_data(data_[key], value)
                elif isinstance(value, QtWidgets.QComboBox):
                    data_[key] = value.currentText()
                elif isinstance(value, QtWidgets.QDoubleSpinBox):
                    data_[key] = value.value()
                elif isinstance(value, QtWidgets.QSpinBox):
                    data_[key] = value.value()
                elif isinstance(value, QtWidgets.QCheckBox):
                    data_[key] = value.isChecked()
                elif isinstance(value, QtWidgets.QRadioButton):
                    data_[key] = value.isChecked()
                else:
                    raise Exception('unknown structure')

        struct = self.get_config_structure()
        data = dict()
        struct_to_data(data, struct)

        return data

    def save_gui_config(self):
        """
        Save the GUI configuration
        :return:
        """
        data = self.get_gui_config_data()
        with open(self.config_file_path(), "w") as f:
            f.write(json.dumps(data, indent=4))

    def apply_gui_config(self, data: dict):
        """
        Apply GUI configuration dictionary
        :param data: GUI configuration dictionary
        """

        def data_to_struct(data_, struct_):
            """
            Recursive function to set the GUI values from the config dictionary
            :param data_: config dictionary with values from the file
            :param struct_: result of self.get_config_structure()
            """
            for key, instance in struct_.items():
                if key in data_:
                    if isinstance(instance, dict):
                        data_to_struct(data_[key], instance)
                    elif isinstance(instance, QtWidgets.QComboBox):
                        val = data_[key]
                        index = instance.findText(val)
                        if -1 < index < instance.count():
                            instance.setCurrentIndex(index)
                    elif isinstance(instance, QtWidgets.QDoubleSpinBox):
                        instance.setValue(float(data_[key]))
                    elif isinstance(instance, QtWidgets.QSpinBox):
                        instance.setValue(int(data_[key]))
                    elif isinstance(instance, QtWidgets.QCheckBox):
                        instance.setChecked(bool(data_[key]))
                    elif isinstance(instance, QtWidgets.QRadioButton):
                        instance.setChecked(bool(data_[key]))
                    else:
                        raise Exception('unknown structure')
                else:
                    print(key)

        struct = self.get_config_structure()
        data_to_struct(data_=data, struct_=struct)

    def load_gui_config(self):
        """
        Load GUI configuration from the local user folder
        """
        if self.config_file_exists():
            with open(self.config_file_path(), "r") as f:
                data = json.load(f)
                self.apply_gui_config(data=data)
                self.change_theme_mode()

    def expand_object_tree_nodes(self):

        proxy = self.ui.dataStructuresTreeView.model()

        for row in range(proxy.rowCount()):
            index = proxy.index(row, 0)
            self.ui.dataStructuresTreeView.expand(index)


def run():
    """
    Main function to run the GUI
    :return:
    """
    qdarktheme.enable_hi_dpi()
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')  # ['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion']

    # Apply the complete dark theme to your Qt App.
    qdarktheme.setup_theme(
        theme='auto',
        custom_colors={
            "primary": "#00aa88ff",
            "primary>list.selectionBackground": "#00aa88be"
        }
    )

    window_ = MainGUI()

    h_ = 780
    window_.resize(int(1.61 * h_), h_)  # golden ratio :)
    window_.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()

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
from GridCalEngine.IO.cim.cgmes_2_4_15.devices.branches.transformer.phase_tap_changer import PhaseTapChanger
from GridCalEngine.IO.cim.cgmes_2_4_15.devices.branches.transformer.phase_tap_changer_table import PhaseTapChangerTable
from GridCalEngine.IO.base.units import UnitMultiplier, UnitSymbol
from GridCalEngine.IO.cim.cgmes_2_4_15.cgmes_enums import cgmesProfile


class PhaseTapChangerTabular(PhaseTapChanger):
    """
    PhaseTapChangerTabular
    """

    def __init__(self, rdfid, tpe):
        PhaseTapChanger.__init__(self, rdfid, tpe)

        self.PhaseTapChangerTable: PhaseTapChangerTable = None

        self.register_property(name='PhaseTapChangerTable',
                               class_type=PhaseTapChangerTable,
                               multiplier=UnitMultiplier.none,
                               unit=UnitSymbol.none,
                               description="The phase tap changer table for this phase tap changer.",
                               profiles=[cgmesProfile.EQ])

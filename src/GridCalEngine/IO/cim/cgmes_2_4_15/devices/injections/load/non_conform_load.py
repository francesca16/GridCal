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
from GridCalEngine.IO.cim.cgmes_2_4_15.devices.injections.load.non_conform_load_group import NonConformLoadGroup
from GridCalEngine.IO.cim.cgmes_2_4_15.cgmes_enums import cgmesProfile
from GridCalEngine.IO.cim.cgmes_2_4_15.devices.injections.load.energy_consumer import EnergyConsumer
from GridCalEngine.IO.base.units import UnitMultiplier, UnitSymbol


class NonConformLoad(EnergyConsumer):

    def __init__(self, rdfid, tpe):
        EnergyConsumer.__init__(self, rdfid, tpe)

        self.LoadGroup: NonConformLoadGroup | None = None

        self.register_property(name='LoadGroup',
                               class_type=NonConformLoadGroup,
                               multiplier=UnitMultiplier.none,
                               unit=UnitSymbol.none,
                               description="",
                               profiles=[cgmesProfile.EQ])

    def get_pq(self):
        return self.p, self.q

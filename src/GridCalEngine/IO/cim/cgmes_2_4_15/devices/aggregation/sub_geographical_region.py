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
from typing import Union
from GridCalEngine.IO.cim.cgmes_2_4_15.cgmes_enums import cgmesProfile
from GridCalEngine.IO.cim.cgmes_2_4_15.devices.aggregation.geographical_region import GeographicalRegion
from GridCalEngine.IO.cim.cgmes_2_4_15.devices.identified_object import IdentifiedObject
from GridCalEngine.IO.base.units import UnitMultiplier, UnitSymbol
from GridCalEngine.IO.cim.cgmes_2_4_15.devices.aggregation.modelling_authority import ModelingAuthority


class SubGeographicalRegion(IdentifiedObject):

    def __init__(self, rdfid, tpe):
        IdentifiedObject.__init__(self, rdfid, tpe)

        self.Region: GeographicalRegion | None = None
        self.ModelingAuthority: Union[ModelingAuthority, None] = None
        self.sourcingActor: str = ''
        self.masUri: str = ''

        self.register_property(name='Region',
                               class_type=GeographicalRegion,
                               multiplier=UnitMultiplier.none,
                               unit=UnitSymbol.none,
                               description="The geographical region to which this "
                                           "sub-geographical region is within.",
                               profiles=[cgmesProfile.EQ, cgmesProfile.EQ_BD])

        self.register_property(name='ModelingAuthority',
                               class_type=ModelingAuthority,
                               multiplier=UnitMultiplier.none,
                               unit=UnitSymbol.none,
                               description="",
                               profiles=[cgmesProfile.EQ])
        self.register_property(name='sourcingActor',
                               class_type=str,
                               multiplier=UnitMultiplier.none,
                               unit=UnitSymbol.none,
                               description="",
                               profiles=[cgmesProfile.EQ])
        self.register_property(name='masUri',
                               class_type=str,
                               multiplier=UnitMultiplier.none,
                               unit=UnitSymbol.none,
                               description="",
                               profiles=[cgmesProfile.EQ])

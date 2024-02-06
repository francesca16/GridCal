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
from GridCalEngine.IO.base.units import UnitMultiplier, UnitSymbol
from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.identified_object import IdentifiedObject
from GridCalEngine.IO.cim.cgmes.cgmes_enums import cgmesProfile


class ACDCTerminal(IdentifiedObject):
	def __init__(self, rdfid='', tpe='ACDCTerminal'):
		IdentifiedObject.__init__(self, rdfid, tpe)

		self.sequenceNumber: int = 0
		from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.operational_limit_set import OperationalLimitSet
		self.OperationalLimitSet: OperationalLimitSet | None = None
		from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.bus_name_marker import BusNameMarker
		self.BusNameMarker: BusNameMarker | None = None
		self.connected: bool = False

		self.register_property(
			name='sequenceNumber',
			class_type=int,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''The orientation of the terminal connections for a multiple terminal conducting equipment.  The sequence numbering starts with 1 and additional terminals should follow in increasing order.   The first terminal is the "starting point" for a two terminal branch.''',
			profiles=[]
		)
		self.register_property(
			name='OperationalLimitSet',
			class_type=OperationalLimitSet,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''The operational limit sets at the terminal.''',
			profiles=[]
		)
		self.register_property(
			name='BusNameMarker',
			class_type=BusNameMarker,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''The bus name marker used to name the bus (topological node).''',
			profiles=[]
		)
		self.register_property(
			name='connected',
			class_type=bool,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''The connected status is related to a bus-branch model and the topological node to terminal relation.  True implies the terminal is connected to the related topological node and false implies it is not. 
In a bus-branch model, the connected status is used to tell if equipment is disconnected without having to change the connectivity described by the topological node to terminal relation. A valid case is that conducting equipment can be connected in one end and open in the other. In particular for an AC line segment, where the reactive line charging can be significant, this is a relevant case.''',
			profiles=[]
		)

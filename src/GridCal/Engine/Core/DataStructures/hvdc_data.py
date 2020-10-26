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
import numpy as np
import scipy.sparse as sp
import GridCal.Engine.Core.topology as tp


class HvdcData:

    def __init__(self, nhvdc, nbus, ntime=1):
        """

        :param nhvdc:
        :param nbus:
        """
        self.nhvdc = nhvdc
        self.ntime = ntime

        self.hvdc_names = np.zeros(nhvdc, dtype=object)
        self.hvdc_active = np.zeros(nhvdc, dtype=bool)

        self.hvdc_loss_factor = np.zeros(nhvdc)

        self.hvdc_rate = np.zeros((nhvdc, ntime))
        self.hvdc_Pf = np.zeros((nhvdc, ntime))
        self.hvdc_Pt = np.zeros((nhvdc, ntime))
        self.hvdc_Vset_f = np.zeros((nhvdc, ntime))
        self.hvdc_Vset_t = np.zeros((nhvdc, ntime))

        self.hvdc_Qmin_f = np.zeros(nhvdc)
        self.hvdc_Qmax_f = np.zeros(nhvdc)
        self.hvdc_Qmin_t = np.zeros(nhvdc)
        self.hvdc_Qmax_t = np.zeros(nhvdc)

        self.C_hvdc_bus_f = sp.lil_matrix((nhvdc, nbus), dtype=int)  # this ons is just for splitting islands
        self.C_hvdc_bus_t = sp.lil_matrix((nhvdc, nbus), dtype=int)  # this ons is just for splitting islands

    def slice(self, elm_idx, bus_idx, time_idx=None):
        """

        :param elm_idx:
        :param bus_idx:
        :param time_idx:
        :return:
        """

        if time_idx is None:
            tidx = elm_idx
        else:
            tidx = np.ix_(elm_idx, time_idx)

        data = HvdcData(nhvdc=len(elm_idx), nbus=len(bus_idx))

        data.hvdc_names = self.hvdc_names[elm_idx]
        data.hvdc_active = self.hvdc_active[elm_idx]

        data.hvdc_rate = self.hvdc_rate[tidx]
        data.hvdc_Pf = self.hvdc_Pf[tidx]
        data.hvdc_Pt = self.hvdc_Pt[tidx]
        data.hvdc_Vset_f = self.hvdc_Vset_f[tidx]
        data.hvdc_Vset_t = self.hvdc_Vset_t[tidx]

        data.hvdc_loss_factor = self.hvdc_loss_factor[elm_idx]
        data.hvdc_Qmin_f = self.hvdc_Qmin_f[elm_idx]
        data.hvdc_Qmax_f = self.hvdc_Qmax_f[elm_idx]
        data.hvdc_Qmin_t = self.hvdc_Qmin_t[elm_idx]
        data.hvdc_Qmax_t = self.hvdc_Qmax_t[elm_idx]

        data.C_hvdc_bus_f = self.C_hvdc_bus_f[np.ix_(elm_idx, bus_idx)]
        data.C_hvdc_bus_t = self.C_hvdc_bus_t[np.ix_(elm_idx, bus_idx)]

        return data

    def get_island(self, bus_idx):
        """
        Get HVDC indices of the island given by the bus indices
        :param bus_idx: list of bus indices
        :return: list of HVDC lines indices
        """
        return tp.get_elements_of_the_island(self.C_hvdc_bus_f + self.C_hvdc_bus_t, bus_idx)

    def get_injections_per_bus(self):
        F = (self.hvdc_active * self.hvdc_Pf) * self.C_hvdc_bus_f
        T = (self.hvdc_active * self.hvdc_Pt) * self.C_hvdc_bus_t
        return F + T

    def get_qmax_from_per_bus(self):
        return (self.hvdc_Qmax_f * self.hvdc_active) * self.C_hvdc_bus_f

    def get_qmin_from_per_bus(self):
        return (self.hvdc_Qmin_f * self.hvdc_active) * self.C_hvdc_bus_f

    def get_qmax_to_per_bus(self):
        return (self.hvdc_Qmax_t * self.hvdc_active) * self.C_hvdc_bus_t

    def get_qmin_to_per_bus(self):
        return (self.hvdc_Qmin_t * self.hvdc_active) * self.C_hvdc_bus_t

    def get_loading(self):
        return self.hvdc_Pf / self.hvdc_rate

    def get_losses(self):
        return (self.hvdc_Pf.T * self.hvdc_loss_factor).T

    def __len__(self):
        return self.nhvdc
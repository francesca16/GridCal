import os
import GridCalEngine.api as gce
from GridCalEngine.Core.DataStructures.numerical_circuit import compile_numerical_circuit_at
from GridCalEngine.Simulations.OPF.NumericalMethods.ac_opf import run_nonlinear_opf, ac_optimal_power_flow
from GridCalEngine.enumerations import TransformerControlType
from scipy.sparse import csc_matrix as csc
from scipy import sparse as sp
import numpy as np

def example_3bus_acopf():
    """

    :return:
    """

    grid = gce.MultiCircuit()

    b1 = gce.Bus(is_slack=True)
    b2 = gce.Bus()
    b3 = gce.Bus()

    grid.add_bus(b1)
    grid.add_bus(b2)
    grid.add_bus(b3)

    # grid.add_line(gce.Line(bus_from=b1, bus_to=b2, name='line 1-2', r=0.001, x=0.05, rate=100))
    grid.add_line(gce.Line(bus_from=b2, bus_to=b3, name='line 2-3', r=0.001, x=0.05, rate=100))
    # grid.add_line(gce.Line(bus_from=b3, bus_to=b1, name='line 3-1_1', r=0.001, x=0.05, rate=100))
    # grid.add_line(Line(bus_from=b3, bus_to=b1, name='line 3-1_2', r=0.001, x=0.05, rate=100))

    grid.add_load(b3, gce.Load(name='L3', P=50, Q=20))
    grid.add_generator(b1, gce.Generator('G1', vset=1.00, Cost=1.0, Cost2=2.0))
    grid.add_generator(b2, gce.Generator('G2', P=10, vset=0.995, Cost=1.0, Cost2=3.0))

    tr1 = gce.Transformer2W(b1, b2, 'Trafo1', control_mode=TransformerControlType.Pf,
                            tap_module=1.1, tap_phase=0.02, r=0.001, x=0.05)
    grid.add_transformer2w(tr1)

    tr2 = gce.Transformer2W(b3, b1, 'Trafo1', control_mode=TransformerControlType.PtQt,
                            tap_module=1.05, tap_phase=-0.02, r=0.001, x=0.05)
    grid.add_transformer2w(tr2)

    nc = compile_numerical_circuit_at(circuit=grid)

    A, B, C, D, E, F = compute_analytic_admittances(nc)

    A_, B_, C_, D_, E_, F_ = compute_finitediff_admittances(nc)
    options = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
    power_flow = gce.PowerFlowDriver(grid, options)
    power_flow.run()

    # print('\n\n', grid.name)
    # print('\tConv:\n', power_flow.results.get_bus_df())
    # print('\tConv:\n', power_flow.results.get_branch_df())

    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=3)
    run_nonlinear_opf(grid=grid, pf_options=pf_options, plot_error=True)


def compute_analytic_admittances(nc):
    tapm_lines = np.r_[nc.k_qf_m, nc.k_qt_m, nc.k_vt_m]
    tapm = nc.branch_data.tap_module
    tapt_lines = nc.k_pf_tau
    tapt = nc.branch_data.tap_angle

    F = nc.branch_data.F
    T = nc.branch_data.T
    Cf = nc.Cf
    Ct = nc.Ct
    ys = 1.0 / (nc.branch_data.R + 1.0j * nc.branch_data.X + 1e-20)

    admittance = nc.compute_admittance()
    Ybus = admittance.Ybus
    M, N = Cf.shape

    dYfdm = []
    dYtdm = []
    dYbusdm = []

    dYfdt = []
    dYtdt = []
    dYbusdt = []

    for l, line in enumerate(tapm_lines):
        i = F[line]
        j = T[line]
        mp = tapm[line]
        tau = tapt[line]
        ylin = ys[line]

        dYffdm = np.zeros(M, dtype=complex)
        dYftdm = np.zeros(M, dtype=complex)
        dYtfdm = np.zeros(M, dtype=complex)
        dYttdm = np.zeros(M, dtype=complex)

        dYffdm[line] = -2 * ylin / (mp * mp * mp)
        dYftdm[line] = ylin / (mp * mp * np.exp(-1.0j * tau))
        dYtfdm[line] = ylin / (mp * mp * np.exp(1.0j * tau))
        dYttdm[line] = 0

        dYfdm.append(sp.diags(dYffdm) * Cf + sp.diags(dYftdm) * Ct)
        dYtdm.append(sp.diags(dYtfdm) * Cf + sp.diags(dYttdm) * Ct)

        dYbusdm.append(Cf.T * dYfdm[l] + Ct.T * dYtdm[l])

    for l, line in enumerate(tapt_lines):
        i = F[line]
        j = T[line]
        mp = tapm[line]
        tau = tapt[line]
        ylin = ys[line]

        dYffdt = np.zeros(M, dtype=complex)
        dYftdt = np.zeros(M, dtype=complex)
        dYtfdt = np.zeros(M, dtype=complex)
        dYttdt = np.zeros(M, dtype=complex)

        dYffdt[line] = 0
        dYftdt[line] = -1j * ylin / (mp * np.exp(-1.0j * tau))
        dYtfdt[line] = 1j * ylin / (mp * np.exp(1.0j * tau))
        dYttdt[line] = 0

        dYfdt.append(sp.diags(dYffdt) * Cf + sp.diags(dYftdt) * Ct)
        dYtdt.append(sp.diags(dYtfdt) * Cf + sp.diags(dYttdt) * Ct)

        dYbusdt.append(Cf.T * dYfdt[l] + Ct.T * dYtdt[l])


    return dYbusdm, dYfdm, dYtdm, dYbusdt, dYfdt, dYtdt


def compute_finitediff_admittances(nc, tol=1e-6):

    tapm_lines = np.r_[nc.k_qf_m, nc.k_qt_m, nc.k_vt_m]
    tapt_lines = nc.k_pf_tau

    Ybus0 = nc.Ybus
    Yf0 = nc.Yf
    Yt0 = nc.Yt

    dYfdm = []
    dYtdm = []
    dYbusdm = []

    dYfdt = []
    dYtdt = []
    dYbusdt = []

    for l in tapm_lines:
        nc.branch_data.tap_module[l] += tol
        nc.reset_calculations()

        dYfdm.append((nc.Yf - Yf0) / tol)
        dYtdm.append((nc.Yt - Yt0) / tol)
        dYbusdm.append((nc.Ybus - Ybus0) / tol)

        nc.branch_data.tap_module[l] -= tol

    for l in tapt_lines:
        nc.branch_data.tap_angle[l] += tol
        nc.reset_calculations()

        dYfdt.append((nc.Yf - Yf0) / tol)
        dYtdt.append((nc.Yt - Yt0) / tol)
        dYbusdt.append((nc.Ybus - Ybus0) / tol)

        nc.branch_data.tap_angle[l] -= tol

    return dYbusdm, dYfdm, dYtdm, dYbusdt, dYfdt, dYtdt


def compute_analytic_admittances_2dev(nc):

    k_m = np.r_[nc.k_qf_m, nc.k_qt_m, nc.k_vt_m]
    tapm = nc.branch_data.tap_module
    k_tau = nc.k_pf_tau
    tapt = nc.branch_data.tap_angle

    F = nc.branch_data.F
    T = nc.branch_data.T
    Cf = nc.Cf
    Ct = nc.Ct
    ys = 1.0 / (nc.branch_data.R + 1.0j * nc.branch_data.X + 1e-20)

    admittance = nc.compute_admittance()
    Ybus = admittance.Ybus
    M, N = Cf.shape

    for l, line in enumerate(tapm_lines):
        i = F[line]
        j = T[line]
        mp = tapm[line]
        tau = tapt[line]
        ylin = ys[line]

        dYffdmdm = np.zeros(M, dtype=complex)
        dYftdmdm = np.zeros(M, dtype=complex)
        dYtfdmdm = np.zeros(M, dtype=complex)
        dYttdmdm = np.zeros(M, dtype=complex)

        dYfdmdm.append(sp.diags(dYffdmdm) * Cf + sp.diags(dYftdmdm) * Ct)
        dYtdmdm.append(sp.diags(dYtfdmdm) * Cf + sp.diags(dYttdmdm) * Ct)

        dYbusdmdm.append(Cf.T * dYfdmdm[l] + Ct.T * dYtdmdm[l])

        if line in tapt_lines:
            dYffdmdt = np.zeros(M, dtype=complex)
            dYftdmdt = np.zeros(M, dtype=complex)
            dYtfdmdt = np.zeros(M, dtype=complex)
            dYttdmdt = np.zeros(M, dtype=complex)

            dYffdmdt[line] = 0
            dYftdmdt[line] = 1j * ylin / (mp * mp * np.exp(-1.0j * tau))
            dYtfdmdt[line] = -1j * ylin / (mp * mp * np.exp(1.0j * tau))
            dYttdmdt[line] = 0

            dYfdmdt.append(sp.diags(dYffdmdt) * Cf + sp.diags(dYftdmdt) * Ct)
            dYtdmdt.append(sp.diags(dYtfdmdt) * Cf + sp.diags(dYttdmdt) * Ct)

            dYbusdmdt.append(Cf.T * dYfdmdt[l] + Ct.T * dYtdmdt[l])

            dYfdtdm.append((sp.diags(dYffdmdt) * Cf + sp.diags(dYftdmdt) * Ct).T)
            dYtdtdm.append((sp.diags(dYtfdmdt) * Cf + sp.diags(dYttdmdt) * Ct).T)

            dYbusdtdm.append((Cf.T * dYfdmdt[l] + Ct.T * dYtdmdt[l]).T)

    for l, line in enumerate(tapt_lines):
        i = F[line]
        j = T[line]
        mp = tapm[line]
        tau = tapt[line]
        ylin = ys[line]

        dYffdtdt = np.zeros(M, dtype=complex)
        dYftdtdt = np.zeros(M, dtype=complex)
        dYtfdtdt = np.zeros(M, dtype=complex)
        dYttdtdt = np.zeros(M, dtype=complex)

        dYffdtdt[line] = 0
        dYftdtdt[line] = ylin / (mp * np.exp(-1.0j * tau))
        dYtfdtdt[line] = ylin / (mp * np.exp(1.0j * tau))
        dYttdtdt[line] = 0

        dYfdtdt.append(sp.diags(dYffdtdt) * Cf + sp.diags(dYftdtdt) * Ct)
        dYtdtdt.append(sp.diags(dYtfdtdt) * Cf + sp.diags(dYttdtdt) * Ct)

        dYbusdtdt.append(Cf.T * dYfdtdt[l] + Ct.T * dYtdtdt[l])


    # Second partial derivative with respect to tap module
    mp = tapm[k_m]
    tau = tapt[k_m]
    ylin = ys[k_m]

    Cf_m = nc.Cf[k_m, :]
    Ct_m = nc.Ct[k_m, :]

    dYffdmdm = 6 * ylin / (mp * mp * mp * mp)
    dYftdmdm = -2 * ylin / (mp * mp * mp * np.exp(-1.0j * tau))
    dYtfdmdm = -2 * ylin / (mp * mp * mp * np.exp(1.0j * tau))
    dYttdmdm = 0

    dYfdmdm = (sp.diags(dYffdmdm) * Cf_m + sp.diags(dYftdmdm) * Ct_m)
    dYtdmdm = (sp.diags(dYtfdmdm) * Cf_m + sp.diags(dYttdmdm) * Ct_m)

    dYbusdmdm = (Cf_m.T * dYfdmdm+ Ct_m.T * dYtdmdm)

    # Second partial derivative with respect to tap module
    mp = tapm[k_m]
    tau = tapt[k_m]
    ylin = ys[k_m]

    Cf_m = nc.Cf[k_m, :]
    Ct_m = nc.Ct[k_m, :]

    dYffdmdm = 6 * ylin / (mp * mp * mp * mp)
    dYftdmdm = -2 * ylin / (mp * mp * mp * np.exp(-1.0j * tau))
    dYtfdmdm = -2 * ylin / (mp * mp * mp * np.exp(1.0j * tau))
    dYttdmdm = 0

    dYfdmdm = (sp.diags(dYffdmdm) * Cf_m + sp.diags(dYftdmdm) * Ct_m)
    dYtdmdm = (sp.diags(dYtfdmdm) * Cf_m + sp.diags(dYttdmdm) * Ct_m)

    dYbusdmdm = (Cf_m.T * dYfdmdm + Ct_m.T * dYtdmdm)

    # Second partial derivative with respect to tap module
    mp = tapm[k_m]
    tau = tapt[k_m]
    ylin = ys[k_m]

    Cf_m = nc.Cf[k_m, :]
    Ct_m = nc.Ct[k_m, :]

    dYffdmdt[line] = 0
    dYftdmdt[line] = 1j * ylin / (mp * mp * np.exp(-1.0j * tau))
    dYtfdmdt[line] = -1j * ylin / (mp * mp * np.exp(1.0j * tau))
    dYttdmdt[line] = 0

    dYfdmdm = (sp.diags(dYffdmdm) * Cf_m + sp.diags(dYftdmdm) * Ct_m)
    dYtdmdm = (sp.diags(dYtfdmdm) * Cf_m + sp.diags(dYttdmdm) * Ct_m)

    dYbusdmdm = (Cf_m.T * dYfdmdm[l] + Ct_m.T * dYtdmdm[l])




    dYfdmdm = []
    dYtdmdm = []
    dYbusdmdm = []

    dYfdmdt = []
    dYtdmdt = []
    dYbusdmdt = []

    dYfdtdm = []
    dYtdtdm = []
    dYbusdtdm = []

    dYfdtdt = []
    dYtdtdt = []
    dYbusdtdt = []

    dYfdtdm = dYfdmdt.T
    dYtdtdm = dYtdmdt.T
    dYbusdtdm = dYbusdmdt.T

    return (dYbusdmdm, dYfdmdm, dYtdmdm, dYbusdmdt, dYfdmdt, dYtdmdt,
            dYbusdtdm, dYfdtdm, dYtdtdm, dYbusdtdt, dYfdtdt, dYtdtdt)



def compute_analytic_admittances_2dev(nc):

    k_m = np.r_[nc.k_qf_m, nc.k_qt_m, nc.k_vt_m]
    tapm = nc.branch_data.tap_module
    k_tau = nc.k_pf_tau
    tapt = nc.branch_data.tap_angle

    F = nc.branch_data.F
    T = nc.branch_data.T
    Cf = nc.Cf
    Ct = nc.Ct
    ys = 1.0 / (nc.branch_data.R + 1.0j * nc.branch_data.X + 1e-20)

    admittance = nc.compute_admittance()
    Ybus = admittance.Ybus
    M, N = Cf.shape

    dYfdmdm = []
    dYtdmdm = []
    dYbusdmdm = []

    dYfdmdt = []
    dYtdmdt = []
    dYbusdmdt = []

    dYfdtdm = []
    dYtdtdm = []
    dYbusdtdm = []

    dYfdtdt = []
    dYtdtdt = []
    dYbusdtdt = []

    for l, line in enumerate(tapm_lines):
        i = F[line]
        j = T[line]
        mp = tapm[line]
        tau = tapt[line]
        ylin = ys[line]

        dYffdmdm = np.zeros(M, dtype=complex)
        dYftdmdm = np.zeros(M, dtype=complex)
        dYtfdmdm = np.zeros(M, dtype=complex)
        dYttdmdm = np.zeros(M, dtype=complex)

        dYffdmdm[line] = 6 * ylin / (mp * mp * mp * mp)
        dYftdmdm[line] = -2 * ylin / (mp * mp * mp * np.exp(-1.0j * tau))
        dYtfdmdm[line] = -2 * ylin / (mp * mp * mp * np.exp(1.0j * tau))
        dYttdmdm[line] = 0

        dYfdmdm.append(sp.diags(dYffdmdm) * Cf + sp.diags(dYftdmdm) * Ct)
        dYtdmdm.append(sp.diags(dYtfdmdm) * Cf + sp.diags(dYttdmdm) * Ct)

        dYbusdmdm.append(Cf.T * dYfdmdm[l] + Ct.T * dYtdmdm[l])

        if line in tapt_lines:
            dYffdmdt = np.zeros(M, dtype=complex)
            dYftdmdt = np.zeros(M, dtype=complex)
            dYtfdmdt = np.zeros(M, dtype=complex)
            dYttdmdt = np.zeros(M, dtype=complex)

            dYffdmdt[line] = 0
            dYftdmdt[line] = 1j * ylin / (mp * mp * np.exp(-1.0j * tau))
            dYtfdmdt[line] = -1j * ylin / (mp * mp * np.exp(1.0j * tau))
            dYttdmdt[line] = 0

            dYfdmdt.append(sp.diags(dYffdmdt) * Cf + sp.diags(dYftdmdt) * Ct)
            dYtdmdt.append(sp.diags(dYtfdmdt) * Cf + sp.diags(dYttdmdt) * Ct)

            dYbusdmdt.append(Cf.T * dYfdmdt[l] + Ct.T * dYtdmdt[l])

            dYfdtdm.append((sp.diags(dYffdmdt) * Cf + sp.diags(dYftdmdt) * Ct).T)
            dYtdtdm.append((sp.diags(dYtfdmdt) * Cf + sp.diags(dYttdmdt) * Ct).T)

            dYbusdtdm.append((Cf.T * dYfdmdt[l] + Ct.T * dYtdmdt[l]).T)

    for l, line in enumerate(tapt_lines):
        i = F[line]
        j = T[line]
        mp = tapm[line]
        tau = tapt[line]
        ylin = ys[line]

        dYffdtdt = np.zeros(M, dtype=complex)
        dYftdtdt = np.zeros(M, dtype=complex)
        dYtfdtdt = np.zeros(M, dtype=complex)
        dYttdtdt = np.zeros(M, dtype=complex)

        dYffdtdt[line] = 0
        dYftdtdt[line] = ylin / (mp * np.exp(-1.0j * tau))
        dYtfdtdt[line] = ylin / (mp * np.exp(1.0j * tau))
        dYttdtdt[line] = 0

        dYfdtdt.append(sp.diags(dYffdtdt) * Cf + sp.diags(dYftdtdt) * Ct)
        dYtdtdt.append(sp.diags(dYtfdtdt) * Cf + sp.diags(dYttdtdt) * Ct)

        dYbusdtdt.append(Cf.T * dYfdtdt[l] + Ct.T * dYtdtdt[l])

    dYfdtdm = dYfdmdt.T
    dYtdtdm = dYtdmdt.T
    dYbusdtdm = dYbusdmdt.T

    return (dYbusdmdm, dYfdmdm, dYtdmdm, dYbusdmdt, dYfdmdt, dYtdmdt,
            dYbusdtdm, dYfdtdm, dYtdtdm, dYbusdtdt, dYfdtdt, dYtdtdt)



if __name__ == '__main__':
    example_3bus_acopf()
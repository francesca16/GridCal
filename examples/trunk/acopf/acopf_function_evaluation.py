from __future__ import annotations
from typing import Union, Dict, Tuple, TYPE_CHECKING
import math
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix as csc
from acopf_functions import *

from GridCalEngine.basic_structures import Vec, CxVec
from GridCalEngine.Utils.MIPS.mips import mips_solver, step_calculation

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from GridCalEngine.Core.DataStructures.numerical_circuit import NumericalCircuit


def x2var(x, nbus, ngen):
    a = 0
    b = nbus
    Vm = x[a: b]

    a = b
    b = a + nbus
    Va = x[a: b]

    a = b
    b = a + ngen
    Pg = x[a: b]

    a = b
    b = a + ngen
    Qg = x[a: b]

    return Vm, Va, Pg, Qg


def var2x(Vm, Va, Pg, Qg):
    return np.r_[Vm, Va, Pg, Qg]


def eval_f(x, Cg, nbus):

    ngen = Cg.shape[1]  # Check

    Vm, Va, Pg, Qg = x2var(x=x, nbus=nbus, ngen=ngen)

    fval = np.sum(Pg)

    return np.array([fval])


def eval_g(x, Ybus, Cg, Sd, nbus):
    """
    Evaluate the
    :param x:
    :param Ybus:
    :param Cg:
    :param Sd:
    :param nbus:
    :param nbr:
    :return:
    """
    ngen = Cg.shape[1]  # Check

    Vm, Va, Pg, Qg = x2var(x=x, nbus=nbus, ngen=ngen)

    V = Vm * np.exp(1j * Va)
    Scalc = V * np.conj(Ybus @ V)

    Sg = Pg + 1j * Qg
    dS = Scalc + Sd - (Cg @ Sg)

    gval = np.r_[dS.real, dS.imag]  # Check, may need slicing

    return gval


def eval_h(x, Yf, Yt, from_idx, to_idx, Cg, rates):
    """

    :param x:
    :param Yf:
    :param Yt:
    :param from_idx:
    :param to_idx:
    :param Cg:
    :param rates:
    :return:
    """
    nbr, nbus = Yf.shape
    ngen = Cg.shape[1]  # Check

    Vm, Va, Pg, Qg = x2var(x=x, nbus=nbus, ngen=ngen)

    V = Vm * np.exp(1j * Va)

    If = np.conj(Yf @ V)

    Sf = V[from_idx] * If
    St = V[to_idx] * np.conj(Yt @ V)

    # Incrementos de las variables.
    hval = np.r_[Sf.real - rates, St.real - rates]

    return hval


def calc_jacobian(func, x, arg=(), h=1e-5):
    """
    Compute the Jacobian matrix of `func` at `x` using finite differences.

    :param func: Vector-valued function (R^n -> R^m).
    :param x: Point at which to evaluate the Jacobian (numpy array).
    :param arg: Arguments to func
    :param h: Small step for finite difference.
    :return: Jacobian matrix as a numpy array.
    """
    nx = len(x)
    f0 = func(x, *arg)
    jac = np.zeros((len(f0), nx))

    for i in range(nx):
        x_plus_h = np.copy(x)
        x_plus_h[i] += h
        f_plus_h = func(x_plus_h, *arg)
        jac[:, i] = (f_plus_h - f0) / h

    return jac


def calc_hessian(func, x, multipliers, arg=(), h=1e-5):
    """
    Compute the Hessian matrix of `func` at `x` using finite differences.

    :param func: Scalar-valued function (R^n -> R).
    :param x: Point at which to evaluate the Hessian (numpy array).
    :param multipliers: array of lagrange multipliers (length of the return of func)
    :param arg: Arguments to func
    :param h: Small step for finite difference.
    :return: Hessian matrix as a numpy array.
    """
    n = len(x)
    n_inequalities = len(multipliers)  # For objective function, it will be passed as 1. The MULT will be 1 aswell.
    hessian = np.zeros((n, n))

    for eq_idx in range(n_inequalities):
        for i in range(n):
            for j in range(n):
                x_ijp = np.copy(x)
                x_ijp[i] += h
                x_ijp[j] += h
                f_ijp = func(x_ijp, *arg)[eq_idx]

                x_ijm = np.copy(x)
                x_ijm[i] += h
                x_ijm[j] -= h
                f_ijm = func(x_ijm, *arg)[eq_idx]

                x_jim = np.copy(x)
                x_jim[i] -= h
                x_jim[j] += h
                f_jim = func(x_jim, *arg)[eq_idx]

                x_jjm = np.copy(x)
                x_jjm[i] -= h
                x_jjm[j] -= h
                f_jjm = func(x_jjm, *arg)[eq_idx]

                hessian[i, j] += multipliers[eq_idx] * (f_ijp - f_ijm - f_jim + f_jjm) / (4 * h * h)

    return hessian


def evaluate_power_flow(x, LAMBDA, PI, nbus, Ybus, Yf, Cg, Sd, Yt, from_idx, to_idx, rates, h=1e-5):
    """

    :param x:
    :param LAMBDA: lagrange multipliers for the hessian
    :param PI: lagrange multipliers for the jacobian
    :param Ybus:
    :param Yf:
    :param Cg:
    :param Sd:
    :param pvpq:
    :param pq:
    :param Yt:
    :param from_idx:
    :param to_idx:
    :param rates:
    :param h:
    :return:
    """
    f = eval_f(x=x, Cg=Cg, nbus=nbus)
    G = eval_g(x=x, Ybus=Ybus, Cg=Cg, Sd=Sd, nbus=nbus)
    H = eval_h(x=x, Yf=Yf, Yt=Yt, from_idx=from_idx, to_idx=to_idx, Cg=Cg, rates=rates)

    fx = calc_jacobian(func=eval_f, x=x, arg=(Cg, nbus), h=h)
    Gx = calc_jacobian(func=eval_g, x=x, arg=(Ybus, Cg, Sd, nbus))
    Hx = calc_jacobian(func=eval_h, x=x, arg=(Yf, Yt, from_idx, to_idx, Cg, rates))

    fxx = calc_hessian(func=eval_f, x=x, arg=(Cg, nbus), multipliers=[1.0], h=h)
    Gxx = calc_hessian(func=eval_g, x=x, arg=(Ybus, Cg, Sd, nbus), multipliers=LAMBDA, h=h)
    Hxx = calc_hessian(func=eval_h, x=x, arg=(Yf, Yt, from_idx, to_idx, Cg, rates), multipliers=LAMBDA, h=h)

    return f, G, H, fx, Gx, Hx, fxx, Gxx, Hxx


def acopf(nc: NumericalCircuit):

    # gather inputs
    nbus = nc.nbus
    nbr = nc.nbr
    ngen = nc.ngen
    Ybus = nc.Ybus
    Yf = nc.Yf
    Yt = nc.Yt
    Cg = nc.generator_data.C_bus_elm
    Sd = nc.load_data.get_injections_per_bus()
    pvpq = np.r_[nc.pv, nc.pq]
    pq = nc.pq
    from_idx = nc.F
    to_idx = nc.T
    rates = nc.rates
    h = 1e-5

    # initialization vars
    x0 = var2x(Vm=np.abs(nc.Vbus),
               Va=np.angle(nc.Vbus),
               Pg=np.zeros(ngen),
               Qg=np.zeros(ngen))
    n_vars = len(x0)
    n_equalities = 2 * nbus
    n_inequalities = 2 * nbr

    x = mips_solver(x0=x0,
                    NV=n_vars,
                    NE=n_equalities,
                    NI=n_inequalities,
                    f_eval=evaluate_power_flow,
                    step_calculator=step_calculation,
                    gamma0=10,
                    max_iter=100,
                    args=(nbus, Ybus, Yf, Cg, Sd, Yt, from_idx, to_idx, rates, h),
                    verbose=1)

    Vm, Va, Pg, Qg = x2var(x=x, nbus=nbus, ngen=ngen)

    return Vm, Va, Pg, Qg


if __name__ == "__main__":
    import os
    import GridCalEngine.api as gce

    fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', 'IEEE 30 Bus with storage.xlsx')

    grid = gce.open_file(filename=fname)

    nc_ = gce.compile_numerical_circuit_at(circuit=grid, t_idx=None)

    Vm, Va, Pg, Qg = acopf(nc=nc_)

    print("Vm:", Vm)
    print("Va:", Va)
    print("Pg:", Pg)
    print("Qg:", Qg)

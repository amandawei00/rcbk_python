import numpy as np
import csv
import pandas as pd
from multiprocessing import Pool
import time
from scipy.interpolate import CubicSpline as cs
from scipy.integrate import dblquad
import subprocess
import warnings
warnings.filterwarnings('ignore')

# variables
n = 399   
r1 = 1.e-6
r2 = 1.e2

xr1 = np.log(r1)
xr2 = np.log(r2)

hr = (xr2 - xr1) / n

hy = 0.2
ymax = 0.4
y = np.arange(0.0, ymax, hy)

# Arrays for N and r in N(r)
xlr_ = [xr1 + i * hr for i in range(n + 1)]
r_ = np.exp(xlr_)
n_ = []

# parameters
nc   = 3        # number of colors
nf   = 3        # number of active flavors
lamb = 0.241  # lambda QCD (default)

beta = (11 * nc - 2. * nf)/(12 * np.pi)
afr  = 0.7     # frozen coupling constant (default)

xr0, r0, n0 = 0., 0., 0.
c2, gamma, qs02, ec = 0. , 0., 0., 0.   # fitting parameters
e  = np.exp(1)

# initial condition
def mv(vr):
    xlog = np.log(1/(lamb * vr) + ec * e)
    xexp = np.power(qs02 * vr * vr, gamma) * xlog/4.0
    return 1 - np.exp(-xexp)

def find_r1(vr, vz, thet):
    r12 = (0.25 * vr * vr) + (vz * vz) - (vr * vz * np.cos(thet))
    return np.sqrt(r12)


def find_r2(vr, vz, thet):
    r22 = (0.25 * vr * vr) + (vz * vz) + (vr * vz * np.cos(thet))
    return np.sqrt(r22)

def nfunc(qlr):
    x = 0.0
    if qlr < xr1:
        x = n_[0] * np.exp(2 * qlr)/(r1 * r1)
    elif qlr >= xr2:
        x = 1.
    else:
        x = cs(xlr_, n_)(qlr)
    return x

def alphaS(rsq):
    if rsq > rfr2:
        return afr
    else:
        xlog = np.log((4 * c2)/(rsq * lamb * lamb))
        return 1/(beta * xlog)

# kernel
def k(vr, vr1, vr2):

    if (vr1 < 1e-20) or (vr2 < 1e-20):
        return 0
    else:
        rr = vr * vr
        r12 = vr1 * vr1
        r22 = vr2 * vr2

        t1 = rr/(r12 * r22)
        t2 = (1/r12) * (alphaS(r12)/alphaS(r22) - 1)
        t3 = (1/r22) * (alphaS(r22)/alphaS(r12) - 1)

        prefac = (nc * alphaS(rr))/(2 * np.pi * np.pi)
        return prefac * (t1 + t2 + t3)

def f(theta, vr):

    z = np.exp(vr)
    r1_ = find_r1(r0, z, theta)
    r2_ = find_r2(r0, z, theta)

    xlr1 = np.log(r1_)
    xlr2 = np.log(r2_)

    nr0 = n0 + kk(xr0)
    nr1 = nfunc(xlr1) + kk(xlr1)
    nr2 = nfunc(xlr2) + kk(xlr2)

    return 4 * z * z * k(r0, r1_, r2_) * (nr1 + nr2 - nr0 - nr1 * nr2)

def intg(xx):
    global xr0, r0, n0

    index = xlr_.index(xx)
    xr0 = xx
    r0  = np.exp(xr0)
    n0 = n_[index]

    return dblquad(f, xr1, xr2, 1.e-6, 0.5 * np.pi, epsabs=0.0, epsrel=1.e-4)[0]

# return type: array
def evolve(order):
    global kk
    # Euler's method
    kk = cs(xlr_, [0 for i in range(len(xlr_))])
    with Pool(processes=5) as pool:
        k1 = np.array(pool.map(intg, xlr_, chunksize=80))

    if order=='RK1':
        return hy * k1

    # RK2
    list_k1 = list(k1 * hy * 0.5)
    kk = cs(xlr_, list_k1)
    with Pool(processes=5) as pool:
        k2 = np.array(pool.map(intg, xlr_, chunksize=80))

    if order=='RK2':
        return hy * k2

    # RK3
    list_k2 = list(k2 * hy * 0.5)
    kk = cs(xlr_, list_k2)
    with Pool(processes=5) as pool:
        k3 = np.array(pool.map(intg, xlr_, chunksize=80))

    # RK4
    list_k3 = list(k3 * hy)
    kk = cs(xlr_, list_k3)
    with Pool(processes=5) as pool:
        k4 = np.array(pool.map(intg, xlr_, chunksize=80))

    if order=='RK4':
        return (1/6) * hy * (k1 + 2 * k2 + 2 * k3 + k4)

# pass fitting variables q_, c_, g_ to set variables in master.py
def master(q_, c2_, g_, ec_, filename='', order='RK4'):
    global n_, qs02, c2, gamma, ec, rfr2

    # variables
    qs02  = q_
    c2    = c2_
    gamma = g_
    ec    = ec_
    rfr2  = 4 * c2/(lamb * lamb * np.exp(1/(beta * afr)))

    l = ['n   ', 'r1  ', 'r2  ', 'y   ', 'hy  ', 'ec  ', 'qs02 ', 'c2  ', 'g ', 'order']
    v = [n, r1, r2, ymax, hy, ec, qs02, c2, gamma, order]
    bk_arr = []
    t1 = time.time()

    # initial condition----------------------------------------------------------
    n_ = [mv(r_[i]) for i in range(len(r_))]
    #----------------------------------------------------------------------------
    # begin evolution
    for i in range(len(y)):
        y0 = y[i]
        # print("y = " + str(y0))

        for j in range(len(r_)):
            print('r = ' + str(r_[j]) + ', N(r,Y) = ' + str(n_[j]))
            bk_arr.append([y0, r_[j], n_[j]])

        # calculate correction and update N(r,Y) to next step in rapidity

        xk = evolve(order)
        n_ = [n_[j] + xk[j] for j in range(len(n_))]

        # remove nan values from solution
        xx = np.array(xlr_)
        nn = np.array(n_)
        idx_finite = np.isfinite(nn)
        f_finite = interpolate.interp1d(xx[idx_finite], nn[idx_finite])
        nn = f_finite(xx)
        n_ = nn.tolist()

        # solutions should not be greater than one or less than zero
        for j in range(len(n_)):
            if n_[j] < 0.:
                n_[j] = np.round(0.0, 2)
            if n_[j] > 0.9999:
                n_[j] = np.round(1.0, 2)

    t2 = time.time()
    print('bk run time: ' + str((t2 - t1)/3600) + ' hours')

    if filename != '':
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(l)
            writer.writerow(v)
            for j in range(len(bk_arr)):
                writer.writerow(bk_arr[j])

    return pd.DataFrame(bk_arr, columns=['y', 'r', 'N(r,Y)'])

if __name__ == "__main__":
    # qsq2, c^2, g, ec, filename
    p = []

    with open('params.csv', 'r') as foo:
        reader = csv.reader(foo, delimiter='\t')
        header = next(reader)
        p      = next(reader)

    bk = master(float(p[0]), float(p[1]), float(p[2]), float(p[3]), p[4], p[5])

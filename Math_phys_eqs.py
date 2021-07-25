import numpy as np
from numpy import *
import warnings
import matplotlib.pyplot as plt


class MathPhysEqs:
    @staticmethod
    def sph_bessel_j(x, l):
        """
        Spherical Bessel Function j_l(x)
        :param x: argument
        :param l:
        :return: j_l(x)
        """
        j = zeros([2, len(x)])
        j[0] = sin(x) / x
        j[1] = sin(x) / x**2 - cos(x) / x
        if l > 1:
            for i in range(2, l+1):
                jn = (2*i-1)/x * j[1] - j[0]
                j[0] = j[1]
                j[1] = jn
            return j[1]
        else:
            return j[l]

    @staticmethod
    def sph_bessel_n(x, l):
        n = zeros([2, len(x)])
        n[0] = -cos(x) / x
        n[1] = -cos(x) / x ** 2 - sin(x) / x
        if l > 1:
            for i in range(2, l + 1):
                nn = (2 * i - 1) / x * n[1] - n[0]
                n[0] = n[1]
                n[1] = nn
            return n[1]
        else:
            return n[l]

    @staticmethod
    def legendre_pn(x, n):
        p = np.zeros(len(x))
        for m in range(0, int(n/2)+1):
            p += (-1)**m * math.factorial(2*n - 2*m)\
                 /(2**n * math.factorial(m) * math.factorial(n-m) * math.factorial(n-2*m))\
                 * x**(n-2*m)
        return p


class PhysSolver:
    @staticmethod
    def Numerov(V, r, E):
        phi = np.zeros((1, len(r)))
        u = np.zeros((2, len(r)))
        d = np.array([0.0, 1.0])
        energy = np.zeros(1)
        h = r[1] - r[0]
        for e in E:
            mididx = MathSolver.find_zero_idx(lambda x: V(x) - e, r)
            # mididx = int(len(r)/2 + 60)

            f = zeros(len(r))
            f[1] = h
            f[-2] = h

            def F1(r0):
                return 1.0 - h/12 * (V(r0)-e)
            for i in range(2, mididx+1):
                f[i] = ((12 - 10 * F1(r[i - 1])) * f[i - 1] - F1(r[i - 2]) * f[i - 2]) / F1(r[i])
            f_mid0 = f[mididx]
            for i in range(len(r)-3, mididx-1, -1):
                f[i] = ((12 - 10 * F1(r[i + 1])) * f[i + 1] - F1(r[i + 2]) * f[i + 2]) / F1(r[i])
            if f_mid0 != 0.0:
                div = f[mididx] / f_mid0
                for i in range(mididx):
                    f[i] *= div
            elif f[mididx] != 0.0:
                div = f_mid0 / f[mididx]
                for i in range(len(r)-1, mididx, -1):
                    f[i] *= div
            f /= max(f)
            f = MathSolver.square_norm(f, h)
            di = abs(2*f[mididx] - f[mididx-1] - f[mididx+1])/h
            if d[0] >= d[1] and d[1] <= di and d[1] < 1:
                phi = append(phi, atleast_2d(u[1]), axis=0)
                print("Found eigenfunction with energy: {}".format(energy[-1]))
            d[0], u[0] = d[1], u[1]
            d[1] = di
            u[1] = f
            energy = np.append(energy, e)
        if len(phi) > 1:
            phi = delete(phi, 0, axis=0)
            energy = delete(energy, 0, axis=0)
        print(len(phi))
        return phi, energy


class MathSolver:
    @staticmethod
    def square_norm(u, h):
        norm = 0
        for i in range(len(u)):
            norm += h * u[i] ** 2
        return u / sqrt(norm)

    @staticmethod
    def find_zero_idx(func, x):
        for i in range(len(x)-1):
            if func(x[i]) * func(x[i+1]) < 0:
                return i+1
            elif func(x[i]) <= 0.0001:
                return i
        warnings.warn('No zero point found in your function! Possibly because of the mesh size is too large. Use '
                      'middle point instead.', UserWarning)
        return int(len(x)/2)


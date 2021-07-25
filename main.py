from pylab import *
from scipy import integrate
import numpy as np
from Math_phys_eqs import *

mesh = 2000
mesh_t = 200
mesh_e = 2000


def V_poly(r):
    return (r - 1) *(r+1)*(r-3)*(r+3) * 0.001

def V_harm(r):
    return r ** 2


r = linspace(-8, 8, mesh)
t = linspace(0, 100, mesh_t)
E = linspace(0.0001, 0.2, mesh_e)
h = r[1] - r[0]

psi, energy = PhysSolver.Numerov(V_harm, r, E)

# for ps in psi:
#     plot(r, ps)

phi = np.zeros(mesh)
for i in range(mesh):
    if -1 < r[i] < 1:
        phi[i] = cos(pi/2 * r[i])

phi = MathSolver.square_norm(phi, h)

a = np.zeros(len(psi))

for i in range(len(a)):
    a[i] = dot(phi, psi[i])

a = a.reshape([len(a), 1])
Psi = [MathSolver.square_norm(sum(a * psi * exp(energy * 1j * ti), axis=0), h) for ti in t]
Psi = np.array(Psi)
print(Psi.shape)

plt.ion()

for i in range(mesh_t):
    plot(r, real(Psi[i]))
    plot(r, imag(Psi[i]))
    plt.pause(0.1)
    plt.clf()

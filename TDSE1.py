# TDSE

import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from matplotlib.animation import FuncAnimation


# Solve TDSE

# ============================================
# Initial Conditions
# ============================================


N = 2001
Frames = 10000
L = 1
x = np.linspace(0,1,N)
dx = x[1] - x[0]
dt = 0.00001

# initial wavefunction
psi0 = np.sqrt(2) * np.sin(np.pi * x)

# Defining the potential. 

def VFunc(x):
    return -1e4 * np.exp(-(x-L/2)**2 / (2*(L/20))**2)

V = VFunc(x)



# ============================================
# Solve the time independent schrodinger equation
# ============================================

# Diagonals of the matrix

mainDiag = 1/(dx**2) + V[1:-1]
otherDiag = -1/(2 * dx**2) * np.ones(len(mainDiag)-1)

# Find eigenenergies and eigenvectors of said matrix

w,v = eigh_tridiagonal(mainDiag,otherDiag)

# w = eigenenergies
# v = eigenvectors

# Energy values we are using
EVals = w[0:70] 

# Values of the eigenstates at specific x points, padded with 0 on each side
psiVals = np.pad(v.T[0:70], [(0,0), (1,1)], mode='constant')

# Constants
cs = np.dot(psiVals, psi0)

# State at time t
def psiNext(t):
    return psiVals.T@(cs*np.exp(-1j * EVals * t))

# The returned eigenfunctions are across columns, so we transpsoe it


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)

wave, = ax.plot([],[])
timeText = ax.text(0.65,16,'')

ax.set_xlim(0,L)
ax.set_ylim(-0.1,4)
ax.set_xlabel('x')
ax.set_ylabel('psi**2')

def animate(i):
    t = 1 * i * dt
    prob = np.abs(psiNext(t))**2
    wave.set_data(x, prob)
    timeText.set_text('$T = {:.1f}$'.format(t))
    return (wave,)

ani = FuncAnimation(fig, animate, frames=Frames, interval=50)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from numba import jit
import numba


# Parameters: Taking L = 1 simplifies it a lot

xNum = 301
tNum = 200000
dt=1e-7
L = 1
dx = L/(xNum - 1)

# Constants

hRed = 1
m = 1

xVals = np.linspace(0,L,xNum)

'''
initialWave = (np.sqrt(2/L) * np.sin(np.pi * xVals) + np.sqrt(2/L) * np.sin(np.pi * xVals * 2))
initialWave = initialWave / np.sqrt(np.sum(np.abs(initialWave)))
'''







# Potential well
'''
@numba.njit
def vPot(x):
    if x < 0:
        return 999
    elif x > L:
        return 999
    else:
        return 0
'''

# Make the potential gaussian

mu = L/2
sigma = mu * 0.1

momentum = 0.0


# Produces the initial wave
initialWave = np.sqrt(2/L) * np.sin(np.pi * xVals)
# We will then multiply it by a gaussian packet to confine it to the well

envelopeSigma = 0.05
envelope = np.exp(- (xVals - mu)**2 / (2 * envelopeSigma ** 2))
initialWave = initialWave * envelope

# Normalise the wave after confining it
initialWaveTotal = np.sqrt(np.sum(np.abs(initialWave)**2) *dx)

initialWave = initialWave / initialWaveTotal




V = 1e4 * -1/(sigma * np.sqrt(2 * np.pi))*np.exp(- (xVals - mu) ** 2 / (2*sigma**2))

#V = -1e4 * np.exp(- (xVals - mu) ** 2 / (2*sigma**2))


#V = np.zeros(len(xVals))
print('If this is not equal to 1, panic: ', np.sum(np.absolute(initialWave**2)*dx)) 
print('If this is large, also panic: ', dt/(dx**2))

#print('Potentials:', V)

#V = np.zeros(xNum, dtype=np.float64)
V[0] = V[-1] = 10

# Calculate wavefunction


@numba.jit("c16[:,:](c16[:,:])",cache=True, nopython=True, nogil=True)
def wavefunc(psi):
    for t in range(0, tNum - 1):
        for x in range(1, xNum -1):
            psi[t+1][x] = psi[t][x] + (1j * hRed)/(2 * m)* (dt)/(dx**2) * (psi[t][x+1] + psi[t][x-1] - 2 * psi[t][x]) - 1j/hRed * dt * V[x] * psi[t][x]
        # Set boundaries to zero
        psi[t+1][0] = psi[t+1][-1] = 0.0 + 0.0j
        # The wavefunction is of course an approxmation, so this will have to normalise it, and make sure the integral doesnt become larger than 1
        if t % 100 == 0:
            total = np.sum(np.absolute(psi[t+1]**2)*dx)
            sqrtTotal = np.sqrt(total)
            for i in range(1, xNum-1):
                psi[t+1][i] = psi[t+1][i] / sqrtTotal
    return psi


wave = np.zeros([tNum, xNum], dtype=np.complex128)
wave[0] = initialWave

#print(wave)
print('Calculating starting')
result = wavefunc(wave)
print('Calculating finished')

#print(result)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)

ax.set_xlim(0,L)
yMax = np.max(np.abs(result))**2
ax.set_ylim(-yMax, yMax * 1.1)

vMax = np.max(np.abs(V))
vScalingInspect = V.copy()
vScalingInspect[0] = vScalingInspect[-1] = np.nan

scaledV = vScalingInspect * (yMax / vMax)

'''
plt.plot(xVals, np.abs(result[10000])**2)
plt.plot(xVals, np.abs(result[100]) ** 2, ls='--')
plt.show()
'''

wavePlt, = ax.plot(xVals, np.abs(result[0])**2)
ax.plot(xVals, scaledV)

skipFrames = 10

def frame(i):
    waveVal = np.abs(result[i*skipFrames])**2
    wavePlt.set_data(xVals,waveVal)
    if i == tNum//skipFrames - 1:
        print('Looped')
    return (wavePlt,)
   
print('Animation starting')

animation = FuncAnimation(fig, frame, frames=tNum//skipFrames, interval=1, blit=True, repeat=True, cache_frame_data=False)

plt.show()

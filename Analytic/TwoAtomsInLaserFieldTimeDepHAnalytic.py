import matplotlib.pyplot as plt
from scipy import constants
import matplotlib as mpl
import numpy as np
import helper
from scipy import linalg

# Define initial state and measurement state
psi_0 = helper.zero_zero_cb
phi = helper.bell_plus_cb

# Inputs
Omega = 2*np.pi* 1e6
phi_L = 2*np.pi* 0

t_0 = 0
t_max = 1e-6
n_steps = 10000

times = np.linspace(t_0, t_max, n_steps)
dt = times[1] - times[0]

# Setup plot
fig, ax = plt.subplots()

prob_tdts = []
psi_t = psi_0

def detuning_t_func(t):
    t_max = 1e-6 # s
    Detuning_max = 2 * np.pi * 50 * 1e6 # rad s-1
    return 8 * Detuning_max * ( (t - t_max/2) / t_max )**3 


for time in times:

    Delta = detuning_t_func(time)

    Omega_prime = np.sqrt( np.pow(Delta, 2) + np.pow(Omega, 2) )

    H4 = np.zeros((4,4), dtype=complex)

    H4[0][0] =  2 * Delta
    H4[3][3] = -2 * Delta

    H4[0][1] = Omega * np.exp(- 1j * phi_L)
    H4[0][2] = Omega * np.exp(- 1j * phi_L)

    H4[1][3] = Omega * np.exp(- 1j * phi_L)
    H4[2][3] = Omega * np.exp(- 1j * phi_L)

    H4[3][2] = Omega * np.exp(  1j * phi_L)
    H4[3][1] = Omega * np.exp(  1j * phi_L)

    H4[2][0] = Omega * np.exp(  1j * phi_L)
    H4[1][0] = Omega * np.exp(  1j * phi_L)

    H4 *= constants.hbar / 2

    # Calculate propagator in eigenbasis of Hamiltonian
    D4_T = linalg.expm( -1j * H4 * dt / constants.hbar)

    psi_tdt = np.dot(D4_T, psi_t)  # Evolve psi_t

    # Calc prob of being in state phi at time t + dt
    prob_tdt = np.abs(np.vdot(phi, psi_tdt)) ** 2

    # Ensure probabilities are real and wavefunctions are normalised
    assert np.isclose(np.abs(np.vdot(psi_tdt, psi_tdt)) ** 2, 1)
    assert np.allclose(prob_tdt, np.real(prob_tdt))
    prob_tdt = np.real(prob_tdt)
    
    # Store probabilities
    prob_tdts.append(prob_tdt)

    psi_t = psi_tdt

# Plot
ax.plot(times, prob_tdts)

plt.xlabel("Time(μs)")
plt.ylabel("Probability")

ax.set_ylim(0, 1)

plt.show()

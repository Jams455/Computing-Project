import matplotlib.pyplot as plt
from scipy import constants
import matplotlib as mpl
import numpy as np
import helper

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
    Detuning_max = 2*np.pi * 50 * 1e6 # Hz
    t_max = 1e-6 # s
    return 8 * Detuning_max * ( (t - t_max/2) / t_max )**3 

for time in times:

    Delta = detuning_t_func(time)

    Omega_prime = np.sqrt( np.pow(Delta, 2) + np.pow(Omega, 2) )

    # Calculate 2D rotation operator matrix elements
    R2_00 = ( Delta + Omega_prime ) / np.sqrt( 2 * Omega_prime * (Omega_prime + Delta) )
    R2_01 = ( Delta - Omega_prime ) / np.sqrt( 2 * Omega_prime * (Omega_prime - Delta) )
    R2_10 = ( Omega * np.exp(1j*phi_L) ) / np.sqrt( 2 * Omega_prime * (Omega_prime + Delta) )
    R2_11 = ( Omega * np.exp(1j*phi_L) ) / np.sqrt( 2 * Omega_prime * (Omega_prime - Delta) )

    # Calculate 2D eigenvalues and eigenvectors
    EIG_VAL2_0 =   constants.hbar * Omega_prime / 2
    EIG_VAL2_1 = - constants.hbar * Omega_prime / 2

    EIG_VAL2_S = np.array([ EIG_VAL2_0,
                            EIG_VAL2_1])

    EIG_VEC2_0 = np.array([R2_00, R2_10])
    EIG_VEC2_1 = np.array([R2_01, R2_11])

    # Calculate 4D eigenvalues and eigenvectors
    EIG_VAL4_S = np.array([ EIG_VAL2_0 + EIG_VAL2_0, 
                            EIG_VAL2_0 + EIG_VAL2_1, 
                            EIG_VAL2_1 + EIG_VAL2_0, 
                            EIG_VAL2_1 + EIG_VAL2_1])

    EIG_VEC4_0 = np.kron(EIG_VEC2_0, EIG_VEC2_0)
    EIG_VEC4_1 = np.kron(EIG_VEC2_0, EIG_VEC2_1)
    EIG_VEC4_2 = np.kron(EIG_VEC2_1, EIG_VEC2_0)
    EIG_VEC4_3 = np.kron(EIG_VEC2_1, EIG_VEC2_1)

    # Calculate the 4D rotation matrix and it's conj transpose
    R4 = np.array([EIG_VEC4_0,
                EIG_VEC4_1,
                EIG_VEC4_2,
                EIG_VEC4_3])

    R4_CT = R4.conj().T

    # Calculate propagator in eigenbasis of Hamiltonian
    D4_T = np.diag( np.exp( -1j * EIG_VAL4_S * dt / constants.hbar) )

    psi_t_eigen = np.dot(R4, psi_t)      # Rotate psi_t into the eigenbasis
    psi_tdt_eigen = np.dot(D4_T, psi_t_eigen)  # Evolve psi_t in the eigenbasis
    psi_tdt = np.dot(R4_CT, psi_tdt_eigen)         # Return to computational basis

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

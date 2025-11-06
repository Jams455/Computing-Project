import matplotlib.pyplot as plt
from scipy import constants
import matplotlib as mpl
import numpy as np
import helper

# Define initial state and measurement state
psi_0 = helper.one_cb
phi = helper.one_cb

# Inputs
Omega = 2*np.pi* 1e6
phi_L = 2*np.pi* 0.234134541

Delta = Omega

times = np.linspace(0, 1e-6, 1000)

Omega_prime = np.sqrt( np.pow(Delta, 2) + np.pow(Omega, 2) )

# Calculate eigenvalues and eigenvectors of Hamiltonian from closed form expressions
R_00 = ( Delta + Omega_prime ) / np.sqrt( 2 * Omega_prime * (Omega_prime + Delta) )
R_01 = ( Delta - Omega_prime ) / np.sqrt( 2 * Omega_prime * (Omega_prime - Delta) )
R_10 = ( Omega * np.exp(1j*phi_L) ) / np.sqrt( 2 * Omega_prime * (Omega_prime + Delta) )
R_11 = ( Omega * np.exp(1j*phi_L) ) / np.sqrt( 2 * Omega_prime * (Omega_prime - Delta) )

EIG0 =   constants.hbar * Omega_prime / 2
EIG1 = - constants.hbar * Omega_prime / 2
EIGS = np.array([EIG0, EIG1])

# Calculate rotation operators
R2 = np.array([[R_00, R_01], [R_10, R_11]])
R2_CT = R2.conj().T

# Setup plot
fig, ax = plt.subplots()

prob_ts = []

for time in times:
    # Calculate propagator in eigenbasis of Hamiltonian
    D_T = np.diag( np.exp( -1j * EIGS * time / constants.hbar) )

    psi_0_eigen = np.dot(R2_CT, psi_0)      # Rotate psi_0 into the eigenbasis
    psi_t_eigen = np.dot(D_T, psi_0_eigen)  # Evolve psi_0 in the eigenbasis
    psi_t = np.dot(R2, psi_t_eigen)         # Return to computational basis

    # Calc prob of being in state phi at time t
    prob_t = np.abs(np.vdot(phi, psi_t)) ** 2

    # Ensure probabilities are real and wavefunctions are normalised
    assert np.isclose(np.abs(np.vdot(psi_t, psi_t)) ** 2, 1)
    assert np.allclose(prob_t, np.real(prob_t))
    prob_t = np.real(prob_t)
    
    # Store probabilities
    prob_ts.append(prob_t)

# Plot
ax.plot(times, prob_ts)

plt.xlabel("Time(μs)")
plt.ylabel("Probability")

ax.set_ylim(0, 1)

plt.show()

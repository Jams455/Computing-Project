# Most basic simulation, evolving |ψ(t=0)> = |0> according to the hamiltonian:
#
# H = ( 0 γ )
#     ( γ 0 )
#

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def solve_schrodinger(H, psi_0, t):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(H)
    eigenvectors = np.array([eigenvectors[:,i] for i in range(len(eigenvectors))])

    # Rotation matrix is formed from a matrix of eigenvectors
    rotation_matrix = eigenvectors.copy()

    # Convert psi_0 into the eigenbasis of H
    rotation_matrix_dagger = rotation_matrix.conj().T  # First compute conjugate transpose of rotation matrix
    psi_0_eigen = np.dot(rotation_matrix_dagger, psi_0)

    # Calculate the evolution operator in the eigenbasis of the Hamiltonian
    diagonal_terms = np.cos( - eigenvalues * t / hbar ) + np.sin( - eigenvalues * t / hbar ) * 1j
    evolution_matrix = np.diag(diagonal_terms)

    # Evolve psi_0_eigen
    psi_t_eigen = np.dot(evolution_matrix, psi_0_eigen)

    # Transform back into computational basis
    psi_t = np.dot(rotation_matrix, psi_t_eigen)

    return psi_t

# Inputs
psi_0 = np.array([1, 0]) # |ψ(t=0)> = |0>
gamma = 1
hbar = 1

# Explicitly calculate Hamiltonian
H = np.zeros((2, 2))
H[0][1] = gamma
H[1][0] = gamma

# Computational basis vectors
zero = np.array([1+0j, 0+0j])
one = np.array([0+0j, 1+0j])

# Set up plotting vars
times = np.linspace(0, 5, 300)
lower_vals = []
higher_vals = []

for time in times:
    # Calculate the wavefunction at different times
    answer = solve_schrodinger(H, psi_0, time)
    
    # Calculate the probability of being in both computational basis states
    lower_vals.append(abs(np.dot(zero, answer))**2)
    higher_vals.append(abs(np.dot(one, answer))**2)

# Plot probabilities of wavefunction being in states |0> and |1> as a function of time
fig, ax = plt.subplots()
ax.plot(times, lower_vals, label="P( |0> )")
ax.plot(times, higher_vals, label="P( |1> )")
plt.legend()
plt.show()

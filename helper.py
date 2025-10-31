import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy import constants


# Calculate wavefunction at times [t_0, t_1, ..., t_n] given Hamiltonian (H) and initial wavefunction psi_0
def solve_schrodinger_time_independent(H, psi_0, t):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eigh(H)
    # ^ This could be brought outside function to avoid repeat calculations if function is called multiple times

    # Rotation matrix is formed from a matrix of eigenvectors
    rotation_matrix = eigenvectors.copy()

    # Convert psi_0 into the eigenbasis of H
    rotation_matrix_dagger = rotation_matrix.conj().T  # First compute conjugate transpose of rotation matrix
    psi_0_eigen = np.dot(rotation_matrix_dagger, psi_0)

    psi_ts = []

    # Calculate the evolution operator in the eigenbasis of the Hamiltonian
    for time in t:
        diagonal_terms = np.exp( - 1j * eigenvalues * time / constants.hbar )
        evolution_matrix = np.diag(diagonal_terms)

        # Evolve psi_0_eigen
        psi_t_eigen = np.dot(evolution_matrix, psi_0_eigen)

        # Transform back into computational basis
        psi_t = np.dot(rotation_matrix, psi_t_eigen)

        # Add to psi_ts list
        psi_ts.append(psi_t)

    return psi_ts

def calc_probability(wavefunction_value, measurement_vector):
    return abs(np.vdot(measurement_vector, wavefunction_value))**2


# Common state vectors in computational basis
zero_cb     = np.array([ 1 + 0j ,  0 + 0j ])
one_cb      = np.array([ 0 + 0j ,  1 + 0j ])

plus_cb     = np.array([ 1 + 0j ,  1 + 0j ]) / np.sqrt(2)
minus_cv    = np.array([ 1 + 0j , -1 + 0j ]) / np.sqrt(2)

plus_i_cb   = np.array([ 1 + 0j ,  0 + 1j ]) / np.sqrt(2)
minus_i_cb  = np.array([ 1 + 0j ,  0 - 1j ]) / np.sqrt(2)

zero_zero_cb = np.kron(zero_cb, zero_cb)
zero_one_cb = np.kron(zero_cb, one_cb)
one_zero_cb = np.kron(one_cb, zero_cb)
one_one_cb = np.kron(one_cb, one_cb)

bell_plus_cb = np.array([ 0 + 0j , 1 + 0j ,  1 + 0j , 0 + 0j ]) / np.sqrt(2)

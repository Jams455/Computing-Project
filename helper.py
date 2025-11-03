import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy import constants


# Calculate wavefunction at times [t_0, t_1, ..., t_n] given Hamiltonian (H) and initial wavefunction psi_0
def solve_schrodinger_time_independent(H, psi_0, t):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eigh(H)

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

def detuning_t_func(t):
    Detuning_max = 2*np.pi * 50 * 1e6 # Hz
    t_max = 1e-6 # s
    return 8 * Detuning_max * ( (t - t_max) / (2 * t_max) )**3 

# Calculate Hamiltonians at time T for different values of V
def Hamiltonian_T(rabi_freq, phi_L, V, t):
    # Calculate 2x2 Hamiltonian
    H_2 = np.zeros((2, 2), dtype=complex)

    H_2[0][0] =   detuning_t_func(t)
    H_2[1][1] = - detuning_t_func(t)

    H_2[0][1] = rabi_freq * np.exp( - 1j * phi_L )
    H_2[1][0] = rabi_freq * np.exp(   1j * phi_L )

    H_2 *= constants.hbar / 2

    # Calculate 4x4 Hamiltonians with different V values
    H_4 = np.kron(H_2, np.diag((1, 1))) + np.kron(np.diag((1, 1)), H_2)

    H_4 += + np.diag([0, 0, 0, 2 * V]) * constants.hbar / 2

    return H_4

# Calculate wavefunction at times [t_0, t_1, ..., t_n] given Hamiltonian ( H = H(t) ) and initial wavefunction psi_0
def solve_schrodinger_time_dependent(rabi_freq, phi_L, V, psi_0, t_0, t_max, n_steps):
    # Store values of wavefunction, psi, at different times, t
    psi_ts = [psi_0]

    psi_t = psi_0 # psi_t keeps track of current wfn

    delta_t = ( t_max - t_0 ) / n_steps

    for time in np.linspace(0, 1e-6, 299):
        # Calculate Hamiltonian
        H = Hamiltonian_T(rabi_freq, phi_L, V, time)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eigh(H)

        # Rotation matrix is formed from a matrix of eigenvectors
        rotation_matrix = eigenvectors.copy()

        # Convert psi_t into the eigenbasis of H(t)
        rotation_matrix_dagger = rotation_matrix.conj().T  # First compute conjugate transpose of rotation matrix
        psi_t_eigen = np.dot(rotation_matrix_dagger, psi_t)

        # Calculate the evolution operator in the eigenbasis of the Hamiltonian
        diagonal_terms = np.exp( - 1j * eigenvalues * delta_t / constants.hbar )
        evolution_matrix = np.diag(diagonal_terms)

        # Evolve psi_t_eigen
        psi_t_plus_dt_eigen = np.dot(evolution_matrix, psi_t_eigen)

        # Transform back into computational basis
        psi_t_plus_dt = np.dot(rotation_matrix, psi_t_plus_dt_eigen)

        # Add to psi_ts list
        psi_ts.append(psi_t_plus_dt)

        psi_t = psi_t_plus_dt

    return psi_ts

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

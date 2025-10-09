# +-----------------------------------------------------------------------------------+
# |                                                                                   |
# |  Now evolving |ψ(t=0)> = |+> according to the hamiltonian of an atom in L field:  |
# |                                                                                   |
# |  H = ( E_0  0  )                                                                  |
# |      (  0  E_1 )                                                                  |
# |                                                                                   |
# +-----------------------------------------------------------------------------------+

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy import constants
import matplotlib as mpl

# Calculate wavefunction at time (t) given Hamiltonian (H) and initial wavefunction psi_0
def solve_schrodinger(H, psi_0, t):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(H)
    eigenvectors = np.array([eigenvectors[:,i] for i in range(len(eigenvectors))]) 
    # ^ This could be brought outside function to avoid repeat calculations if function is called multiple times

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

def calc_probability(wavefunction_value, measurement_vector):
    return abs(np.dot(measurement_vector, wavefunction_value))**2

# Common state vectors in computational basis
zero_cb     = np.array([ 1 + 0j ,  0 + 0j ])
one_cb      = np.array([ 0 + 0j ,  1 + 0j ])

plus_cb     = np.array([ 1 + 0j ,  1 + 0j ]) / np.sqrt(2)
minus_cv    = np.array([ 1 + 0j , -1 + 0j ]) / np.sqrt(2)

plus_i_cb   = np.array([ 1 + 0j ,  0 + 1j ]) / np.sqrt(2)
minus_i_cb  = np.array([ 1 + 0j ,  0 - 1j ]) / np.sqrt(2)

# Inputs
psi_0 = plus_cb # |ψ(t=0)> = |+>
hbar = constants.hbar
E_0 = -13.6 * 1.6e-19
E_1 =  -3.4 * 1.6e-19

# Explicitly calculate Hamiltonian
H = np.zeros((2, 2))
H[0][0] = E_0
H[1][1] = E_1

# Set up plotting vars
times = np.linspace(0, 1e-15, 300)

zero_cb_vals     = []
one_cb_vals      = []

plus_cb_vals     = []
minus_cv_vals    = []

plus_i_cb_vals   = []
minus_i_cb_vals  = []

# Group all cb vectors so they are iterable
all_cb_vector_labels = ["0", "1", "+", "-", "+i", "-i"]
all_cb_vectors = [zero_cb, one_cb, plus_cb, minus_cv, plus_i_cb, minus_i_cb]
all_y_data = [zero_cb_vals, one_cb_vals, plus_cb_vals, minus_cv_vals, plus_i_cb_vals, minus_i_cb_vals]

for time in times:
    # Calculate the wavefunction at different times
    answer = solve_schrodinger(H, psi_0, time)
    
    # Calculate the probability of being in different states
    for cb_vector, y_data in zip(all_cb_vectors, all_y_data):
        y_data.append(calc_probability(cb_vector, answer))

# Create subplot axes
fig, axs_md = plt.subplots(2, 4, sharex=True, sharey=True)
axs = [axs_md[0][0], axs_md[1][0], axs_md[0][1], axs_md[1][1], axs_md[0][2], axs_md[1][2]]

# Format plot
for ax, ax_label, y_data in zip(axs, all_cb_vector_labels, all_y_data):
    ax.plot(times, y_data)

    ax.set_ylim(-0.1, 1.1)
    ax.set_title(rf"$|{ax_label}\rangle$")

    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(direction="in", top=True, right=True, labelsize='small')

    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)

fig.suptitle("Time Evolution of Different State Probabilities")
fig.supylabel("Probability")
fig.supxlabel("Time (s)")

# Display Hamiltonian on plot
mpl.rcParams['text.usetex'] = True

latex_matrix = (
    r"$\displaystyle H = \left( \begin{array}{cc}"
    r"E_0 & 0 \\"
    r"0 & E_1"
    r"\end{array} \right)$"
)

axs_md[0, 3].axis('off')
axs_md[1, 3].axis('off')

axs_md[0, 3].text(0.5, 0.85, latex_matrix, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].set_title("Hamiltonian")

plt.show()


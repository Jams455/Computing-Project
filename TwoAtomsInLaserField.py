# +-----------------------------------------------------------------------------------+
# |                                                                                   |
# |  Now evolving two interacting atoms in laser field according to the hamiltonian:  |
# |                                                                                   |
# |  H =      (    2∆      Ωe^(-iϕ_L)  Ωe^(-iϕ_L)       0     )                       |
# |      hbar ( Ωe^(iϕ_L)      0           0       Ωe^(-iϕ_L) )                       |
# |      / 2  ( Ωe^(iϕ_L)      0           0       Ωe^(-iϕ_L) )                       |
# |           (    0       Ωe^(iϕ_L)   Ωe^(iϕ_L)    -2∆ + 2V  )                       |
# |                                                                                   |
# +-----------------------------------------------------------------------------------+

import matplotlib.pyplot as plt
from scipy import constants
import matplotlib as mpl
import numpy as np
import helper

# Inputs
psi_0 = helper.zero_zero_cb # |ψ(t=0)>

detuning = 1
detuning *= 2 * np.pi * 1e6

rabi_frequency = 1
rabi_frequency *= 2 * np.pi * 1e6

phi_L = 1
phi_L *= 2 * np.pi

V = 1
V *= 2 * np.pi * 1e8

# Calculate 2x2 Hamiltonian
H_2 = np.zeros((2, 2), dtype=complex)

H_2[0][0] =   detuning
H_2[1][1] = - detuning

H_2[0][1] = rabi_frequency * np.exp( - 1j * phi_L )
H_2[1][0] = rabi_frequency * np.exp(   1j *  phi_L )

H_2 *= constants.hbar / 2

# Calculate 4x4 Hamiltonian

H_4 = np.kron(H_2, np.diag((1, 1))) + np.kron(np.diag((1, 1)), H_2)
H_4[3][3] += 2 * V

# Group all cb vectors so they are iterable
all_cb_vector_labels = ["00", "01", "10", "11", "++"]
all_cb_vectors = [helper.zero_zero_cb, helper.zero_one_cb, helper.one_zero_cb, helper.one_one_cb, helper.bell_plus_cb]
all_y_data = [[], [], [], [], []]

# Calculate the wavefunction at different times
times = np.linspace(0, 1e-6, 300)
psi_ts = helper.solve_schrodinger_time_independent(H_4, psi_0, times)

for psi_t in psi_ts:
    # Check each wavefunction is normalised
    assert np.isclose(np.vdot(psi_t, psi_t), 1.0)

    # Calculate the probability of being in different states
    for cb_vector, y_data in zip(all_cb_vectors, all_y_data):
        y_data.append(helper.calc_probability(cb_vector, psi_t))



# ************************** PLOTTING **************************



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
fig.supxlabel(r"Time ($s$)")

# Display Hamiltonian and other params on plot
mpl.rcParams['text.usetex'] = True

latex_matrix = (
    r"$\displaystyle H = \frac{\hbar}{2} \left( \begin{array}{cc}"
    r"\Delta & \Omega e^{-i \phi_L} \\"
    r"\Omega e^{i \phi_L} & - \Delta"
    r"\end{array} \right)$"
)

latex_phi_L = (rf"$\phi_L = {round(phi_L/np.pi*100)/100}\pi\ rad$")
latex_detuning = (rf"$\Delta = {detuning /np.pi / 1e6}\pi\times10^6\ rad\ s^{{-1}}$")
latex_rabi_freq = (rf"$\Omega = {rabi_frequency / np.pi  / 1e6}\pi\times10^6\ rad\ s^{{-1}}$")

axs_md[0, 3].axis('off')
axs_md[1, 3].axis('off')

axs_md[0, 3].text(0.5, 0.85, latex_matrix, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].text(0.5, 0.6, latex_phi_L, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].text(0.5, 0.45, latex_detuning, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].text(0.5, 0.3, latex_rabi_freq, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].set_title("Parameters")

plt.show()

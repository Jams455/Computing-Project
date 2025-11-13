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

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

# Inputs
psi_0 = helper.zero_zero_cb # |ψ(t=0)>

detuning = 0
detuning *= 2 * np.pi * 1e6

rabi_frequency = 1
rabi_frequency *= 1e6 * 2 * np.pi

phi_L = 0.0
phi_L *= 2 * np.pi

Vs = np.linspace(0, 100, 100)
Vs *= 1e6

fontsize = 16

# Calculate 2x2 Hamiltonian
H_2 = np.zeros((2, 2), dtype=complex)

H_2[0][0] =   detuning
H_2[1][1] = - detuning

H_2[0][1] = rabi_frequency #* np.exp( - 1j * phi_L )
H_2[1][0] = rabi_frequency #* np.exp(   1j * phi_L )

H_2 *= constants.hbar / 2

# Calculate 4x4 Hamiltonians with different V values
H_4 = np.kron(H_2, np.diag((1, 1))) + np.kron(np.diag((1, 1)), H_2)

# Group all cb vectors so they are iterable
all_cb_vector_labels = [rf"$|00\rangle$", rf"$|01\rangle$", rf"$|11\rangle$", rf"$|\Psi^+\rangle$"]
all_cb_vectors = [helper.zero_zero_cb, helper.zero_one_cb, helper.one_one_cb, helper.bell_plus_cb]

# Create subplot axes
fig, axs_md = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (12, 8), constrained_layout=True)
axs = [axs_md[0][0], axs_md[1][0], axs_md[0][1], axs_md[1][1]]

cmap = plt.get_cmap('cividis', len(Vs))

for i, V in enumerate(Vs):
    H_4V = H_4.copy() + (np.diag([0, 0, 0, 2 * V]) * constants.hbar / 2)

    all_y_data = [[], [], [], [], []]

    # Calculate the wavefunction at different times
    times = np.linspace(0, 1e-6, 300)
    psi_ts = helper.solve_schrodinger_time_independent(H_4V, psi_0, times)

    for psi_t in psi_ts:
        # Check each wavefunction is normalised
        assert np.isclose(np.vdot(psi_t, psi_t), 1.0)

        # Calculate the probability of being in different states at time t
        for cb_vector, y_data in zip(all_cb_vectors, all_y_data):
            y_data.append(helper.calc_probability(cb_vector, psi_t))



# ************************** PLOTTING **************************



    # Format plot
    for ax, ax_label, y_data in zip(axs, all_cb_vector_labels, all_y_data):

        ax.plot(times*1e6, y_data, color=cmap(i*5), alpha=1, linewidth=0.8)

        ax.set_ylim(-0.1, 1.1)
        ax.set_title(ax_label, fontsize=fontsize)

        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(direction="in", top=True, right=True, labelsize='small')

        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)


axs_md[0][0].set_ylabel("Probability of State", fontsize=fontsize)
axs_md[1][0].set_ylabel("Probability of State", fontsize=fontsize)

axs_md[1][0].set_xlabel(r"Time ($\mu$s)", fontsize=fontsize)
axs_md[1][1].set_xlabel(r"Time ($\mu$s)", fontsize=fontsize)

# Normalization for V values
norm = Normalize(vmin=Vs.min()*1e-6, vmax=Vs.max()*1e-6)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required for colorbar

# Add vertical colorbar to the right of all subplots
cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label(r"$V\ (\mathrm{MHz})$", fontsize=fontsize)

# Save and show fig
plt.savefig("Time Independent", dpi=600)
plt.show()

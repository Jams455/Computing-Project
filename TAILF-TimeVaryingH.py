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

rabi_frequency = 1
rabi_frequency *= 2 * np.pi * 1e6

phi_L = 0.0
phi_L *= 2 * np.pi

Vs = np.linspace(0, 100, 100)
Vs *= 1e6

display_params = True

t_0 = 0
t_max = 1e-6
n_steps = 300
times = np.linspace(0, 1e-6, 300)
detuning_t = helper.detuning_t_func(times)

# Group all cb vectors so they are iterable
all_cb_vector_labels = ["00", "01", "11", "++"]
all_cb_vectors = [helper.zero_zero_cb, helper.zero_one_cb, helper.one_one_cb, helper.bell_plus_cb]

# Create subplot axes
fig, axs_md = plt.subplots(2, 3, sharex=True, figsize = (14, 8))
axs = [axs_md[0][0], axs_md[1][0], axs_md[0][1], axs_md[1][1]]

for ax in axs:
    ax.sharey(axs[0])

for V in Vs:
    all_y_data = [[], [], [], []]

    # Calculate the wavefunction at different times
    psi_ts = helper.solve_schrodinger_time_dependent(rabi_frequency, phi_L, V, psi_0, t_0, t_max, n_steps)

    for psi_t in psi_ts:
        # Check each wavefunction is normalised
        assert np.isclose(helper.calc_probability(psi_t, psi_t), 1.0)

        # Calculate the probability of being in different states at time t
        for cb_vector, y_data in zip(all_cb_vectors, all_y_data):
            y_data.append(helper.calc_probability(cb_vector, psi_t))



# ************************** PLOTTING **************************



    # Format plot
    for ax, ax_label, y_data in zip(axs, all_cb_vector_labels, all_y_data):
        
        ax.plot(times, y_data, color='blue' if V==0 else 'red', alpha=1 if V==0 else 0.3, linewidth=1 if V==0 else 0.6)

        ax.set_ylim(-0.1, 1.1)
        ax.set_title(rf"$|{ax_label}\rangle$")

        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(direction="in", top=True, right=True, labelsize='small')

        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)

axs_md[0][2].set_ylabel("Detuning (Hz)")
axs_md[0][2].plot(times, detuning_t)

fig.suptitle("Time Evolution of Different State Probabilities")
fig.supylabel("Probability of State")
fig.supxlabel(r"Time ($s$)")

if display_params:
    # Display Hamiltonian and other params on plot
    mpl.rcParams['text.usetex'] = True

    latex_matrix = (
        r"$\displaystyle H = \frac{\hbar}{2} \left( \begin{array}{cccc}"
        r"2 \Delta & \Omega e^{-i \phi_L} & \Omega e^{-i \phi_L} & 0 \\"
        r"\Omega e^{i \phi_L} & 0 & 0 & \Omega e^{-i \phi_L} \\"
        r"\Omega e^{i \phi_L} & 0 & 0 & \Omega e^{-i \phi_L} \\"
        r"0 & \Omega e^{i \phi_L} & \Omega e^{i \phi_L} & - 2 \Delta + 2V \\"
        r"\end{array} \right)$"
    )

    latex_phi_L = (rf"$\phi_L = {round(phi_L/np.pi*100)/100}\pi\ \textrm{{rad}}$")
    latex_V = (rf"$( 0 \le V \le 100 ) \cdot 2\pi\ \textrm{{MHz}} $")
    latex_rabi_freq = (rf"$\Omega = {rabi_frequency / np.pi  / 1e6}\pi\times10^6\ \textrm{{rad}}\ \textrm{{s}}^{{-1}}$")

    axs_md[1, 2].text(0.5, 0.8, latex_matrix, fontsize=12, ha='center', va='center', transform=axs_md[1, 2].transAxes)
    axs_md[1, 2].text(0.5, 0.55, latex_phi_L, fontsize=12, ha='center', va='center', transform=axs_md[1, 2].transAxes)
    axs_md[1, 2].text(0.5, 0.4, latex_V, fontsize=12, ha='center', va='center', transform=axs_md[1, 2].transAxes)
    axs_md[1, 2].text(0.5, 0.25, latex_rabi_freq, fontsize=12, ha='center', va='center', transform=axs_md[1, 2].transAxes)
    axs_md[1, 2].set_title("Parameters")

axs_md[1, 2].axis('off')

#plt.savefig("Time Dependent", dpi=600)

plt.show()

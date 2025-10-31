# +---------------------------------------------------------------------------------+
# |                                                                                 |
# |  Most basic simulation, evolving |ψ(t=0)> = |+> according to the hamiltonian:   |
# |                                                                                 |
# |  H = ( E_0  0  )                                                                |
# |      (  0  E_1 )                                                                |
# |                                                                                 |
# +---------------------------------------------------------------------------------+

import matplotlib.pyplot as plt
from scipy import constants
import matplotlib as mpl
import numpy as np
import helper

# Inputs
psi_0 = helper.plus_cb # |ψ(t=0)> = |+>
hbar = constants.hbar
E_0 = -13.6 * 1.6e-19
E_1 =  -3.4 * 1.6e-19

# Explicitly calculate Hamiltonian
H = np.zeros((2, 2))
H[0][0] = E_0
H[1][1] = E_1

# Group all cb vectors so they are iterable
all_cb_vector_labels = ["0", "1", "+", "-", "+i", "-i"]
all_cb_vectors = [helper.zero_cb, helper.one_cb, helper.plus_cb, helper.minus_cv, helper.plus_i_cb, helper.minus_i_cb]
all_y_data = [[], [], [], [], [], []]

# Calculate the wavefunction at different times
times = np.linspace(0, 1e-15, 300)
psi_ts = helper.solve_schrodinger_time_independent(H, psi_0, times)

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
    ax.plot(times*1e15, y_data)

    ax.set_ylim(-0.1, 1.1)
    ax.set_title(rf"$|{ax_label}\rangle$")

    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(direction="in", top=True, right=True, labelsize='small')

    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)

fig.suptitle("Time Evolution of Different State Probabilities")
fig.supylabel("Probability")
fig.supxlabel(r"Time ($fs$)")

# Display Hamiltonian and other params on plot
mpl.rcParams['text.usetex'] = True

latex_matrix = (
    r"$\displaystyle H = \left( \begin{array}{cc}"
    r"E_0 & 0 \\"
    r"0 & E_1"
    r"\end{array} \right)$"
)

latex_E_0 = (rf"$E_0 = {E_0/1.6e-19}eV$")
latex_E_1 = (rf"$E_1 = {E_1/1.6e-19}eV$")

axs_md[0, 3].axis('off')
axs_md[1, 3].axis('off')

axs_md[0, 3].text(0.5, 0.85, latex_matrix, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].text(0.5, 0.6, latex_E_0, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].text(0.5, 0.45, latex_E_1, fontsize=12, ha='center', va='center', transform=axs_md[0, 3].transAxes)
axs_md[0, 3].set_title("Parameters")

plt.show()

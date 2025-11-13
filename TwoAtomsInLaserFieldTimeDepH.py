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
rabi_frequency *= 2 * np.pi * 1e6 # rad s-1

phi_L = 0
phi_L *= 2 * np.pi

Vs = np.linspace(100, 0, 100, dtype=complex)
Vs *= 1e6*2*np.pi

t_0 = 0
t_max = 1e-6
n_steps = 500
times = np.linspace(t_0, t_max, n_steps)

# Create data containers
lin_detuning_t = helper.detuning_t_linear(times, t_max)
cub_detuning_t = helper.detuning_t_cubic(times, t_max)

lin_bell_plus_probs_t = [] # [V_ind][t_ind]
cub_bell_plus_probs_t = [] # [V_ind][t_ind]


for V in Vs:
    lin_bell_plus_probs_t.append([])
    cub_bell_plus_probs_t.append([])

    # Calculate the wavefunction at different times
    lin_det_psi_ts = helper.solve_schrodinger_time_dependent(rabi_frequency, phi_L, V, helper.detuning_t_linear, psi_0, t_0, t_max, n_steps)
    cub_det_psi_ts = helper.solve_schrodinger_time_dependent(rabi_frequency, phi_L, V, helper.detuning_t_cubic, psi_0, t_0, t_max, n_steps)

    for lin_det_psi_t, cub_det_psi_t in zip(lin_det_psi_ts, cub_det_psi_ts):
        # Check each wavefunction is normalised
        assert np.isclose(helper.calc_probability(lin_det_psi_t, lin_det_psi_t), 1.0)
        assert np.isclose(helper.calc_probability(cub_det_psi_t, cub_det_psi_t), 1.0)

        # Calculate the probability of being in the bell+ state
        lin_bell_plus_probs_t[-1].append(np.array(helper.calc_probability(helper.bell_plus_cb, lin_det_psi_t)))
        cub_bell_plus_probs_t[-1].append(np.array(helper.calc_probability(helper.bell_plus_cb, cub_det_psi_t)))

lin_bell_plus_probs_t = np.array(lin_bell_plus_probs_t)
cub_bell_plus_probs_t = np.array(cub_bell_plus_probs_t)

print("max lin prob: ", lin_bell_plus_probs_t.max())
print("max cub prob: ", cub_bell_plus_probs_t.max())

lin_i, lin_j = np.unravel_index(np.argmax(lin_bell_plus_probs_t), lin_bell_plus_probs_t.shape)
cub_i, cub_j = np.unravel_index(np.argmax(cub_bell_plus_probs_t), cub_bell_plus_probs_t.shape)

print(f"({lin_i}, {lin_j})")
print(f"({cub_i}, {cub_j})")

print(lin_bell_plus_probs_t[lin_i, lin_j], f" --->> V={Vs[lin_i]/(2e6*np.pi)} & t={times[lin_j]*1e6}")
print(cub_bell_plus_probs_t[cub_i, cub_j], f" --->> V={Vs[cub_i]/(2e6*np.pi)} & t={times[cub_j]*1e6}")

# ************************** PLOTTING **************************


fontsize = 16
cmap='cividis'

# Create subplot axes
fig, axs = plt.subplots(3, 2, sharex=True, figsize = (12, 8), constrained_layout=True)

((lin_det_ax, cub_det_ax), (lin_scatt_ax, cub_scatt_ax), (lin_cmap_ax, cub_cmap_ax)) = axs

lin_det_ax.sharey(cub_det_ax)
lin_cmap_ax.sharey(cub_cmap_ax)
lin_scatt_ax.sharey(cub_scatt_ax)

cub_det_ax.label_outer()
cub_cmap_ax.label_outer()
cub_scatt_ax.label_outer()

# Convert units for plotting

times *= 1e6                        # Convert to micro seconds
#Vs *= 1e-6 / (2*np.pi)              # Convert to MHz
cub_detuning_t /= 1e6 * 2 * np.pi   # Convert to M rad s-1
lin_detuning_t /= 1e6 * 2 * np.pi   # Convert to M rad s-1

# Format line plots
lin_scatt_ax.set_ylim(-0.1, 1.1)
cub_scatt_ax.set_ylim(-0.1, 1.1)

lin_scatt_ax.set_yticks([0, 0.5, 1])
cub_scatt_ax.set_yticks([0, 0.5, 1])

lin_scatt_ax.tick_params(direction="in", top=True, right=True)   #, labelsize='small')
cub_scatt_ax.tick_params(direction="in", top=True, right=True)   #, labelsize='small')

lin_scatt_ax.minorticks_on()
cub_scatt_ax.minorticks_on()

lin_scatt_ax.tick_params(which="minor", direction="in", top=True, right=True)
cub_scatt_ax.tick_params(which="minor", direction="in", top=True, right=True)

lin_scatt_ax.set_ylabel(r"$|\Psi^+\rangle$ Probability", fontsize=fontsize)

# Format detuning plots
lin_det_ax.tick_params(direction="in", top=True, right=True)
cub_det_ax.tick_params(direction="in", top=True, right=True)

lin_det_ax.minorticks_on()
cub_det_ax.minorticks_on()

lin_det_ax.tick_params(which="minor", direction="in", top=True, right=True)
cub_det_ax.tick_params(which="minor", direction="in", top=True, right=True)

lin_det_ax.set_ylabel(r"Detuning $\Delta$ (MHz)", fontsize=fontsize)

# Format cmap plots
lin_cmap_ax.set_ylabel(r"Interaction Strength"+"\n"+" $V$ (MHz)", fontsize=fontsize)
lin_cmap_ax.set_xlabel(r"Time ($\mu$s)", fontsize=fontsize)
cub_cmap_ax.set_xlabel(r"Time ($\mu$s)", fontsize=fontsize)

# Plot detunings
lin_det_ax.plot(times, lin_detuning_t, color='blue', alpha=1, linewidth=1)
cub_det_ax.plot(times, cub_detuning_t, color='blue', alpha=1, linewidth=1)


# Plot line plots
Vs_to_plot = [99, 97, 95, 0]
colours = ['red', 'green', 'blue', 'purple']
linestyle = ['-', '--', '-.', ':']
labels = [r"$V\ =\ 0\ $MHz", r"$V\ =\ 2\ $MHz", r"$V\ =\ 4\ $MHz", r"$V\ =\ 100\ $MHz"]

for i, V_ind in enumerate(Vs_to_plot):
    lin_scatt_ax.plot(times, lin_bell_plus_probs_t[V_ind], color=colours[i], alpha = 0.8, linewidth = 1.5, linestyle=linestyle[i], label=labels[i])
    cub_scatt_ax.plot(times, cub_bell_plus_probs_t[V_ind], color=colours[i], alpha = 0.8, linewidth = 1.5, linestyle=linestyle[i], label=labels[i])

lin_scatt_ax.legend(fontsize=fontsize-4)
cub_scatt_ax.legend(fontsize=fontsize-4)

# Plot colourmap plots
lin_img = lin_cmap_ax.imshow(lin_bell_plus_probs_t, cmap=cmap, interpolation='nearest', aspect='auto', vmin=0, vmax=1,
                                extent=[np.real(times.min()), np.real(times.max()), np.real(Vs.min()), np.real(Vs.max())])

cub_img = cub_cmap_ax.imshow(cub_bell_plus_probs_t, cmap=cmap, interpolation='nearest', aspect='auto', vmin=0, vmax=1,
                                extent=[np.real(times.min()), np.real(times.max()), np.real(Vs.min()), np.real(Vs.max())])


cbar = fig.colorbar(lin_img, ax=[lin_cmap_ax, cub_cmap_ax],
                    location='right', fraction=0.046, pad=0.04)
cbar.set_label(r"$|\Psi^+\rangle$ Probability", fontsize=fontsize)


# Save plot and display
plt.savefig("Time Dependent", dpi=600)
plt.show()

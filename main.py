import numpy as np
import string
import matplotlib.pyplot as plt

# use the colorblind color package
plt.style.use('seaborn-colorblind')
# use 14 size font for all text
plt.rcParams.update({'font.size': 14})
# get the color cycle
prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

delta_i = {
    'delta_i,BSE': -0.479,
    'delta_i,BSE (+/-err)': 0.027,
    'delta_i,Lunar': -0.064,
    'delta_i,Lunar (+/-err)': 0.043,
    'delta_i,Lunar-BSE': 0.415,
    'delta_i,Lunar-BSE (+/-err)': 0.05,
}
saturation_index = 0.989  # P_i / P_sat
alpha_kin = (39 / 41) ** 0.43  # the kinetic fractionation factor
alpha_evap = 1 + (1 - saturation_index) * (alpha_kin - 1)  # the evaporative fractionation factor
alpha_phys = np.sqrt(39 / 41)
# f_melt_vap = f_melt / (f_melt + (1 - f_melt) * (1 - f_esc))
runs = {
    "Canonical": {
        "f_melt": 0.191,
        "f_esc": 0.74,
        "color": prop_cycle[0],
        # "linestyle": "solid",
    },
    "Half-Earths": {
        "f_melt": 0.504,
        "f_esc": 0.16,
        "color": prop_cycle[1],
        # "linestyle": "dashed",
    },
}
for run in runs:
    runs[run]['f_melt_vap'] = runs[run]['f_melt'] / (runs[run]['f_melt'] +
                                                     (1 - runs[run]['f_melt']) * (1 - runs[run]['f_esc']))

print(
    f"saturation index: {saturation_index}\n"
    f"kinetic fractionation factor: {alpha_kin}\n"
    f"evaporative fractionation factor: {alpha_evap}\n"
)


def initial_vaporization(f, delta_initial):
    residual_melt = (f ** (alpha_evap - 1) - 1) * 1000
    extract_vapor = ((1 - f ** alpha_evap) / (1 - f) - 1) * 1000
    return {
        'residual melt': delta_initial + residual_melt,
        'extract vapor': delta_initial + extract_vapor,
    }


def physical_fractionation(delta_retained_vapor, run):
    retained_vapor = delta_retained_vapor + (
                (1000 + delta_retained_vapor) * ((1 - run['f_esc']) ** (alpha_phys - 1) - 1))
    escaping_vapor = delta_retained_vapor + ((1000 + delta_retained_vapor) *
                                             (alpha_phys * (1 - run['f_esc']) ** (alpha_phys - 1) - 1))
    return {
        'retained vapor': retained_vapor,
        'escaping vapor': escaping_vapor,
    }


def calc_f_melt_vap(x, run):
    # return f_melt / (f_melt + x * (1 - f_melt) * (1 - f_esc))
    return run['f_melt'] / (run['f_melt'] + x * (1 - run['f_melt']) * (1 - run['f_esc']))


def two_reservoir_mixing(delta_residual_melt, delta_retained_vapor, x_vap_recondense, run):
    f = calc_f_melt_vap(x_vap_recondense, run)
    return delta_residual_melt * f + delta_retained_vapor * (1 - f)


# # do some preliminary calcs
# delta_residual_vap = initial_vaporization(f_melt, delta_i['delta_i,BSE'])
# delta_phys_frac_vapor = physical_fractionation(delta_residual_vap['extract vapor'])
# d = {
#     "delta_i,residual melt": delta_residual_vap['residual melt'],
#     "delta_i,extract vapor": delta_residual_vap['extract vapor'],
#     "delta_i,retained vapor": delta_phys_frac_vapor['retained vapor'],
#     "delta_i,escaping vapor": delta_phys_frac_vapor['escaping vapor'],
#     "delta_i,mixed (no physical fractionation)": two_reservoir_mixing(
#         delta_residual_vap['residual melt'], delta_residual_vap['extract vapor'], 1
#     ),
#     "delta_i,mixed (physical fractionation)": two_reservoir_mixing(
#         delta_residual_vap['residual melt'], delta_phys_frac_vapor['retained vapor'], 1
#     ),
#     'full recondensation f_melt/vap': calc_f_melt_vap(1),
# }

f = np.array(list(np.arange(0.0001, 0.1, 0.001)) + list(np.arange(0.1, 1, 0.01)))

# make a 2 column 1 row figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey='all')
axs = axs.flatten()
letters = string.ascii_lowercase
# ================================= Initial Setup =================================
for ax in axs:
    ax.axhline(y=delta_i['delta_i,Lunar-BSE'], color='grey', linewidth=2.0)
    ax.axhspan(
        delta_i['delta_i,Lunar-BSE'] - delta_i['delta_i,Lunar-BSE (+/-err)'],
        delta_i['delta_i,Lunar-BSE'] + delta_i['delta_i,Lunar-BSE (+/-err)'],
        alpha=0.2, color='k'
    )
    # turn on minor ticks
    ax.minorticks_on()
    ax.grid()
    ax.set_ylim([-0.3, 0.7])
    for name, run in runs.items():
        ax.plot(
            [], [], color=run['color'], linewidth=2.0, label=name
        )
    # alphabetically label each subplot in the top left corner
    ax.text(
        0.05, 0.9, letters[axs.tolist().index(ax)], transform=ax.transAxes,
        size=16, weight='bold'
    )

# ================================= Plot Residual Melt / Vapor Extract =================================
axs[0].plot(
    f * 100, np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'])['residual melt'] for f_i in f]) -
    delta_i['delta_i,BSE'], linewidth=2.0, color="k", linestyle="solid", label=r'$\delta_{\rm K, residual\ melt}$'
)
# get the bounds where this line intersects the shaded region
axs[0].plot(
    f * 100, np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'])['extract vapor'] for f_i in f]) -
    delta_i['delta_i,BSE'], linewidth=2.0, color="k", linestyle="dashdot", label=r'$\delta_{\rm K, vapor\ extract}$'
)
# plot the vertical line at f_melt
for name, run in runs.items():
    axs[0].axvline(x=run["f_melt"] * 100, color=run['color'], linewidth=2.0)
axs[0].set_xlabel(r'VMF$_{\rm K}$ (%)')
axs[0].set_ylabel(r'$\Delta_{\rm K, Lunar-BSE}$ ($\perthousand$)')
axs[0].legend(fontsize=12)

# ================================= Plot Recondensation =================================
# axs[1].plot(
#     f * 100, np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'])['residual melt'] for f_i in f]) -
#     delta_i['delta_i,BSE'],
#     linewidth=2.0, label='Residual Melt'
# )
for index, (name, run) in enumerate(runs.items()):
    label1 = r'$\delta_{\rm K, disk}$' + " (Physical Fractionation)"
    label2 = r'$\delta_{\rm K, disk}$' + " (No Physical Fractionation)"
    if index != 0:
        axs[1].plot(
            [], [], linewidth=2.0, color='k', linestyle="solid", label=label1
        )
        axs[1].plot(
            [], [], linewidth=2.0, color='k', linestyle="dashdot", label=label2
        )
    delta_residual_vap = initial_vaporization(run['f_melt'], delta_i['delta_i,BSE'])
    axs[1].plot(
        f * 100, np.array([two_reservoir_mixing(delta_residual_vap['residual melt'],
                    physical_fractionation(delta_i['delta_i,Lunar'], run)['retained vapor'], x, run) for x in f]) -
        delta_i['delta_i,BSE'], linewidth=2.0, color=run['color']
    )
    axs[1].plot(
        f * 100,
        np.array([two_reservoir_mixing(delta_residual_vap['residual melt'], delta_residual_vap['extract vapor'], x, run)
        for x in f]) - delta_i['delta_i,BSE'], linewidth=2.0, color=run['color'], linestyle="dashdot"
    )
axs[1].set_xlabel(r'x (%)')
axs[1].set_xscale('log')
axs[1].set_xlim(10 ** -2, 40)
axs[1].legend(fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig("k_isotope_fractionation.png", format='png', dpi=200, bbox_inches='tight')

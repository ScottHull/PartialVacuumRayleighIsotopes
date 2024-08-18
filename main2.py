import numpy as np
import string
import matplotlib.pyplot as plt
import labellines as ll
from random import randint

# use the colorblind color package
plt.style.use('seaborn-colorblind')
# use 14 size font for all text
plt.rcParams.update({'font.size': 18})
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
# saturation_index = 0.989  # P_i / P_sat
alpha_kin = (39 / 41) ** 0.43  # the kinetic fractionation factor
alpha_phys = np.sqrt(39 / 41)
# f_melt_vap = f_melt / (f_melt + (1 - f_melt) * (1 - f_esc))
runs = {
    "Canonical": {
        # "f_melt": 0.191,
        "f_melt": 0.8741,
        "f_esc": 0.74,
        "color": prop_cycle[0],
        # "linestyle": "solid",
    },
    "Half-Earths": {
        # "f_melt": 0.504,
        "f_melt": 0.8132,
        "f_esc": 0.16,
        "color": prop_cycle[1],
        # "linestyle": "dashed",
    },
}
for run in runs:
    runs[run]['f_melt_vap'] = runs[run]['f_melt'] / (runs[run]['f_melt'] +
                                                     (1 - runs[run]['f_melt']) * (1 - runs[run]['f_esc']))

# print(
#     f"saturation index: {saturation_index}\n"
#     f"kinetic fractionation factor: {alpha_kin}\n"
#     f"evaporative fractionation factor: {alpha_evap}\n"
# )


def initial_vaporization(f, delta_initial, saturation_index=0.989):
    alpha_evap = 1 + (1 - saturation_index) * (alpha_kin - 1)  # the evaporative fractionation factor
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
fig, axs = plt.subplots(1, 1, figsize=(8, 8), sharey='all')
axs = [axs]
letters = string.ascii_lowercase
# ================================= Initial Setup =================================
for ax in axs:
    actual_line = ax.axhline(y=delta_i['delta_i,Lunar-BSE'], color='grey', linewidth=2.0)
    # Add the label to the grey "Actual" line
    ll.labelLine(actual_line, x=0.5, label='Actual')

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
    # ax.text(
    #     0.95, 0.05, letters[axs.tolist().index(ax)], transform=ax.transAxes,
    #     size=16, weight='bold'
    # )

# ================================= Plot Residual Melt / Vapor Extract =================================
residual_melt_line = axs[0].plot(
    (1 - f), np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'])['residual melt'] for f_i in f]) -
    delta_i['delta_i,BSE'], linewidth=2.0, color="k", linestyle="solid"
)
ll.labelLine(residual_melt_line[0], x=0.5, label=r'($S_{\rm K} = 0.99$)', fontsize=16)

# get the bounds where this line intersects the shaded region
extract_vapor_line = axs[0].plot(
    (1 - f), np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'])['extract vapor'] for f_i in f]) -
    delta_i['delta_i,BSE'], linewidth=2.0, color="k", linestyle="dashdot"
)
ll.labelLine(extract_vapor_line[0], x=0.5, label=r'($S_{\rm K} = 0.99$)', fontsize=16)

axs[0].fill_between(
    (1 - f),
    np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'], saturation_index=0.82)['residual melt'] for f_i in f]) -
    delta_i['delta_i,BSE'],
    np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'], saturation_index=0.92)['residual melt'] for f_i in f]) -
    delta_i['delta_i,BSE'],
    color='cyan',
    alpha=0.3
)

s_082 = axs[0].plot(
    (1 - f), np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'], saturation_index=0.82)['residual melt'] for f_i in f]) -
    delta_i['delta_i,BSE'], linewidth=2.0, color="black", linestyle="solid", label=r'($S_{\rm K} = 0.82$)'
)

s_092 = axs[0].plot(
    (1 - f), np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'], saturation_index=0.92)['residual melt'] for f_i in f]) -
    delta_i['delta_i,BSE'], linewidth=2.0, color="black", linestyle="solid", label=r'($S_{\rm K} = 0.92$)'
)

ll.labelLine(s_082[0], x=0.08, fontsize=16, label=r'($S_{\rm K} = 0.82$)')
ll.labelLine(s_092[0], x=0.20, fontsize=16, label=r'($S_{\rm K} = 0.92$)')

# plot the vertical line at f_melt
for name, run in runs.items():
    axs[0].axvline(x=(1 - run["f_melt"]), color=run['color'], linewidth=4.0)
axs[0].set_xlabel(r'VMF$_{\rm K}$ (%)')
axs[0].set_ylabel(r'$\Delta {\rm ^{41}K}$ ($\perthousand$)')
axs[0].legend(fontsize=12, loc='lower right')

# ================================= Plot Recondensation =================================
# axs[1].plot(
#     (1 - f), np.array([initial_vaporization(f_i, delta_i['delta_i,BSE'])['residual melt'] for f_i in f]) -
#     delta_i['delta_i,BSE'],
#     linewidth=2.0, label='Residual Melt'
# )
for index, (name, run) in enumerate(runs.items()):
    label1 = r'$\delta_{\rm K, disk}$' + " (Physical Fractionation)"
    label2 = r'$\delta_{\rm K, disk}$' + " (No Physical Fractionation)"
    delta_residual_vap = initial_vaporization(run['f_melt'], delta_i['delta_i,BSE'])
    physically_fractionated_vapor = physical_fractionation(delta_residual_vap['extract vapor'], run)

# axs[0].set_xlim(0, 0.9)
# axs[0].set_title("Initial Vaporization")
# increase the linewidth in each legend
for ax in axs:
    for line in ax.get_lines():
        line.set_linewidth(3.0)

# # Add annotation for the cyan line
# axs[0].annotate(
#     r'$\mathbf{0.7 \leq S_{\rm K} \leq 0.8}$',
#     xy=(22 / 100, 0.55),  # Adjust the coordinates to point inside the shaded cyan region
#     xytext=(38 / 100, 0.50),  # Adjust the text position as needed
#     arrowprops=dict(facecolor='cyan', edgecolor='black', linewidth=1.5, shrink=0.05),
#     fontsize=22,  # Increase font size
#     color='cyan',
#     fontweight='bold'  # Make the font bold
# )

# Add annotation for the black line
# axs[0].annotate(
#     r'$\mathbf{S = 0.99}$',
#     xy=(46 / 100, 0.14),  # Adjust the coordinates to point inside the shaded cyan region
#     xytext=(70 / 100, 0.08),  # Adjust the text position as needed
#     arrowprops=dict(facecolor='black', edgecolor='black', linewidth=1.5, shrink=0.05),
#     fontsize=22,  # Increase font size
#     color='black',
#     fontweight='bold'  # Make the font bold
# )

# Set x-axis labels to range from 0 to 100
for ax in axs:
    ax.set_xticks(np.linspace(0, 1, 11))  # Set tick positions from 0 to 1 with 10 intervals
    ax.set_xticklabels([str(int(x * 100)) for x in np.linspace(0, 1, 11)])  # Set tick labels from 0 to 100

axs[0].set_xlim(0, 0.9)

plt.tight_layout()
# plt.show()
plt.savefig("k_isotope_fractionation_single.png", format='png', dpi=200, bbox_inches='tight')

# go through each run, print the residual melt and vapor values
for run in runs:
    residual_melt = initial_vaporization(runs[run]['f_melt'], delta_i['delta_i,BSE'])['residual melt'] - delta_i['delta_i,BSE']
    vapor_extract = initial_vaporization(runs[run]['f_melt'], delta_i['delta_i,BSE'])['extract vapor'] - delta_i['delta_i,BSE']
    retained_physically_fractionated_vapor = physical_fractionation(vapor_extract + delta_i['delta_i,BSE'], runs[run])['retained vapor'] - delta_i['delta_i,BSE']
    escaping_physically_fractionated_vapor = physical_fractionation(vapor_extract + delta_i['delta_i,BSE'], runs[run])['escaping vapor'] - delta_i['delta_i,BSE']
    print(
        f"{run}:\n"
        f"residual melt: {residual_melt}\n"
        f"vapor extract: {vapor_extract}\n"
        f"retained physically fractionated vapor: {retained_physically_fractionated_vapor}\n"
        f"escaping physically fractionated vapor: {escaping_physically_fractionated_vapor}\n"
    )

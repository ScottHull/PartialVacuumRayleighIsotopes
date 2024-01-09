import numpy as np
import string
import matplotlib.pyplot as plt

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
saturation_index = 0.001  # P_i / P_sat
alpha_kin = (39 / 41) ** 0.43  # the kinetic fractionation factor
alpha_evap = 1 + (1 - saturation_index) * (alpha_kin - 1)  # the evaporative fractionation factor
alpha_phys = np.sqrt(39 / 41)
# f_melt_vap = f_melt / (f_melt + (1 - f_melt) * (1 - f_esc))
runs = {
    "BSM": {
            "G": {
                "f_melt": 0.1110142203842716,
                "f_esc": 92 / 100,
                "impactor%": 70.758 / 100,  # %
                "color": prop_cycle[0],
                # "linestyle": "solid",
            },
            "H": {
                "f_melt": 0.6161276097168675,
                "f_esc": 88 / 100,
                "impactor%": 65.81413641 / 100,  # %
                "color": prop_cycle[1],
                # "linestyle": "solid",
            },
            "K": {
                "f_melt": 0.7661014262659456,
                "f_esc": 89 / 100,
                "impactor%": 16.6080938 / 100,  # %
                "color": prop_cycle[2],
                # "linestyle": "solid",
            },
            "L": {
                "f_melt": 0.696745959557069,
                "f_esc": 85 / 100,
                "impactor%": 71.23136432 / 100,  # %
                "color": prop_cycle[3],
                # "linestyle": "solid",
            },
        },
    "D-type": {
        "G": {
            "f_melt": 0.4747236882995573,
            "f_esc": 92 / 100,
            "impactor%": 70.758 / 100,  # %
            "color": prop_cycle[0],
            # "linestyle": "solid",
        },
        "H": {
            "f_melt": 0.7503664568288573,
            "f_esc": 88 / 100,
            "impactor%": 65.81413641 / 100,  # %
            "color": prop_cycle[1],
            # "linestyle": "solid",
        },
        "K": {
            "f_melt": 0.8537028910311829,
            "f_esc": 89 / 100,
            "impactor%": 16.6080938 / 100,  # %
            "color": prop_cycle[2],
            # "linestyle": "solid",
        },
        "L": {
            "f_melt": 0.8068516595175024,
            "f_esc": 88 / 100,
            "impactor%": 71.23136432 / 100,  # %
            "color": prop_cycle[3],
            # "linestyle": "solid",
        },
    },
    "Mixed": {
        "G": {
            "f_melt": 0.4031781207842613,
            "f_esc": 92 / 100,
            "impactor%": 70.758 / 100,  # %
            "color": prop_cycle[0],
            # "linestyle": "solid",
        },
        "H": {
            "f_melt": 0.7171408019555808,
            "f_esc": 88 / 100,
            "impactor%": 65.81413641 / 100,  # %
            "color": prop_cycle[1],
            # "linestyle": "solid",
        },
        "K": {
            "f_melt": 0.7777525051897846,
            "f_esc": 89 / 100,
            "impactor%": 16.6080938 / 100,  # %
            "color": prop_cycle[2],
            # "linestyle": "solid",
        },
        "L": {
            "f_melt": 0.7867401495665421,
            "f_esc": 85 / 100,
            "impactor%": 71.23136432 / 100,  # %
            "color": prop_cycle[3],
            # "linestyle": "solid",
        },
    },
}

f = np.array(list(np.arange(0.0001, 0.1, 0.001)) + list(np.arange(0.1, 1, 0.01)))

def initial_vaporization(f, delta_initial):
    residual_melt = (f ** (alpha_evap - 1) - 1) * 1000
    extract_vapor = ((1 - f ** alpha_evap) / (1 - f) - 1) * 1000
    return {
        'offset residual melt': residual_melt,
        'offset extract vapor': extract_vapor,
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


# make a len(runs) rows x 2 columns figure
fig, axs = plt.subplots(len(runs), 2, figsize=(10, len(runs) * 5), sharex='all')
for initial_comp_index, (initial_comp_name, initial_comp) in enumerate(runs.items()):
    axs[initial_comp_index, 0].text(
        0.65, 0.90, f"{initial_comp_name}", transform=axs[initial_comp_index, 0].transAxes, fontsize=22, fontweight='bold'
    )
    axs[initial_comp_index, 0].plot(
        f * 100, [initial_vaporization(f_i, delta_i['delta_i,BSE'])['offset residual melt'] for f_i in f], linewidth=2.0, color='black', alpha=1
    )
    for run_index, (run_name, run) in enumerate(initial_comp.items()):
        run['f_melt_vap'] = run['f_melt'] / (run['f_melt'] + (1 - run['f_melt']) * (1 - run['f_esc']))
        initial_vap_result = initial_vaporization(run['f_melt'], delta_i['delta_i,BSE'])
        axs[initial_comp_index, 0].scatter(
            run['f_melt'] * 100, initial_vap_result['offset residual melt'],
            marker='o', color=run['color'], s=160, alpha=1, label=f'Run {run_name}'
        )
        axs[initial_comp_index, 1].plot(
            f * 100,
            np.array([two_reservoir_mixing(initial_vap_result['offset residual melt'],
                                           physical_fractionation(delta_i['delta_i,Lunar'], run)['retained vapor'], x, run) for x in f]),
            linewidth=3.0, color=run['color']
        )
        axs[initial_comp_index, 1].plot(
            f * 100,
            np.array([two_reservoir_mixing(initial_vap_result['offset residual melt'], initial_vap_result['offset extract vapor'], x, run)
            for x in f]), linewidth=3.0, color=run['color'], linestyle="dashdot"
        )

letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs.flatten()):
    ax.grid()
    if index % 2 == 0:
        # ax.set_yscale('log')
        ax.set_ylim(0, 55)
        ax.set_ylabel(r'$\Delta \rm ^{41}K$')
    ax.text(0.05, 0.9, letters[index], transform=ax.transAxes, fontsize=18, fontweight='bold')

axs.flatten()[-2].set_xlabel(r'$f_{\rm melt}$ (%)')
axs.flatten()[-1].set_xlabel('% Recondensed')
axs.flatten()[0].set_title("Impact Vaporization", fontsize=18)
axs.flatten()[1].set_title("Retained Vapor Recondensation", fontsize=18)

axs[0, 0].legend(loc='lower left', fontsize=16)

plt.tight_layout()
# plt.show()
plt.savefig("mars_k_isotopes.png", format='png', dpi=200)

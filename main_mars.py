import os
import numpy as np
import pandas as pd
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
alpha_kin = (39 / 41) ** 0.43  # the kinetic fractionation factor
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

saturation_indices = {
    'G': 0.07153514070532385,
    'H': 0.102519441414331,
    'K': 0.04789976312884592,
    'L': 0.04141597509920847,
    # 'G': 0.8,
    # 'H': 0.8,
    # 'K': 0.8,
    # 'L': 0.8,
}

f = np.array(list(np.arange(0.0001, 0.1, 0.001)) + list(np.arange(0.1, 1, 0.01)))

def initial_vaporization(run_name, f, delta_initial, saturation_index=None):
    if saturation_index is None:
        saturation_index = saturation_indices[run_name]
    alpha_evap = 1 + (1 - saturation_index) * (alpha_kin - 1)  # the evaporative fractionation factor
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

global_solutions = {"run_name": [], 'f_melt': [], 'initial vaporization melt': [], 'initial vaporization vapor': [],
                    # 'no recondensation no phys frac': [],
                    'full recondensation no phys frac': [],
                    # 'no recondensation with phys frac': [],
                    'full recondensation with phys frac': []}

# make a len(runs) rows x 2 columns figure
fig, axs = plt.subplots(len(runs), 2, figsize=(10, len(runs) * 5), sharex='all')
for initial_comp_index, (initial_comp_name, initial_comp) in enumerate(runs.items()):
    axs[initial_comp_index, 0].text(
        0.65, 0.90, f"{initial_comp_name}", transform=axs[initial_comp_index, 0].transAxes, fontsize=22, fontweight='bold'
    )
    # TODO: re-enable this!
    # shade the region between the min and max saturation index
    axs[initial_comp_index, 0].fill_between(
        f * 100, [initial_vaporization("", f_i, delta_i['delta_i,BSE'], saturation_index=min(saturation_indices.values()))['offset residual melt'] for f_i in f],
        [initial_vaporization("", f_i, delta_i['delta_i,Lunar'], saturation_index=max(saturation_indices.values()))['offset residual melt'] for f_i in f],
        color='grey', alpha=0.5
    )
    # axs[initial_comp_index, 0].plot(
    #     f * 100, [initial_vaporization(f_i, delta_i['delta_i,BSE'])['offset residual melt'] for f_i in f], linewidth=2.0, color='black', alpha=1
    # )
    for run_index, (run_name, run) in enumerate(initial_comp.items()):
        global_solutions['run_name'].append(f"{run_name} ({initial_comp_name})")
        global_solutions['f_melt'].append(f"{run['f_melt']:.2f}")
        run['f_melt_vap'] = run['f_melt'] / (run['f_melt'] + (1 - run['f_melt']) * (1 - run['f_esc']))
        initial_vap_result = initial_vaporization(run_name, run['f_melt'], delta_i['delta_i,BSE'])
        global_solutions['initial vaporization melt'].append(f"{initial_vap_result['offset residual melt']:.2f}")
        global_solutions['initial vaporization vapor'].append(f"{initial_vap_result['offset extract vapor']:.2f}")
        recondensation_no_phys_frac = np.array([two_reservoir_mixing(initial_vap_result['offset residual melt'], initial_vap_result['offset extract vapor'], x, run)
            for x in f])
        # recondensation_w_phys_frac = np.array([two_reservoir_mixing(initial_vap_result['offset residual melt'],
        #                                    physical_fractionation(delta_i['delta_i,Lunar'], run)['retained vapor'], x, run) for x in f])
        recondensation_w_phys_frac = np.array([two_reservoir_mixing(initial_vap_result['offset residual melt'],
                                           physical_fractionation(initial_vap_result['offset extract vapor'], run)['retained vapor'], x, run) for x in f])
        # global_solutions['no recondensation no phys frac'].append(recondensation_no_phys_frac[0])
        global_solutions['full recondensation no phys frac'].append(f"{recondensation_no_phys_frac[-1]:.2f}")
        # global_solutions['no recondensation with phys frac'].append(recondensation_w_phys_frac[0])
        global_solutions['full recondensation with phys frac'].append(f"{recondensation_w_phys_frac[-1]:.2f}")
        axs[initial_comp_index, 0].scatter(
            run['f_melt'] * 100, initial_vap_result['offset residual melt'],
            marker='o', color=run['color'], s=160, alpha=1, label=f'Run {run_name}'
        )
        axs[initial_comp_index, 1].plot(
            f * 100, recondensation_w_phys_frac, linewidth=3.0, color=run['color']
        )
        axs[initial_comp_index, 1].plot(
            f * 100, recondensation_no_phys_frac, linewidth=3.0, color=run['color'], linestyle="dashdot"
        )

df = pd.DataFrame(global_solutions).to_latex(index=False)
if "mars_isotopes.tex" in os.listdir():
    os.remove("mars_isotopes.tex")
with open("mars_isotopes.tex", "w") as f:
    f.write(df)
f.close()

letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs.flatten()):
    ax.grid()
    if index % 2 == 0:
        # ax.set_yscale('log')
        ax.set_ylim(0, 55)
        ax.set_ylabel(r'$\Delta \rm ^{41}K$')
        # flip the x-axis
        ax.invert_xaxis()
    ax.text(0.05, 0.9, letters[index], transform=ax.transAxes, fontsize=18, fontweight='bold')

axs.flatten()[-2].set_xlabel(r'$f_{\rm melt}$ (%)')
axs.flatten()[-1].set_xlabel('% Recondensed')
axs.flatten()[0].set_title("Impact Vaporization", fontsize=18)
axs.flatten()[1].set_title("Retained Vapor Recondensation", fontsize=18)

axs[0, 0].legend(loc='lower right', fontsize=16)

plt.tight_layout()
# plt.show()
plt.savefig("mars_k_isotopes.png", format='png', dpi=200)

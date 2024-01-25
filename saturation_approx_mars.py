from math import erfc
import numpy as np
import matplotlib.pyplot as plt

# use the colorblind color package
plt.style.use('seaborn-colorblind')

runs = {
    'Run G': {
        'T': 2201.89,
        'P': 14.08,
        'U_inf': 4.56e3,
        'sigma_ik': 3 * 10 ** -10,
        'Omega_ik': 0.7,
        'M_k': 44.08 / 1000,
        'm_i': 39.0983 * (1.67 * 10 ** -27),
        'm_k': 44.08 * (1.67 * 10 ** -27),
        # 'L': 3389.5 * 1000,
        'L': 11.2 * 1000,
        'gamma_i,cn': 1,
        'xi': 500,
    },
    'Run H': {
        'T': 2271.61,
        'P': 30.42,
        'U_inf': 4.42e3,
        'sigma_ik': 3 * 10 ** -10,
        'Omega_ik': 0.7,
        'M_k': 44.08 / 1000,
        'm_i': 39.0983 * (1.67 * 10 ** -27),
        'm_k': 44.08 * (1.67 * 10 ** -27),
        # 'L': 3389.5 * 1000,
        'L': 11.2 * 1000,
        'gamma_i,cn': 1,
        'xi': 500,
    },
    'Run K': {
        'T': 2213.51,
        'P': 5.88,
        'U_inf': 4.44e3,
        'sigma_ik': 3 * 10 ** -10,
        'Omega_ik': 0.7,
        'M_k': 44.08 / 1000,
        'm_i': 39.0983 * (1.67 * 10 ** -27),
        'm_k': 44.08 * (1.67 * 10 ** -27),
        # 'L': 3389.5 * 1000,
        'L': 11.2 * 1000,
        'gamma_i,cn': 1,
        'xi': 500,
    },
    'Run L': {
        'T': 2103.03,
        'P': 4.10,
        'U_inf': 4.30e3,
        'sigma_ik': 3 * 10 ** -10,
        'Omega_ik': 0.7,
        'M_k': 44.08 / 1000,
        'm_i': 39.0983 * (1.67 * 10 ** -27),
        'm_k': 44.08 * (1.67 * 10 ** -27),
        # 'L': 3389.5 * 1000,
        'L': 11.2 * 1000,
        'gamma_i,cn': 1,
        'xi': 500,
    },
}

gas_constant = 8.31446261815324  # J/mol K
C_Pk = 4 * gas_constant
k_B = 1.380649 * 10 ** -23  # J/K  # Boltzmann constant

def rho_k(run):
    run['rho_k'] = (run['M_k'] * run['P']) / (gas_constant * run['T'])
    return run['rho_k']

def eta_k(run):
    a = 5 / (16 * np.pi ** 0.5)
    b = (run['m_k'] * k_B * run['T']) ** 0.5 / (run['sigma_ik'] ** 2 * run['Omega_ik'])
    run['eta_k'] = a * b
    return run['eta_k']

def K_k(run):
    a = C_Pk + (5 / 4) * gas_constant
    b = run['eta_k'] / run['M_k']
    run['K_k'] = a * b
    return run['K_k']

def reynolds_number(run):
    run['Re'] = (run['rho_k'] * run['U_inf'] * run['L']) / run['eta_k']
    return run['Re']

def prandtl_number(run):
    run['Pr'] = (run['eta_k'] * C_Pk) / run['K_k']
    return run['Pr']

def diffusion_coefficient(run):
    a = 3 / (8 * run['P'] * run['sigma_ik'] ** 2 * run['Omega_ik'])
    b = (k_B * run['T']) ** 3 / (2 * np.pi)
    c = (run['m_i'] + run['m_k']) / (run['m_i'] * run['m_k'])
    run['D_ik'] = a * np.sqrt(b * c)
    return run['D_ik']

def k_c(run, r):
    a = run['P'] / (gas_constant * run['T'])
    b = run['D_ik'] ** (2 / 3) / r
    c = (run['K_k'] / (run['rho_k'] * C_Pk)) ** (1 / 3)
    d = 0.3 * (run['U_inf'] * run['L']) ** 0.5 * (run['rho_k'] / run['eta_k']) ** (1 / 6)
    run['k_c'] = a * b * (c + d)
    return run['k_c']

def nusselt_number(run):
    run['Nu'] = 2 + 0.6 * run['Re'] ** 0.5 * run['Pr'] ** (1 / 3)
    return run['Nu']

def _lambda(run, r):
    run['lambda'] = r / run['Nu']
    return run['lambda']

def saturation_index(run):
    a = (run['gamma_i,cn'] * run['P']) / (run['k_c'] * np.sqrt(2 * np.pi * gas_constant * run['M_k'] * run['T']))
    b = 1 - np.exp(run['xi']) * erfc(np.sqrt(run['xi']))
    run['saturation_index'] = 1 - (1 / (1 + a * b))
    return run['saturation_index']

radii = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000,
                  500000, 1000000, 5000000])

target = 0.989
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axhline(y=target, color='red', linewidth=2.0, label=fr"$S_i = {target}$")
for index, (name, run) in enumerate(runs.items()):
    # if index == 0:
    #     ax.axvline(run['L'] / 1000, color='grey', linestyle='--', linewidth=2.0, label="Lunar Radius")
    S_as_func_of_r = []
    # for r in [1.5]:
    for r in radii:
        rho_k(run)
        eta_k(run)
        K_k(run)
        reynolds_number(run)
        prandtl_number(run)
        diffusion_coefficient(run)
        k_c(run, r)
        nusselt_number(run)
        _lambda(run, r)
        saturation_index(run)
        S_as_func_of_r.append(run['saturation_index'])
        print(f'{name} at r = {r} m: {run["saturation_index"]}')
        # print(run)
    ax.plot(radii / 1000, S_as_func_of_r, label=name)
#
ax.axvline(1.5 / 1000, color='black', linestyle='--', linewidth=2.0)
ax.set_xlabel('r (km)')
ax.set_ylabel(r'$S_{\rm K}$')
ax.set_xscale('log')
ax.grid()
ax.legend()
# turn on minor for both axes
ax.minorticks_on()
plt.tight_layout()
# plt.show()
plt.savefig("S_i_vs_r_mars.png", format='png', dpi=200, bbox_inches='tight')

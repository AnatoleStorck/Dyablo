from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import unyt as u
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root

import pyablo

import sys


xmf_filename = sys.argv[1]  # "test_sod_2D_main.xmf"
target_precision = float(sys.argv[2])
png_filename = sys.argv[3]
# Solar metallicity in Nordlund+2014, keep consistent with test_cooling.ini
Zsun = 0.014
Z = 5 * Zsun
print("Validate Cooling")
print(f"XMF filename : {xmf_filename}")
print(f"Target Precision : {target_precision}")
print(f"PNG output filename : {png_filename}")
print(f"Assuming metallicity : {Z=} = {Z/Zsun} Zsun, with {Zsun=}")

# --------------------------------
# Read cooling table
with h5py.File("analytical_cooling_table.h5") as f:
    log_nH_grid = f["CoolingRates/Primordial/MMW"].attrs["Parameter1"]
    # Reverse redshift grid to have increasing order
    redshift_grid = f["CoolingRates/Primordial/MMW"].attrs["Parameter2"]
    T_grid = f["CoolingRates/Primordial/MMW"].attrs["Temperature"]

    redshift_grid[-1] += 1e-5

    cooling = RegularGridInterpolator(
        (log_nH_grid, redshift_grid, T_grid),
        f["CoolingRates/Primordial/Cooling"][:] + f["CoolingRates/Metals/Cooling"][:] * Z / Zsun,
        bounds_error=False,
        method="linear",
    )
    heating = RegularGridInterpolator(
        (log_nH_grid, redshift_grid, T_grid),
        f["CoolingRates/Primordial/Heating"][:] + f["CoolingRates/Metals/Heating"][:] * Z / Zsun,
        bounds_error=False,
        method="linear",
    )
    mu = RegularGridInterpolator(
        (log_nH_grid, redshift_grid, T_grid),
        f["CoolingRates/Primordial/MMW"][:],
        bounds_error=False,
        method="linear",
    )

# Open output file
reader = pyablo.XdmfReader()
series = reader.readTimeSeries(xmf_filename)
outputs = [fname.replace(".xmf", ".h5") for fname in series]



def read_output(filename):
    gamma0 = 5 / 3
    time_unit = 10 * u.Myr
    length_unit = 100 * u.pc
    velocity_unit = 1 * length_unit / time_unit
    rho_unit = u.mp / u.cm**3
    energy_unit = rho_unit * velocity_unit**2

    with h5py.File(filename) as f:
        attrs = dict(f["scalar_data"].attrs)
        rho = f["rho"][:] * rho_unit
        e_tot = f["e_tot"][:] * energy_unit
        vx = f["rho_vx"][:] / rho.d * velocity_unit
        vy = f["rho_vy"][:] / rho.d * velocity_unit
        vz = f["rho_vz"][:] / rho.d * velocity_unit

    P = (e_tot - 0.5 * rho * (vx**2 + vy**2 + vz**2)) * (gamma0 - 1)

    T_over_mu = P / rho * u.mp / u.kb

    return {"attrs": attrs, "rho": rho, "P": P, "T_over_mu": T_over_mu}


data = [read_output(output) for output in outputs]


def dT_dt(_t, T, rho, z):
    rho = rho * u.mp / u.cm**3
    nH = (rho * 0.76 / u.mp).to("1/cm**3")
    log_nH = np.log10(nH)

    H = heating((log_nH, z, T)) * u.erg * u.cm**3 / u.s
    C = cooling((log_nH, z, T)) * u.erg * u.cm**3 / u.s
    µ = mu((log_nH, z, T))

    dT_dt = 2 * µ / (3 * u.kb * nH) * ((H - C) * nH * nH)

    return dT_dt.to("K/s").d


rho0 = data[0]["rho"][0].to("mp/cm**3").d
nH0 = rho0 * 0.76
T_over_mu0 = data[0]["T_over_mu"][0].to("K").d

def mu_metals(T):
    rhoZ = rho0 * Z
    mui = mu((np.log10(nH0), 0, T))
    return rho0 / (rho0 / mui + rhoZ / 16)

# Convert T/µ to T
T_sim = np.array(
    [
        root(lambda T: T / mu_metals(T) - Tµ, Tµ).x[0]
        for Tµ in [float(dt["T_over_mu"][0].to("K")) for dt in data]
    ]
)
print(f"Initial: {T_sim[0]:.1f} K, nH = {nH0:.2e} cm^-3")
z0 = 0

# Find "exact" solution by integrating the ODE
t_eval = (np.array([dt["attrs"]["time"] for dt in data]) * 10 * u.Myr).to("s")

result = solve_ivp(dT_dt, (0, t_eval[-1]), [T_sim[0]], args=(rho0, z0), t_eval=t_eval)
T_exp = result.y[0]

# Make sure the simulation matches the ODE solution
# There may be relatively large difference close to the stiffest points, remove them
dlogT_dlogt = np.gradient(np.log10(T_exp), np.log10(t_eval.to("Myr")))
dlogT_dlogt[:2] = 0  # Ignore first two points (nan on time)
mask = np.abs(dlogT_dlogt) < 20   # 1 dex of T per dex of time
nan_it = np.where(mask, 1, np.nan)
nan_aint = np.where(~mask, 1, np.nan)

dT = np.linalg.norm((T_sim[mask] - T_exp[mask]) / (T_sim[mask] + T_exp[mask]) * 2)
print(f"χ² difference : {dT:.2e}")

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(t_eval.to("Myr"), T_exp, label="ODE solver", c="gray")
ax.plot(t_eval.to("Myr"), T_sim * nan_it, "+", c="k", label="Simulation")
ax.plot(
    t_eval.to("Myr"), T_sim * nan_aint, "+", c="red", label="Simulation (stiff part)"
)

Teq = root(
    lambda T: heating((np.log10(nH0), z0, T)) - cooling((np.log10(nH0), z0, T)), 1e4
).x[0]

ax.axhline(Teq, ls=":", c="gray", label=f"Equilibrium T = {Teq:.1f} K")

ax.legend()
ax.set(
    xlabel="Time [Myr]",
    ylabel=r"$T$ [K]",
    yscale="log",
    xlim=(4e-2, 1e2),
    xscale="log",
)
fig.savefig(png_filename)

np.testing.assert_allclose(np.log10(T_sim[mask]), np.log10(T_exp[mask]), rtol=target_precision)

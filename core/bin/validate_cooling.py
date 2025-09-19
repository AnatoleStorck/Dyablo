from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import unyt as u
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root

plt.rcParams["figure.dpi"] = 200

with h5py.File(
    "/home/cphyc/Documents/prog/grackle_data_files/input/CloudyData_UVB=HM2012_high_density.h5"
) as f:
    log_nH_grid = f["CoolingRates/Primordial/MMW"].attrs["Parameter1"]
    # Reverse redshift grid to have increasing order
    redshift_grid = f["CoolingRates/Primordial/MMW"].attrs["Parameter2"]
    T_grid = f["CoolingRates/Primordial/MMW"].attrs["Temperature"]

    redshift_grid[-1] += 1e-5

    cooling = RegularGridInterpolator(
        (log_nH_grid, redshift_grid, T_grid),
        f["CoolingRates/Primordial/Cooling"][:],
        bounds_error=False,
        method="linear",
    )
    heating = RegularGridInterpolator(
        (log_nH_grid, redshift_grid, T_grid),
        f["CoolingRates/Primordial/Heating"][:],
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
outputs = sorted(Path(".").glob("cooling_iter*.h5"))


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

# Convert T/µ to T
T_sim = np.array(
    [
        root(lambda T: T / mu((np.log10(nH0), 0, T)) - Tµ, Tµ).x[0]
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
dT = np.linalg.norm((T_sim - T_exp) / (T_sim + T_exp) * 2)
# There may be relatively large difference close to the stiffest points, remove them
dT_dt = np.gradient(T_exp, t_eval.to("Myr"))
mask = np.abs(dT_dt) < 2e5  # K/Myr
nan_it = np.where(mask, 1, np.nan)
nan_aint = np.where(~mask, 1, np.nan)

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
fig.savefig("/tmp/verify_cooling.png")

np.testing.assert_allclose(T_sim[mask], T_exp[mask], rtol=0.05)

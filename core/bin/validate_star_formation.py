import pyablo
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.integrate import solve_ivp

import h5py

xmf_filename = sys.argv[1]
target_precision = float(sys.argv[2])
rtol = 1e-10
png_filename = sys.argv[3]

print("Validate Star Formation")
print(f"XMF filename : {xmf_filename}")
print(f"Target Precision : {target_precision}")
print(f"PNG output filename : {png_filename}")

reader = pyablo.XdmfReader()
amr_series = reader.readTimeSeries(xmf_filename)
part_series = reader.readTimeSeries(
    xmf_filename.replace("main", "particles_particles_main")
)

m_gas, m_part = [], []
m_gas_all = []
p_gas, p_part = [], []
e_gas, e_part = [], []
t = []

for amr_fname, part_fname in zip(amr_series, part_series):
    snap = reader.readSnapshot(amr_fname)
    mg = snap.getTotalMass()
    Ncells = snap.getNCells()
    cells = range(Ncells)
    vol_g = np.asarray(snap.getCellsVolume(cells))
    rhov_g = np.asarray(snap.getMomentum(cells))
    pg = (rhov_g * vol_g[:, None]).sum(axis=0)
    eg = snap.getTotalEnergy()

    with h5py.File(part_fname.replace(".xmf", ".h5")) as part_snap:
        mm = part_snap["mass"][:]
        vv = np.stack(
            [part_snap[f"v{k}"][:] for k in "xyz"],
            axis=-1,
        )
        mp = mm.sum()
        pp = (mm[:, None] * vv).sum(axis=0)
        ep = (0.5 * mm * (vv**2).sum(axis=1)).sum()

    m_gas.append(mg)
    m_gas_all.append(vol_g * np.asarray(snap.getDensity(cells)))
    m_part.append(mp)
    p_gas.append(pg)
    p_part.append(pp)
    e_gas.append(eg)
    e_part.append(ep)
    t.append(snap.getTime())

# Convert to numpy arrays
m_gas = np.asarray(m_gas)
m_part = np.asarray(m_part)
p_gas = np.asarray(p_gas)
p_part = np.asarray(p_part)
e_gas = np.asarray(e_gas)
e_part = np.asarray(e_part)
t = np.asarray(t)
m_gas_all = np.asarray(m_gas_all)

################ CONSERVATION ################
# Mass is conserved
np.testing.assert_allclose(m_gas + m_part, m_gas[0], rtol=rtol)
# Zero momentum in particles initially (no particles)
np.testing.assert_allclose(p_part[0], 0, rtol=rtol)
# Non-zero momentum in the gas
assert all(p_gas[0, :] != 0), "Some initial momentum"
# Momentum is conserved
for i in range(3):
    np.testing.assert_allclose((p_part + p_gas)[:, i], p_gas[0, i], rtol=rtol)
# Energy is conserved
np.testing.assert_allclose(e_part + e_gas, e_gas[0], rtol=rtol)


################ EXPECTED BEHAVIOUR ################


def sfr(_t, rho_gas, eps_ff):
    """Compute star formation rate

    Parameters
    ----------
    rho_gas: density, in mp/cm³

    Returns
    -------
    dM*/dt: star formation rate in mp/cm³/Myr
    """
    G = 6.67408e-11  # MKS
    n = rho_gas * 1.67262158e-21     # kg/m³

    tstar = 0.5427 / np.sqrt(G * n)  # s
    tstar /= 3.1536e+13              # Myr
    return - eps_ff * rho_gas / tstar

analytical = solve_ivp(lambda t, rho: sfr(t, rho, .1), t_span=(0, 100), y0=[100], method="RK45", t_eval=t)

# Safety check: should have been evaluated at same time as sim
np.testing.assert_allclose(analytical.t, t)

# Make sure obtained behaviour is close to analytical formula
np.testing.assert_allclose(analytical.y[0], m_gas, rtol=target_precision)

################ PLOTS ################
# Layout: conservation (left spanning both rows),
# gas vs theory (top right), error (bottom right)
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], hspace=0.4)

# Conservation plot spanning both rows in first column
ax0 = fig.add_subplot(gs[:, 0])
ax0.plot(t, (m_gas + m_part) / m_gas[0] - 1, label=r"$\Delta m$")
for i, axis in enumerate("xyz"):
    ax0.plot(
        t,
        (p_gas[:, i] + p_part[:, i]) / p_gas[0, i] - 1,
        label=r"$\Delta p_{%s}$" % axis,
    )
ax0.plot(t, (e_gas + e_part) / e_gas[0] - 1, label=r"$\Delta e$")
ax0.set(
    xlabel="Time",
    ylabel=r"Relative change $\dfrac{\Delta q}{q_0}$",
    title="Check conservation",
)
ax0.legend()

# Gas mass vs theoretical solution (top right)
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(t, m_gas, label="Simulated", lw=1.5)
ax1.plot(t, analytical.y[0], label="Theory", ls="--")
ax1.set(
    xlabel="Time",
    ylabel=r"Gas mass density $\rho_g$",
    title="Gas mass vs Theory",
)
ax1.legend()

# Error between simulation and theory (bottom right)
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(t, (m_gas - analytical.y[0]) / analytical.y[0], color="gray", lw=1)
ax2.set(
    xlabel="Time",
    ylabel="Relative error",
    title="Simulation vs Theory Error",
)

plt.tight_layout()
fig.savefig(png_filename)

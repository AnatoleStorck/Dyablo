import os
import pyablo
import numpy as np
import sys

import h5py

xmf_filename = sys.argv[1]
rtol = float(sys.argv[2])

print("Validate Star Feedback")
print(f"XMF filename : {xmf_filename}")
print(f"Target Precision : {rtol}")

if not os.path.exists(xmf_filename):
    raise FileNotFoundError(f"File {xmf_filename} not found")

reader = pyablo.XdmfReader()
amr_series = reader.readTimeSeries(xmf_filename)
part_series = reader.readTimeSeries(
    xmf_filename.replace("main", "particles_particles_main")
)

m_gas, m_part = [], []
m_gas_all, m_part_all = [], []
p_gas, p_part = [], []
e_gas, e_part = [], []
M_metal_gas, M_metal_part = [], []
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
    Zg = np.array(snap.readAllFloat("metallicity")) / np.array(snap.getDensity(cells))

    # Gas metallicity should be constant
    np.testing.assert_allclose(Zg, 1, rtol=rtol)

    t.append(snap.getTime())
    with h5py.File(part_fname.replace(".xmf", ".h5")) as part_snap:
        mm = part_snap["mass"][:]
        vv = np.stack(
            [part_snap[f"v{k}"][:] for k in "xyz"],
            axis=-1,
        )
        mp = mm.sum()
        pp = (mm[:, None] * vv).sum(axis=0)
        ep = (0.5 * mm * (vv**2).sum(axis=1)).sum()

        Zp = part_snap["metallicity"][:]
        tp = part_snap["birth_time"][:]

        # Particle metallicity should be constant
        np.testing.assert_allclose(Zp, 1, rtol=rtol)

    m_gas.append(mg)
    m_gas_all.append(vol_g * np.asarray(snap.getDensity(cells)))
    m_part.append(mp)
    m_part_all.append(mm)
    p_gas.append(pg)
    p_part.append(pp)
    e_gas.append(eg)
    e_part.append(ep)
    M_metal_part.append((mm * Zp).sum())
    M_metal_gas.append((vol_g * np.asarray(snap.getDensity(cells)) * Zg).sum())

# Convert to numpy arrays
m_gas = np.asarray(m_gas)
m_part = np.asarray(m_part)
p_gas = np.asarray(p_gas)
p_part = np.asarray(p_part)
e_gas = np.asarray(e_gas)
e_part = np.asarray(e_part)
M_metal_gas = np.asarray(M_metal_gas)
M_metal_part = np.asarray(M_metal_part)
t = np.asarray(t)
m_gas_all = np.asarray(m_gas_all)
m_part_all = np.asarray(m_part_all)

################ CONSERVATION ################
# Mass is conserved
np.testing.assert_allclose(m_gas + m_part, m_gas[0] + m_part[0], rtol=rtol)
# Check stellar mass
np.testing.assert_allclose(m_part_all[:, 0], 1, rtol=rtol)  # First star doesn't explode
np.testing.assert_allclose(m_part_all[:, 1], np.where(t > 0, 0.9, 1), rtol=rtol)  # Second star explode in first step
np.testing.assert_allclose(m_part_all[:, 2], np.where(t > 10, 0.9, 1), rtol=rtol)  # Third star explode after tSNII
for i in range(3):
    # Check velocities: should be conserved
    np.testing.assert_allclose(p_part[:, i] / m_part, p_part[:, i] / m_part, rtol=rtol)
    # Total momentum should be conserved
    np.testing.assert_allclose(p_gas[:, i] + p_part[:, i], p_gas[0, i] + p_part[0, i], rtol=rtol)
    # Total metallicity should be conserved
    np.testing.assert_allclose(M_metal_gas + M_metal_part, M_metal_gas[0] + M_metal_part[0], rtol=rtol)
    print(M_metal_gas, M_metal_part)

import os
import numpy as np

path_BPASS = "/data/anatole/BPASS"
out_file   = os.path.join(path_BPASS, "BPASS_photon_rates.dat")

# Metadata to be reused in ParticleUpdate_radiative_feedback_BPASS.cpp
METAL_NAMES = ["zem5", "zem4", "z001", "z002", "z003", "z004",
               "z006", "z008", "z010", "z014", "z020", "z030", "z040"]
METAL_VALS  = np.array([1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3,
                        6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2])
AGES        = 10.0 ** (6.0 + 0.1 * np.arange(51))                # yr
WVLS_A      = np.arange(int(24.796838), int(2214.0034) + 1)      # Å, 1 Å spacing

BIN_EDGES_EV = np.array([5.6, 11.2, 13.6, 15.2, 24.6, 54.4, 500.0])
HC_EV_A      = 12398.419                                         # h*c [eV * Å]
HC_ERG_A     = 1.9864458e-8                                      # h*c [erg * Å]

def eV_to_A(eV):
    return HC_EV_A / eV

# Bin edges in Å, sorted ascending (high eV = short λ)
bin_lo_A = eV_to_A(BIN_EDGES_EV[1:])    # shorter wavelength side
bin_hi_A = eV_to_A(BIN_EDGES_EV[:-1])   # longer  wavelength side
N_BINS   = len(bin_lo_A)


def integrate_bin(spectrum_erg_s_A, lam_A, lo_A, hi_A):
    """Trapezoid integral of (spectrum * lam / hc) over [lo_A, hi_A]."""
    photon_density = spectrum_erg_s_A * lam_A / HC_ERG_A   # photons / s / Å
    mask = (lam_A >= lo_A) & (lam_A <= hi_A)
    if mask.sum() < 2:
        return 0.0
    return np.trapezoid(photon_density[mask], lam_A[mask])


rates = np.zeros((len(METAL_VALS), len(AGES), N_BINS))

for im, m in enumerate(METAL_NAMES):
    src = f"{path_BPASS}/reduced_spectra-bin-imf_chab300.{m}.5.6to500eV.dat.npy"
    # shape: (n_wvl, n_ages) — see BPASS_reduction.py
    spec = np.load(src)
    for ia in range(len(AGES)):
        sp = spec[:, ia]
        for ib in range(N_BINS):
            rates[im, ia, ib] = integrate_bin(sp, WVLS_A, bin_lo_A[ib], bin_hi_A[ib])

# Write as plain text: metal slow, age fast, 6 columns per row
with open(out_file, "w") as f:
    for im in range(len(METAL_VALS)):
        for ia in range(len(AGES)):
            f.write(" ".join(f"{rates[im, ia, ib]:.8e}" for ib in range(N_BINS)) + "\n")

print(f"Wrote {out_file}  shape={rates.shape}")

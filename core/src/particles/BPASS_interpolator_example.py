import numpy as np
from scipy.interpolate import RegularGridInterpolator

# BPASS (v2.2.1) spectra (interpolated over metallicity and age)

path_BPASS = "/data/anatole/BPASS"

# Grids for BPASS spectra
METAL_NAMES     = ["zem5", "zem4", "z001", "z002", "z003", "z004", "z006", "z008", "z010", "z014", "z020", "z030", "z040"]
METAL_VALS      = np.array([1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2])
AGES            = 10.0 ** (6.0 + 0.1 * np.arange(51))
# Spectra which was already reduced to the relevant wavelength range (5.6 eV to 500 eV) or (24.796838 Å to 2214.0034 Å)
REDUCEDBPASS_WVLS = np.arange(int(24.796838), int(2214.0034) + 1)


all_spec = np.zeros((len(METAL_VALS), len(AGES), len(REDUCEDBPASS_WVLS)))

for i, m in enumerate(METAL_NAMES):
    dat = np.load(f"{path_BPASS}/reduced_spectra-bin-imf_chab300.{m}.5.6to500eV.dat.npy", mmap_mode="r").T
    all_spec[i] = dat

interp = RegularGridInterpolator(
    (METAL_VALS, AGES),
    all_spec,
    bounds_error=False,
    fill_value=None
) # now give interp a star metallicity and age (in years), and it returns the spectrum in erg/s/A (normalized to 1 Msun)


# the metallicity and ages should be clipped to the range of the BPASS spectra
# i.e np.clip(metallicity, METAL_VALS.min(), METAL_VALS.max()) and np.clip(age, AGES.min(), AGES.max())
# then the spectra should be scaled by the mass of the star particle in Msun


# We want to take that spectra and put it into 6 spectral bins: [5.6-11.2 eV], [11.2-13.6 eV], [13.6-15.2 eV], [15.2-24.6 eV], [24.6-54.4 eV], [54.4-500 eV]
# In the end, what we want is the photon/s rate in each of those bins, for every cell containing at least one star (if more than one, we sum the rates).

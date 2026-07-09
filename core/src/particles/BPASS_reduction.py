import numpy as np

# Grab the raw BPASS spectra, narrow them to a specific wavelength range, and convert to erg/s/A.
path_BPASS = "/data/anatole/BPASS"

lsun    = 3.826e33               #[erg/s] solar luminosity
metals  = ["zem5", "zem4", "z001", "z002", "z003", "z004", "z006", "z008", "z010", "z014", "z020", "z030", "z040"]

# Wavelengths in the BPASS spectra are 1 Å spaced, from 1 Å to 100000 Å (100 μm)
wvls = np.arange(1,100001)

def eV_to_A(eV):
    return 12398.419 / eV

# Select relevant wavelength range
lmin = int(eV_to_A(500.0))  # 24.796838 Å
lmax = int(eV_to_A(5.6))    # 2214.0034 Å
filt = (wvls >= lmin) & (wvls <= lmax)

# Loop over metallicities
for m in metals:
    # Load in the data --> units are Lsun/A for a 10^6 Msol cluster
    dat = np.loadtxt(f"{path_BPASS}/spectra-bin-imf_chab300.{m}.dat")

    dat = dat[filt, 1:]  # Remove ages from the first column
    dat /= 1e6          # Divide out the mass
    dat *= lsun         # Convert to erg/s/A

    np.save(f"{path_BPASS}/reduced_spectra-bin-imf_chab300.{m}.5.6to500eV.dat", dat)

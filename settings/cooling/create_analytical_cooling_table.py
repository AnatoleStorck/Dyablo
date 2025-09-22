"""
Create an analytical cooling table with key features of typical cooling functions.
The cooling and heating rates are *not* meant to be physically accurate, but
capture the main qualitative features of realistic cooling functions.

They are to be used for testing the robustness of the cooling implementation
in Dyablo.
"""

import numpy as np
import h5py

# Define grid ranges
log_nH_grid = np.linspace(-6, 2, 30)  # log10(density in cm^-3)
redshift_grid = np.linspace(0, 10, 10)
T_grid = np.logspace(1, 8, 100)  # temperature in K


def cooling_rate(nH, z, T):
    """Mimic a typical cooling function.

    Parameters:
    -----------
    nH : float
        Hydrogen number density in cm^-3
    z : float
        Redshift
    T : float
        Temperature in K

    Returns:
    --------
    Cooling rate in erg cm^3 s^-1
    """
    # Main features: sharp rise at ~1e4 K, peak, dip, then slow increase
    # H line cooling bump
    Lambda_H = 2e-22 * np.exp(-((np.log10(T) - 4.4) ** 2) / 0.08)
    # He line cooling bump
    Lambda_He = 8e-23 * np.exp(-((np.log10(T) - 5.4) ** 2) / 0.14)
    # Free-free cooling (dominates at high T)
    Lambda_ff = 1.5e-27 * T**0.5
    # Combine
    return Lambda_H + Lambda_He + Lambda_ff


def heating_rate(nH, z, T):
    """Mimic a typical heating function.
    Parameters:
    -----------
    nH : float
        Hydrogen number density in cm^-3
    z : float
        Redshift
    T : float
        Temperature in K

    Returns:
    --------
    Heating rate in erg cm^3 s^-1
    """
    H_break = cooling_rate(nH, z, 8e3)
    # Smooth transition: heating drops above 2e4 K
    transition = 1 / (1 + np.exp(-(np.log10(T) - np.log10(2e4)) * 5))
    return H_break * (1 - transition)


def mean_molecular_weight(nH, z, T):
    """Approximate mean molecular weight as a function of temperature.

        Parameters:
    -----------
    nH : float
        Hydrogen number density in cm^-3
    z : float
        Redshift
    T : float
        Temperature in K

    Returns:
    --------
    Mean molecular weight in atomic mass units (excluding metals)
    """
    # Assume X = 0.76 (hydrogen), Y = 0.24 (helium)
    X = 0.76
    Y = 0.24
    # Fully ionized: mu = 1 / (2*X + 0.75*Y)
    mu_ionized = 1 / (2 * X + 0.75 * Y)
    # Fully neutral: mu = 1 / (X + Y/4)
    mu_neutral = 1 / (X + Y / 4)
    # Smooth transition around T ~ 1e4 K
    transition = 1 / (1 + np.exp(-(np.log10(T) - 4) * 10))
    return mu_ionized * transition + mu_neutral * (1 - transition)


def cooling_metals(nH, z, T):
    """Mimic metal cooling contribution.

    Parameters:
    -----------
    nH : float
        Hydrogen number density in cm^-3
    z : float
        Redshift
    T : float
        Temperature in K

    Returns:
    --------
    Cooling rate in erg cm^3 s^-1"""
    return (
        np.exp(-((np.log10(T) - 5.5) ** 2) / 0.25) * 5e-22
        + np.exp(-((np.log10(T) - 5.5) ** 2) / 4) * 1e-24
    )


def heating_metals(nH, z, T):
    """Mimic metal heating contribution.

        Parameters:
    -----------
    nH : float
        Hydrogen number density in cm^-3
    z : float
        Redshift
    T : float
        Temperature in K

    Returns:
    --------
    Heating rate in erg cm^3 s^-1
    """
    return 1 / (1 + np.exp((np.log10(T) - 6) * 3)) * 3e-27


# Allocate arrays
cooling_table = np.zeros((len(log_nH_grid), len(redshift_grid), len(T_grid)))
heating_table = np.zeros_like(cooling_table)
mu_table = np.zeros_like(cooling_table)
cooling_table_metal = np.zeros_like(cooling_table)
heating_table_metal = np.zeros_like(cooling_table)

# Fill tables
for i, log_nH in enumerate(log_nH_grid):
    nH = 10**log_nH
    for j, z in enumerate(redshift_grid):
        for k, T in enumerate(T_grid):
            cooling_table[i, j, k] = cooling_rate(nH, z, T)
            heating_table[i, j, k] = heating_rate(nH, z, T)
            mu_table[i, j, k] = mean_molecular_weight(nH, z, T)
            # Add metal contributions
            cooling_table_metal[i, j, k] += cooling_metals(nH, z, T)
            heating_table_metal[i, j, k] += heating_metals(nH, z, T)

attrs = {
    "Dimension": cooling_table.shape,
    "Parameter1": log_nH_grid,
    "Parameter1_Name": "hden",
    "Parameter2": redshift_grid,
    "Parameter2_Name": "redshift",
    "Temperature": T_grid,
    "Rank": 3,
}
with h5py.File("analytical_cooling_table.h5", "w") as f:
    # Store as CoolingRates/Primordial
    prim = f.create_group("CoolingRates/Primordial")
    prim.create_dataset("Cooling", data=cooling_table)
    prim.create_dataset("Heating", data=heating_table)
    prim.create_dataset("MMW", data=mu_table)

    # Store as CoolingRates/Metals with zero arrays
    metals = f.create_group("CoolingRates/Metals")
    metals.create_dataset("Cooling", data=cooling_table_metal)
    metals.create_dataset("Heating", data=heating_table_metal)

    for dataset in (
        prim["Cooling"],
        prim["Heating"],
        prim["MMW"],
        metals["Cooling"],
        metals["Heating"],
    ):
        dataset.attrs.update(attrs)

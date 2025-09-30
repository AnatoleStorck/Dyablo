import h5py
import numpy as np

for iout in (0, 4):
    for field, multiplier in zip(("rho", "e_tot"), (27, 3)):
        print(f" {iout=} {field=} ".center(80, "="))

        with h5py.File(f"./reference/test_cosmo_iter{iout:07d}.h5", "r") as f:
            ref_field = f[field][:]
            
        with h5py.File(f"./test_cosmo_iter{iout:07d}.h5", "r") as f:
            new_field = f[field][:] * multiplier

        np.testing.assert_allclose(new_field, ref_field, rtol=1e-5)
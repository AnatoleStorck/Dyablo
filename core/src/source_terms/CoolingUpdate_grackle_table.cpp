// #include "CoolingUpdate_base.h"
#include "utils/units/Units.h"
#include "SourceUpdate_base.h"
#include "states/State_hydro.h"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace dyablo {

  namespace {
  /***
   * @brief Find the index `value` in regularly spaced array `arr`
   *
   * Note: assumes arr to be regularly spaced!
   */
  KOKKOS_INLINE_FUNCTION
  size_t find_regular_grid( real_t arr_min, real_t arr_max, real_t arr_spacing, const real_t value, const size_t N ) {
    if (value <= arr_min)   return 0;
    if (value >= arr_max) return N-2;
    size_t i = (size_t)((value - arr_min) / arr_spacing);
    if (i >= N-1) i = N-2;
    return i;
  }

  /***
   * @brief Get the index of the value in the axes
   *
   * Returns the index of the first value i such that
   * axes(i) <= value <= axes(i+1)
   * and the fractional distance between axes(i) and value
   * (i.e. (value - axes(i)) / (axes(i+1) - axes(i)) )
   */
  void get_index( const Kokkos::View<const real_t*, Kokkos::HostSpace>& axes, const real_t value, size_t& i, real_t& f )
  {
    size_t imax = axes.extent(0) - 1;
    // Handle out-of-bounds
    if (value < axes(0)) {
      i = 0;
      f = 0;
    } else if (value > axes(imax)) {
      i = imax - 1;
      f = 1;
    } else {
      int left = 0;
      int right = imax;
      // Binary search, stop when j == i + 1
      while (left < right - 1) {
        size_t mid = (left + right) / 2;
        if (value < axes(mid)) {
          right = mid;
        } else {
          left = mid;
        }
      }
      i = left;

      // Fractional distance
      f = (value - axes(i)) / (axes(i+1) - axes(i));
      f = FMIN(1, FMAX(0, f));
    }
  }

  struct GrackleTable {
    Kokkos::View<real_t***, Kokkos::LayoutRight, Kokkos::HostSpace> data;
    Kokkos::View<real_t*, Kokkos::HostSpace> log_nH;
    Kokkos::View<real_t*, Kokkos::HostSpace> redshift;
    Kokkos::View<real_t*, Kokkos::HostSpace> temperature;
  };


  constexpr real_t kB = Units::KBOLTZ().convert_to(Units::erg() / Units::K());

  /***
   * @brief Evaluate the Lagrange polynomial at a given point
   */
  KOKKOS_INLINE_FUNCTION
  static real_t lagrange_eval( const real_t x,
                               const real_t x0, const real_t f0,
                               const real_t x1, const real_t f1,
                               const real_t x2, const real_t f2 ) {
    real_t x_x1 = x - x1,
            x_x0 = x - x0,
            x_x2 = x - x2,
            x0_x1 = x0 - x1,
            x0_x2 = x0 - x2,
            x1_x2 = x1 - x2;
    real_t L0 = (x_x1 * x_x2) / ( x0_x1 * x0_x2);
    real_t L1 = (x_x0 * x_x2) / (-x0_x1 * x1_x2);
    real_t L2 = (x_x0 * x_x1) / ( x0_x2 * x1_x2);
    return f0 * L0 + f1 * L1 + f2 * L2;
  }

  /***
   * @brief Evaluate the derivative of the Lagrange polynomial at a given point
   */
  KOKKOS_INLINE_FUNCTION
  static real_t lagrange_deriv( const real_t x,
                                const real_t x0, const real_t f0,
                                const real_t x1, const real_t f1,
                                const real_t x2, const real_t f2 ) {
    real_t x_x0 = (x - x0),
            x_x1 = (x - x1),
            x_x2 = (x - x2),
            x0_x1 = (x0 - x1),
            x0_x2 = (x0 - x2),
            x1_x2 = (x1 - x2);
    real_t denom0 = +x0_x1 * x0_x2;
    real_t denom1 = -x0_x1 * x1_x2;
    real_t denom2 =  x0_x2 * x1_x2;
    real_t dL0 = (x_x2 + x_x1) / denom0;
    real_t dL1 = (x_x2 + x_x0) / denom1;
    real_t dL2 = (x_x0 + x_x1) / denom2;
    return f0 * dL0 + f1 * dL1 + f2 * dL2;
  }

  void check_spacing(Kokkos::View<const real_t*> arr, const real_t spacing, const std::string& name) {
    int err = 0;
    Kokkos::parallel_reduce("check_spacing", arr.extent(0) - 2, KOKKOS_LAMBDA(const size_t i, int& err) {
      real_t d0 = spacing;
      real_t di = arr(i+1) - arr(i);

      if (FABS(di - d0) / d0 > 1e-6) err += 1;
    }, Kokkos::Sum<int>(err));

    DYABLO_ASSERT_HOST_RELEASE(err == 0, name << " grid is not regularly spaced.");
  }

  Kokkos::View<real_t*> compute_log10(Kokkos::View<const real_t*> in) {
    Kokkos::View<real_t*> out("log10_output", in.extent(0));
    DYABLO_ASSERT_HOST_RELEASE(in.extent(0) == out.extent(0), "Input and output array must have the same size.");
    Kokkos::parallel_for("compute_log10", in.extent(0), KOKKOS_LAMBDA(const size_t i) {
      out(i) = log10(in(i));
    });
    return out;
  }

  /***
   * @brief 2D table (log_nH, T) with quadratic interpolation on temperature
   */
  struct Table2DQuadT {
    const Kokkos::View<real_t*> log_nH_grid;  // length NH_points
    const Kokkos::View<real_t*> T_grid;       // length NT_points
    const Kokkos::View<real_t**> values;      // shape (NH_points, NT_points)
    const Kokkos::View<real_t*> log_T_grid;   // length NT_points
    const size_t NH_points;
    const size_t NT_points;

    const real_t log_nH_min, log_nH_max, log_nH_spacing;
    const real_t log_T_min, log_T_max, log_T_spacing;

    Table2DQuadT( Kokkos::View<real_t*> log_nH_grid,
                  const real_t log_nH_min, const real_t log_nH_max, const real_t log_nH_spacing,
                  Kokkos::View<real_t*> T_grid,
                  const real_t log_T_min, const real_t log_T_max, const real_t log_T_spacing,
                  Kokkos::View<real_t**> values )
      : log_nH_grid  ( log_nH_grid ),
        T_grid       ( T_grid ),
        values       ( values ),
        log_T_grid   ( compute_log10(T_grid) ),
        NH_points    ( log_nH_grid.extent(0) ),
        NT_points    ( T_grid.extent(0) ),
        log_nH_min   ( log_nH_min ),
        log_nH_max   ( log_nH_max ),
        log_nH_spacing ( log_nH_spacing ),
        log_T_min     ( log_T_min ),
        log_T_max     ( log_T_max ),
        log_T_spacing ( log_T_spacing )
      {
        check_spacing(this->log_nH_grid, log_nH_spacing, "log_nH");
        check_spacing(this->log_T_grid, log_T_spacing, "log_T");
      }

    /***
     * @brief Find the three indices along the T dimension for quadratic interpolation
     */
    KOKKOS_INLINE_FUNCTION
    void find_T_quad( const real_t log_T, size_t &j0, size_t &j1, size_t &j2 ) const {
      if (log_T <= log_T_grid(1)) { j0 = 0; j1 = 1; j2 = 2; return; }
      if (log_T >= log_T_grid(NT_points-2)) { j0 = NT_points-3; j1 = NT_points-2; j2 = NT_points-1; return; }

      size_t j = find_regular_grid(log_T_min, log_T_max, log_T_spacing, log_T, NT_points);

      if (j == 0) j = 1;
      if (j >= NT_points-1) j = NT_points-2;
      j0 = j - 1; j1 = j; j2 = j + 1;
    }

    /***
     * @brief Interpolate the table at given (log_nH, T)
     */
    KOKKOS_INLINE_FUNCTION
    real_t interp( const real_t log_nH, const real_t T, const real_t log_T ) const {
      size_t i0 = find_regular_grid(log_nH_min, log_nH_max, log_nH_spacing, log_nH, NH_points);

      size_t i1 = i0 + 1;
      size_t j0, j1, j2;
      find_T_quad(log_T, j0, j1, j2);

      real_t x0 = T_grid(j0),      x1  = T_grid(j1),     x2  = T_grid(j2);
      real_t f00 = values(i0, j0), f01 = values(i0, j1), f02 = values(i0, j2);
      real_t f10 = values(i1, j0), f11 = values(i1, j1), f12 = values(i1, j2);

      real_t val0 = lagrange_eval(T, x0, f00, x1, f01, x2, f02);
      real_t val1 = lagrange_eval(T, x0, f10, x1, f11, x2, f12);

      real_t log_nH0 = log_nH_grid(i0), log_nH1 = log_nH_grid(i1);
      real_t w = (log_nH - log_nH0) / (log_nH1 - log_nH0 + 1e-300);
      w = FMIN(1, FMAX(0, w));
      return (1 - w) * val0 + w * val1;
    }

    /***
     * @brief Interpolate the derivative dF/dT at given (log_nH, T)
     */
    KOKKOS_INLINE_FUNCTION
    real_t dFdT( real_t log_nH, real_t T, real_t log_T ) const {
      size_t i0 = find_regular_grid(log_nH_min, log_nH_max, log_nH_spacing, log_nH, NH_points);
      size_t i1 = i0 + 1;
      size_t j0, j1, j2;
      find_T_quad(log_T, j0, j1, j2);

      real_t x0 = T_grid(j0),      x1  = T_grid(j1),     x2  = T_grid(j2);
      real_t f00 = values(i0, j0), f01 = values(i0, j1), f02 = values(i0, j2);
      real_t f10 = values(i1, j0), f11 = values(i1, j1), f12 = values(i1, j2);

      real_t d0 = lagrange_deriv(T, x0, f00, x1, f01, x2, f02);
      real_t d1 = lagrange_deriv(T, x0, f10, x1, f11, x2, f12);

      real_t log_nH0 = log_nH_grid(i0), log_nH1 = log_nH_grid(i1);
      real_t w = (log_nH - log_nH0) / (log_nH1 - log_nH0 + 1e-300);
      w = FMIN(1, FMAX(0, w));
      return (1 - w) * d0 + w * d1;
    }
  };


  /***
   * @brief Compute the value of the cooling function at a given (T, nH)
   */
  template<bool include_metals>
  KOKKOS_INLINE_FUNCTION
  real_t compute_f( const real_t nH, const real_t log_nH, const real_t Z_solar, const real_t T, const real_t log_T,
                    const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab,
                    const Table2DQuadT& Hmetals_tab, const Table2DQuadT& Cmetals_tab ) {
    real_t H   = Htab.interp(log_nH, T, log_T);
    real_t C   = Ctab.interp(log_nH, T, log_T);
    real_t mu  = Mutab.interp(log_nH, T, log_T);

    if constexpr (include_metals) {
      H += Hmetals_tab.interp(log_nH, T, log_T) * Z_solar;
      C += Cmetals_tab.interp(log_nH, T, log_T) * Z_solar;
    }

    return 2 * mu * nH / (3 * kB) * (H - C);
  }


  /***
   * @brief Compute the cooling function and its temperature derivative at a given (T, nH)
   */
  template<bool include_metals>
  KOKKOS_INLINE_FUNCTION
  void compute_J( const real_t nH, const real_t log_nH, const real_t Z_solar, const real_t T, const real_t log_T,
                  const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab,
                  const Table2DQuadT& Hmetals_tab, const Table2DQuadT& Cmetals_tab,
                  real_t& f, real_t& J ) {
    real_t H   = Htab.interp(log_nH, T, log_T);
    real_t C   = Ctab.interp(log_nH, T, log_T);
    real_t mu  = Mutab.interp(log_nH, T, log_T);
    real_t dmu = Mutab.dFdT(log_nH, T, log_T);

    if constexpr (include_metals) {
      H += Hmetals_tab.interp(log_nH, T, log_T) * Z_solar;
      C += Cmetals_tab.interp(log_nH, T, log_T) * Z_solar;
    }

    real_t prefac = 2 * mu * nH / (3 * kB);
    real_t S  = prefac * (H - C);
    real_t dH = Htab.dFdT(log_nH, T, log_T);
    real_t dC = Ctab.dFdT(log_nH, T, log_T);

    if constexpr (include_metals) {
      dH += Hmetals_tab.dFdT(log_nH, T, log_T) * Z_solar;
      dC += Cmetals_tab.dFdT(log_nH, T, log_T) * Z_solar;
    }

    real_t St = prefac * (dH - dC) + S * dmu / mu;

    f = S;
    J = St;
  }

  /***
   * @brief Perform a single Rosenbrock3 step
   *
   * Updates T in place and returns an error estimate
   */
  template<bool include_metals>
  KOKKOS_INLINE_FUNCTION
  void rosenbrock3_step(
            const real_t nH, const real_t log_nH,
            const real_t Z_solar,
            const real_t T,
            const real_t h,
            const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab,
            const Table2DQuadT& Hmetals_tab, const Table2DQuadT& Cmetals_tab,
            real_t &T_out, real_t &err_est ) {
    constexpr real_t sqrt2 = 1.4142135623730951;
    constexpr real_t gamma = 1.0 / (2.0 + sqrt2);
    constexpr real_t d31 = - (4 + sqrt2) / (2 + sqrt2);
    constexpr real_t d32 = (6 + sqrt2) / (2 + sqrt2);

    real_t f, J;
    compute_J<include_metals>(nH, log_nH, Z_solar, T, log10(T), Htab, Ctab, Mutab, Hmetals_tab, Cmetals_tab, f, J);

    real_t denom = 1.0 / (1.0 - gamma * h * J);

    real_t k1 = f * denom;

    real_t T1 = T + h * k1 / 2;
    real_t k2 = (compute_f<include_metals>(nH, log_nH, Z_solar, T1, log10(T1), Htab, Ctab, Mutab, Hmetals_tab, Cmetals_tab) - gamma * h * J * k1) * denom;

    real_t T2 = T + h * k2;
    real_t k3 = (compute_f<include_metals>(nH, log_nH, Z_solar, T2, log10(T2), Htab, Ctab, Mutab, Hmetals_tab, Cmetals_tab) - d31 * h * J * k1 - d32 * h * J * k2) * denom;

    T_out = T + h / 6 * (k1 + 4 * k2 + k3);
    err_est = FABS(h / 6 * (k1 - 2 * k2 + k3));
  }

  /***
   * @brief Evolve the temperature from T0 to T_final using adaptive Rosenbrock3
   *
   * Returns the final temperature and the number of steps taken
   */
  template<bool include_metals>
  KOKKOS_INLINE_FUNCTION
  real_t evolve_rosenbrock( const real_t nH, const real_t log_nH, const real_t Z_solar, const real_t T0, const real_t t_final,
                            const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab,
                            const Table2DQuadT& Hmetals_tab, const Table2DQuadT& Cmetals_tab,
                            const real_t dt_init, int &Nsteps, const real_t tol=1e-3 ) {
    real_t t = 0.0;
    real_t T = T0;
    real_t h = dt_init;

    while (t < t_final) {
      if (t + h > t_final) h = t_final - t;

      real_t Tnew, err;

      rosenbrock3_step<include_metals>(
        nH, log_nH, Z_solar, T, h, Htab, Ctab, Mutab, Hmetals_tab, Cmetals_tab, Tnew, err
      );

      real_t scale = fabs(T) + 1e-40;
      real_t rel_err = err / scale;

      if (rel_err < tol) {
        T = Tnew;
        t += h;
        h *= fmin(2.0, 0.9 * pow(tol / (rel_err + 1e-16), 0.5));
      } else {
        h *= fmax(0.1, 0.9 * pow(tol / (rel_err + 1e-16), 0.5));
      }
      Nsteps++;
    }
    return T;
  }

  /***
   * @brief Compute mu from temperature, density, and metallicity
   */
  KOKKOS_INLINE_FUNCTION
  void compute_mu( const real_t log_nH, const real_t T, const real_t Z, const Table2DQuadT& mutab,
                   real_t &mu, real_t &mu_noZ ) {
    mu_noZ = mutab.interp(log_nH, T, log10(T));
    // We assume µ_Z = 16 for solar metallicity
    mu = 1 / (1 / mu_noZ + Z / 16);
  }

  /***
   * @brief Given log_nH and T_over_mu, find T such that T/mu(T, nH) = T_over_mu
   *
   * Uses Newton-Raphson method
   */
  KOKKOS_INLINE_FUNCTION
  real_t find_T( const real_t log_nH, const real_t T_over_mu, const real_t Z, const Table2DQuadT& mutab ){
    int max_iter = 20;
    real_t tol = 1e-6;
    real_t T = T_over_mu; // initial guess
    int iter = 0;
    for (iter = 0; iter < max_iter; ++iter) {
      real_t log_T = log10(T);
      real_t mu, mu_noZ;
      // Note: chain rule applies here
      compute_mu(log_nH, T, Z, mutab, mu, mu_noZ);
      real_t dmu_dT = mutab.dFdT(log_nH, T, log_T) * pow(mu / mu_noZ, 2);

      real_t f = T / mu - T_over_mu;
      real_t df_dT = (mu - T * dmu_dT) / (mu * mu);
      real_t delta = -f / df_dT;
      T += delta;
      if (fabs(delta / T) < tol) break;
    }
    if (iter == max_iter) {
      // If we didn't converge, just return the last value
      // (this should be rare)
      printf("Warning: find_T did not converge after %d iterations, last T = %g K\n", max_iter, T);
    }
    return T;
  }


  enum VarIndex_Cooling {Irho, IE_tot, Irho_vx, Irho_vy, Irho_vz, Imetals};
}


/**
 * @brief Cooling function using grackle-format cooling tables.
 */

class SourceUpdate_cooling_grackle_table : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  real_t gamma0;
  real_t smallr;
  real_t smallc;
  real_t smallp;
  std::string cooling_table;
  real_t Zsolar;

  GrackleTable C, H, mu;
  GrackleTable C_metals, H_metals;

public:
  SourceUpdate_cooling_grackle_table(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),
    gamma0(configMap.getValue<real_t>("hydro", "gamma", 5.0/3.0)),
    cooling_table(configMap.getValue<std::string>("cooling", "grackle_table")),
    Zsolar(configMap.getValue<real_t>("cooling", "Zsolar", 0.014))
  {
    // Open the cooling table
    hid_t m_hdf5_file;
    m_hdf5_file = H5Fopen(cooling_table.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Read cooling tables
    this->C = readTable(m_hdf5_file, "CoolingRates/Primordial/Cooling");
    this->H = readTable(m_hdf5_file, "CoolingRates/Primordial/Heating");
    this->mu = readTable(m_hdf5_file, "CoolingRates/Primordial/MMW");
    this->C_metals = readTable(m_hdf5_file, "CoolingRates/Metals/Cooling");
    this->H_metals = readTable(m_hdf5_file, "CoolingRates/Metals/Heating");

    // Close HDF5 file
    H5Fclose(m_hdf5_file);
    m_hdf5_file = 0;
  }

  ~SourceUpdate_cooling_grackle_table() {}

  /**
   * Read a grackle table from an HDF5 file
   */
  GrackleTable readTable( const hid_t m_hdf5_file, const std::string& name )
  {
    herr_t status;

    hid_t hdf5_type = H5T_NATIVE_DOUBLE;
    hsize_t dims[3];

    hid_t dataset_properties = H5Pcreate(H5P_DATASET_ACCESS);
    hid_t dataset = H5Dopen2(m_hdf5_file, name.c_str(), dataset_properties);
    hid_t filespace = H5Dget_space(dataset);

    H5Sget_simple_extent_dims(filespace, dims, NULL);

    hid_t read_properties = H5Pcreate(H5P_DATASET_XFER);

    // Read "Parameter1" (array of Hydrogen densities)
    hid_t attr_nH = H5Aopen(dataset, "Parameter1", H5P_DEFAULT);
    hsize_t adims_nH[1];
    hid_t atype_nH = H5Aget_type(attr_nH);
    hid_t aspace = H5Aget_space(attr_nH);
    H5Sget_simple_extent_dims(aspace, adims_nH, NULL);
    if (adims_nH[0] != dims[0])
    DYABLO_ASSERT_HOST_RELEASE(
      adims_nH[0] == dims[0], "Error reading cooling table: " + name + " Parameter1 size mismatch"
    )

    // Read "Parameter2" (array of redshifts)
    hid_t attr_redshift = H5Aopen(dataset, "Parameter2", H5P_DEFAULT);
    hsize_t adims_redshift[1];
    hid_t atype_redshift = H5Aget_type(attr_redshift);
    aspace = H5Aget_space(attr_redshift);
    H5Sget_simple_extent_dims(aspace, adims_redshift, NULL);
    DYABLO_ASSERT_HOST_RELEASE(
      adims_redshift[0] == dims[1], "Error reading cooling table: " + name + " Parameter2 size mismatch"
    )

    // Read "Temperature" (array of temperatures)
    hid_t attr_temperature = H5Aopen(dataset, "Temperature", H5P_DEFAULT);
    hsize_t adims_temperature[1];
    hid_t atype_temperature = H5Aget_type(attr_temperature);
    aspace = H5Aget_space(attr_temperature);
    H5Sget_simple_extent_dims(aspace, adims_temperature, NULL);
    DYABLO_ASSERT_HOST_RELEASE(
      adims_temperature[0] == dims[2], "Error reading cooling table: " + name + " Parameter3 size mismatch"
    )

    // Allocate memory
    GrackleTable table;
    // We NEED a LayoutRight here because we read data in C order from HDF5
    table.data = Kokkos::View<real_t***, Kokkos::LayoutRight, Kokkos::HostSpace>(name, dims[0], dims[1], dims[2]);
    table.log_nH = Kokkos::View<real_t*, Kokkos::HostSpace>("log_nH", adims_nH[0]);
    table.redshift = Kokkos::View<real_t*, Kokkos::HostSpace>("redshift", adims_redshift[0]);
    table.temperature = Kokkos::View<real_t*, Kokkos::HostSpace>("temperature", adims_temperature[0]);

    status = H5Dread(dataset, hdf5_type, filespace, filespace, read_properties, table.data.data());
    if (status != 0) throw std::runtime_error("Error reading cooling table (data): " + name);
    status = status, H5Aread(attr_nH, atype_nH, table.log_nH.data());
    if (status != 0) throw std::runtime_error("Error reading cooling table (log_nH): " + name);
    status = H5Aread(attr_redshift, atype_redshift, table.redshift.data());
    if (status != 0) throw std::runtime_error("Error reading cooling table (redshift): " + name);
    status = H5Aread(attr_temperature, atype_temperature, table.temperature.data());
    if (status != 0) throw std::runtime_error("Error reading cooling table (temperature): " + name);

    return table;
  }

  template<int ndim, bool include_metals>
  void update_aux( UserData& U,
                   ScalarSimulationData& scalar_data)
  {
    ForeachCell& foreach_cell = this->foreach_cell;
    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
    timers.get("Cooling simple").start();

    std::vector<dyablo::UserData_fields::FieldAccessor_FieldInfo> fields_info = {
      {"rho_next",    Irho},
      {"e_tot_next",  IE_tot},
      {"rho_vx_next", Irho_vx},
      {"rho_vy_next", Irho_vy},
      {"rho_vz_next", Irho_vz},
    };
    if constexpr (include_metals) {
      fields_info.push_back({"metallicity", Imetals});
    }

    auto Uout = U.getAccessor(fields_info);

    const auto& CTable = this->C;
    const auto& HTable = this->H;
    const auto& muTable = this->mu;
    const auto& CMetals_Table = this->C_metals;
    const auto& HMetals_Table = this->H_metals;

    real_t aexp = scalar_data.get<real_t>("aexp");
    real_t redshift = 1/aexp - 1;

    // -----------------------------------------------------------
    // Interpolate on redshift space
    size_t iz;
    real_t fz;
    get_index(CTable.redshift, redshift, iz, fz);

    // Lambda to combine redshift slabs and push to device
    auto combine_slabs = [=]( const Kokkos::View<real_t***, Kokkos::HostSpace>& table_h, const std::string& name ) -> Kokkos::View<real_t**> {
      // Copy a slab of thickness 2 around iz from host to device
      Kokkos::View<real_t**> slab_d(name + "_slab_device", table_h.extent(0), table_h.extent(2));
      auto slab_h = Kokkos::create_mirror_view(slab_d);

      Kokkos::parallel_for("combine_slabs", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, table_h.extent(0) * table_h.extent(2)),
      KOKKOS_LAMBDA(const size_t idx) {
        size_t i = idx / table_h.extent(2);
        size_t j = idx % table_h.extent(2);
        slab_h(i, j) = (1 - fz) * table_h(i, iz, j) + fz * table_h(i, iz + 1, j);
      });

      // Copy interpolated slab to device
      Kokkos::deep_copy(slab_d, slab_h);
      return slab_d;
    };

    auto Hz = combine_slabs(HTable.data, "heating_rate");
    auto Cz = combine_slabs(CTable.data, "cooling_rate");
    auto muz = combine_slabs(muTable.data, "mu_table");
    auto Hz_metals = combine_slabs(HMetals_Table.data, "metal_heating_rate");
    auto Cz_metals = combine_slabs(CMetals_Table.data, "metal_cooling_rate");

    Kokkos::View<real_t*> log_nH_d("log_nH", CTable.log_nH.extent(0));
    Kokkos::View<real_t*> temperature_d("temperature", CTable.temperature.extent(0));
    Kokkos::deep_copy(log_nH_d, CTable.log_nH);
    Kokkos::deep_copy(temperature_d, CTable.temperature);

    real_t log_nH_min = CTable.log_nH(0), log_nH_max = CTable.log_nH(CTable.log_nH.extent(0)-1);
    real_t log_T_min  = log10(CTable.temperature(0)), log_T_max = log10(CTable.temperature(CTable.temperature.extent(0)-1));

    real_t log_nH_spacing = CTable.log_nH(1) - CTable.log_nH(0);
    real_t log_T_spacing  = log10(CTable.temperature(1)) - log10(CTable.temperature(0));

    DYABLO_ASSERT_HOST_RELEASE(std::isnormal(log_nH_spacing) && std::isnormal(log_T_spacing), "Error: Cooling table have abnormal spacing.");

    Table2DQuadT Htab = Table2DQuadT(log_nH_d, log_nH_min, log_nH_max, log_nH_spacing, temperature_d, log_T_min, log_T_max, log_T_spacing, Hz);
    Table2DQuadT Ctab = Table2DQuadT(log_nH_d, log_nH_min, log_nH_max, log_nH_spacing, temperature_d, log_T_min, log_T_max, log_T_spacing, Cz);
    Table2DQuadT mutab = Table2DQuadT(log_nH_d, log_nH_min, log_nH_max, log_nH_spacing, temperature_d, log_T_min, log_T_max, log_T_spacing, muz);
    Table2DQuadT Hmetals_tab = Table2DQuadT(log_nH_d, log_nH_min, log_nH_max, log_nH_spacing, temperature_d, log_T_min, log_T_max, log_T_spacing, Hz_metals);
    Table2DQuadT Cmetals_tab = Table2DQuadT(log_nH_d, log_nH_min, log_nH_max, log_nH_spacing, temperature_d, log_T_min, log_T_max, log_T_spacing, Cz_metals);

    // ----------------------------------------------------------
    // Cooling timeloop
    Units::Time code_time = Units::code_units().getUnit(Units::s());
    const real_t dt_tot_s = (scalar_data.get<real_t>("dt") * code_time).convert_to(Units::s());
    real_t XH = Units::XH().convert_to(Units::one());

    real_t gamma0 = this->gamma0;

    auto mp_per_cc     = Units::PROTON_MASS() / Units::cm3();
    auto mp_over_kb    = Units::PROTON_MASS() / Units::KBOLTZ();
    auto K = Units::Kelvin();
    auto code_density  = Units::code_units().getUnit<Units::Density>();
    auto code_pressure = Units::code_units().getUnit<Units::Pressure>();

    real_t Zsolar = this->Zsolar;

    foreach_cell.foreach_cell( "Cooling::update", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell) {
        dyablo::ConsHydroState u;
        dyablo::getConservativeState<ndim>(Uout, iCell, u);
        PrimHydroState q = dyablo::consToPrim<ndim>(u, gamma0);

        // Initial state
        auto rho_physical = Units::supercomoving_to_physical<Units::Density>(q.rho, aexp) * code_density;
        auto P_physical = Units::supercomoving_to_physical<Units::Pressure>(q.p, aexp) * code_pressure;
        real_t nH = (rho_physical * XH).convert_to(mp_per_cc);
        real_t log_nH = log10(nH);

        // Compute T/µ
        real_t T_over_mu = (P_physical / rho_physical * mp_over_kb).convert_to(K);

        real_t Z = 0;
        if constexpr (include_metals) {
          Z = Uout.at(iCell, Imetals) / q.rho;
        }

        // Newton-Raphson to find T from T/µ
        real_t T = find_T(log_nH, T_over_mu, Z, mutab);

        // Do cooling timestep
        int Nsteps = 0;
        real_t Tend = evolve_rosenbrock<include_metals>(
          nH, log_nH, Z / Zsolar, T, dt_tot_s,
          Htab, Ctab, mutab, Hmetals_tab, Cmetals_tab,
          dt_tot_s, Nsteps
        );

        // Convert back to pressure
        {
          real_t mu, mu_noZ;
          compute_mu(log_nH, Tend, Z, mutab, mu, mu_noZ);
          T_over_mu = Tend / mu;
        }
        P_physical = T_over_mu * K * rho_physical / mp_over_kb;
        q.p = Units::physical_to_supercomoving<Units::Pressure>(P_physical.convert_to(code_pressure), aexp);

        // Update state
        u = dyablo::primToCons<ndim>(q, gamma0);
        dyablo::setConservativeState<ndim>(Uout, iCell, u);
    });


    timers.get("Cooling simple").stop();
  }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    int ndim = foreach_cell.getDim();

    bool has_metals = U.has_field("metallicity");

    if (ndim == 2)
      throw "2D not implemented";
    else if (ndim == 3 && has_metals)
      update_aux<3, true>(U, scalar_data);
    else if (ndim == 3 && !has_metals)
      update_aux<3, false>(U, scalar_data);
  }

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::SourceUpdate_cooling_grackle_table,
                  "SourceUpdate_cooling_grackle_table");
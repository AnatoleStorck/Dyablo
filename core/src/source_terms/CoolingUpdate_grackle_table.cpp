// #include "CoolingUpdate_base.h"
#include "utils/units/Units.h"
#include "SourceUpdate_base.h"
#include "states/State_hydro.h"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace dyablo {

  namespace {

  /***
   * @brief Binary search for value in arr
   * Returns `i` such that arr(i) <= value < arr(i+1)
   * */
  KOKKOS_INLINE_FUNCTION
  size_t bisect( const Kokkos::View<const real_t*> arr, const real_t value, size_t left, size_t right ) {
    while (left < right - 1) {
      size_t mid = (left + right) / 2;
      if (value < arr(mid)) {
        right = mid;
      } else {
        left = mid;
      }
    }
    return left;
  }

  /***
   * @brief Find the index `value` in regularly spaced array `arr`
   *
   * Note: assumes arr to be regularly spaced!
   */
  KOKKOS_INLINE_FUNCTION
  size_t find_regular_grid( const Kokkos::View<const real_t*> arr, const real_t value, const size_t N ) {
    real_t d = arr(1) - arr(0);
    if (value <= arr(0)) return 0;
    if (value >= arr(N-1)) return N-2;
    size_t i = (size_t)((value - arr(0)) / d);
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
  KOKKOS_INLINE_FUNCTION
  std::tuple<size_t, real_t> get_index(const Kokkos::View<const real_t*>& axes, const real_t value)
  {
    size_t imax = axes.extent(0) - 1;
    size_t i = 0;
    real_t f;
    // Handle out-of-bounds
    if (value < axes(0)) {
      i = 0;
      f = 0;
    } else if (value > axes(imax)) {
      i = imax - 1;
      f = 1;
    } else {
      // Binary search, stop when j == i + 1
      size_t i = bisect(axes, value, 0, imax);

      // Fractional distance
      f = (value - axes(i)) / (axes(i+1) - axes(i));
      f = FMIN(1, FMAX(0, f));
    }
    return {i, f};
  }

  struct GrackleTable {
    Kokkos::View<real_t***, Kokkos::LayoutRight> data;
    Kokkos::View<real_t*> var1;
    Kokkos::View<real_t*> var2;
    Kokkos::View<real_t*> temperature;
  };


  constexpr real_t kB = Units::KBOLTZ().convert_to(Units::erg() / Units::K());

  /***
   * @brief 2D table (log_nH, T) with quadratic interpolation on temperature
   */
  struct Table2DQuadT {
    Kokkos::View<real_t*> log_nH_grid;  // length NH_points
    Kokkos::View<real_t*> log_T_grid;   // length NT_points
    Kokkos::View<real_t*> T_grid;       // length NT_points
    Kokkos::View<real_t**> values;      // shape (NH_points, NT_points)
    size_t NH_points;
    size_t NT_points;

    Table2DQuadT() = default;
    Table2DQuadT(Kokkos::View<real_t*> nH_grid_,
                 Kokkos::View<real_t*> T_grid_,
                 Kokkos::View<real_t**> values_)
      : log_nH_grid(nH_grid_), T_grid(T_grid_), values(values_), NH_points(nH_grid_.extent(0)), NT_points(T_grid_.extent(0)) {
        // precompute log_T_grid
        Kokkos::View<real_t*> log_T_grid("log_T_grid", NT_points);
        auto log_T_grid_host = Kokkos::create_mirror_view(log_T_grid);
        auto T_grid_host = Kokkos::create_mirror_view(this->T_grid);
        Kokkos::deep_copy(T_grid_host, this->T_grid);

        for (size_t j = 0; j < NT_points; ++j) {
          log_T_grid_host(j) = log10(T_grid_host(j));
        }
        Kokkos::deep_copy(log_T_grid, log_T_grid_host);
        this->log_T_grid = log_T_grid;
      }

    /***
     * @brief Find the index of the cell along the nH dimension
     */
    KOKKOS_INLINE_FUNCTION
    size_t find_nH_cell(real_t log_nH) const {
      if (log_nH <= log_nH_grid(0))           return 0;
      if (log_nH >= log_nH_grid(NH_points-1)) return NH_points-2;

      size_t j = find_regular_grid(log_nH_grid, log_nH, NH_points);

      return j;
    }

    /***
     * @brief Find the three indices along the T dimension for quadratic interpolation
     */
    KOKKOS_INLINE_FUNCTION
    void find_T_quad(const real_t log_T, size_t &j0, size_t &j1, size_t &j2) const {
      if (log_T <= log_T_grid(1)) { j0 = 0; j1 = 1; j2 = 2; return; }
      if (log_T >= log_T_grid(NT_points-2)) { j0 = NT_points-3; j1 = NT_points-2; j2 = NT_points-1; return; }

      size_t j = find_regular_grid(log_T_grid, log_T, NT_points);

      if (j == 0) j = 1;
      if (j >= NT_points-1) j = NT_points-2;
      j0 = j - 1; j1 = j; j2 = j + 1;
    }

    /***
     * @brief Evaluate the Lagrange polynomial at a given point
     */
    KOKKOS_INLINE_FUNCTION
    static real_t lagrange_eval(const real_t x,
                                const real_t x0, const real_t f0,
                                const real_t x1, const real_t f1,
                                const real_t x2, const real_t f2) {
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
    static real_t lagrange_deriv(const real_t x,
                                 const real_t x0, const real_t f0,
                                 const real_t x1, const real_t f1,
                                 const real_t x2, const real_t f2) {
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

    /***
     * @brief Interpolate the table at given (log_nH, T)
     */
    KOKKOS_INLINE_FUNCTION
    real_t interp(real_t log_nH, real_t T, real_t log_T) const {
      size_t i0 = find_nH_cell(log_nH);
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
      if (w < 0) w = 0;
      if (w > 1) w = 1;
      return (1 - w) * val0 + w * val1;
    }

    /***
     * @brief Interpolate the derivative dF/dT at given (log_nH, T)
     */
    KOKKOS_INLINE_FUNCTION
    real_t dFdT(real_t log_nH, real_t T, real_t log_T) const {
      size_t i0 = find_nH_cell(log_nH);
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
      if (w < 0) w = 0;
      if (w > 1) w = 1;
      return (1 - w) * d0 + w * d1;
    }
  };

  /***
   * @brief Compute the value of the cooling function at a given (T, nH)
   */
  KOKKOS_INLINE_FUNCTION
  real_t compute_f(const real_t T, const real_t log_T, const real_t nH, const real_t log_nH,
                   const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab) {
    real_t H   = Htab.interp(log_nH, T, log_T);
    real_t C   = Ctab.interp(log_nH, T, log_T);
    real_t mu  = Mutab.interp(log_nH, T, log_T);

    return 2 * mu * nH / (3 * kB) * (H - C);
  }


  /***
   * @brief Compute the cooling function and its temperature derivative at a given (T, nH)
   */
  KOKKOS_INLINE_FUNCTION
  std::tuple<real_t, real_t> compute_J(const real_t T, const real_t log_T, const real_t nH, const real_t log_nH,
                                       const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab) {
    real_t H   = Htab.interp(log_nH, T, log_T);
    real_t C   = Ctab.interp(log_nH, T, log_T);
    real_t mu  = Mutab.interp(log_nH, T, log_T);
    real_t dmu = Mutab.dFdT(log_nH, T, log_T);

    real_t prefac = 2 * mu * nH / (3 * kB);
    real_t S  = prefac * (H - C);
    real_t St = prefac * (Htab.dFdT(log_nH, T, log_T) - Ctab.dFdT(log_nH, T, log_T)) + S * dmu / mu;

    return { S, St };
  }

  /***
   * @brief Perform a single Rosenbrock3 step
   *
   * Updates T in place and returns an error estimate
   */
  KOKKOS_INLINE_FUNCTION
  void rosenbrock3_step(real_t &T, const real_t nH, const real_t log_nH, const real_t h,
                        const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab,
                        real_t &err_est) {
    constexpr real_t sqrt2 = 1.4142135623730951;
    constexpr real_t gamma = 1.0 / (2.0 + sqrt2);
    constexpr real_t d31 = - (4 + sqrt2) / (2 + sqrt2);
    constexpr real_t d32 = (6 + sqrt2) / (2 + sqrt2);

    const real_t log_T = log10(T);

    const auto& [f, J] = compute_J(T, log_T, nH, log_nH, Htab, Ctab, Mutab);

    real_t denom = 1.0 / (1.0 - gamma * h * J);

    real_t k1 = f * denom;

    real_t T1 = T + h * k1 / 2;
    real_t k2 = (compute_f(T1, log10(T1), nH, log_nH, Htab, Ctab, Mutab) - gamma * h * J * k1) * denom;

    real_t T2 = T + h * k2;
    real_t k3 = (compute_f(T2, log10(T2), nH, log_nH, Htab, Ctab, Mutab) - d31 * h * J * k1 - d32 * h * J * k2) * denom;

    T += h / 6 * (k1 + 4 * k2 + k3);
    err_est = FABS(h / 6 * (k1 - 2 * k2 + k3));
  }

  /***
   * @brief Evolve the temperature from T0 to T_final using adaptive Rosenbrock3
   *
   * Returns the final temperature and the number of steps taken
   */
  KOKKOS_INLINE_FUNCTION
  real_t evolve_rosenbrock(const real_t T0, const real_t nH, const real_t log_nH, const real_t t_final,
                           const Table2DQuadT& Htab, const Table2DQuadT& Ctab, const Table2DQuadT& Mutab,
                           const real_t dt_init, int &Nsteps, const real_t tol=1e-3) {
    real_t t = 0.0;
    real_t T = T0;
    real_t h = dt_init;

    while (t < t_final) {
      if (t + h > t_final) h = t_final - t;

      real_t T_save = T;
      real_t err;
      rosenbrock3_step(T, nH, log_nH, h, Htab, Ctab, Mutab, err);

      real_t scale = fabs(T) + 1e-40;
      real_t rel_err = err / scale;

      if (rel_err < tol) {
        t += h;
        h *= fmin(2.0, 0.9 * pow(tol / (rel_err + 1e-16), 0.5));
      } else {
        T = T_save;
        h *= fmax(0.1, 0.9 * pow(tol / (rel_err + 1e-16), 0.5));
      }
      Nsteps++;
    }
    return T;
  }

  /***
   * @brief Given log_nH and T_over_mu, find T such that T/mu(T, nH) = T_over_mu
   *
   * Uses Newton-Raphson method
   */
  KOKKOS_INLINE_FUNCTION
  real_t find_T(const real_t log_nH, const real_t T_over_mu, const Table2DQuadT& mutab){
    int max_iter = 20;
    real_t tol = 1e-6;
    real_t T = T_over_mu; // initial guess
    for (int iter = 0; iter < max_iter; ++iter) {
      real_t log_T = log10(T);
      real_t mu = mutab.interp(log_nH, T, log_T);
      real_t dmu_dT = mutab.dFdT(log_nH, T, log_T);
      real_t f = T / mu - T_over_mu;
      real_t df_dT = (mu - T * dmu_dT) / (mu * mu);
      real_t delta = -f / df_dT;
      T += delta;
      if (fabs(delta / T) < tol) break;
    }
    return T;
  }


  enum VarIndex_Cooling {Irho, IE_tot, Irho_vx, Irho_vy, Irho_vz};
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

  GrackleTable C, H, mu;

public:
  SourceUpdate_cooling_grackle_table(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),
    gamma0(configMap.getValue<real_t>("hydro", "gamma", 5.0/3.0)),
    cooling_table(configMap.getValue<std::string>("cooling", "grackle_table"))
  {
    // Open the cooling table
    hid_t m_hdf5_file;
    m_hdf5_file = H5Fopen(cooling_table.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Read cooling tables
    this->C = readTable(m_hdf5_file, "CoolingRates/Primordial/Cooling");
    this->H = readTable(m_hdf5_file, "CoolingRates/Primordial/Heating");
    this->mu = readTable(m_hdf5_file, "CoolingRates/Primordial/MMW");

    // Close HDF5 file
    H5Fclose(m_hdf5_file);
    m_hdf5_file = 0;
  }

  ~SourceUpdate_cooling_grackle_table() {}

  /**
   * Read a grackle table from an HDF5 file
   */
  GrackleTable readTable(const hid_t m_hdf5_file, const std::string& name)
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
    table.data = Kokkos::View<real_t***, Kokkos::LayoutRight>(name, dims[0], dims[1], dims[2]);
    table.log_nH = Kokkos::View<real_t*>("log_nH", adims_nH[0]);
    table.redshift = Kokkos::View<real_t*>("redshift", adims_redshift[0]);
    table.temperature = Kokkos::View<real_t*>("temperature", adims_temperature[0]);

    #ifdef HDF5_IS_CUDA_AWARE
      status = H5Dread(dataset, hdf5_type, filespace, filespace, read_properties, table.data.data());
      status = std::min(status, H5Aread(attr_nH, atype_nH, table.log_nH.data()));
      status = std::min(status, H5Aread(attr_redshift, atype_redshift, table.redshift.data()));
      status = std::min(status, H5Aread(attr_temperature, atype_temperature, table.temperature.data()));
    #else
      auto data_host = Kokkos::create_mirror_view(table.data);
      status = H5Dread(dataset, hdf5_type, filespace, filespace, read_properties, data_host.data());

      auto nH_host = Kokkos::create_mirror_view(table.log_nH);
      status = std::min(status, H5Aread(attr_nH, atype_nH, nH_host.data()));

      auto redshift_host = Kokkos::create_mirror_view(table.redshift);
      status = std::min(status, H5Aread(attr_redshift, atype_redshift, redshift_host.data()));

      auto Temperature_host = Kokkos::create_mirror_view(table.temperature);
      status = std::min(status, H5Aread(attr_temperature, atype_temperature, Temperature_host.data()));

      Kokkos::deep_copy(table.data, data_host);
      Kokkos::deep_copy(table.log_nH, nH_host);
      Kokkos::deep_copy(table.redshift, redshift_host);
      Kokkos::deep_copy(table.temperature, Temperature_host);
    #endif
    if (status < 0)
      throw std::runtime_error("Error reading cooling table: " + name);

    return table;
  }

  template<int ndim>
  void update_aux( UserData& U,
                   ScalarSimulationData& scalar_data)
  {
    ForeachCell& foreach_cell = this->foreach_cell;
    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
    timers.get("Cooling simple").start();

    auto Uout = U.getAccessor({
          {"rho_next",    Irho},
          {"e_tot_next",  IE_tot},
          {"rho_vx_next", Irho_vx},
          {"rho_vy_next", Irho_vy},
          {"rho_vz_next", Irho_vz},
    });

    const auto& CTable = this->C;
    const auto& HTable = this->H;
    const auto& muTable = this->mu;

    real_t aexp = scalar_data.get<real_t>("aexp");
    real_t redshift = 1/aexp - 1;

    // -----------------------------------------------------------
    // Interpolate on redshift space
    Kokkos::View<real_t**> Cz("cooling_rate_at_z", CTable.data.extent(0), CTable.data.extent(2));
    Kokkos::View<real_t**> Hz("heating_rate_at_z", HTable.data.extent(0), HTable.data.extent(2));
    Kokkos::View<real_t**> muz("mu_at_z", muTable.data.extent(0), muTable.data.extent(2));

    Kokkos::parallel_for("Cooling::interpolate_redshift",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {CTable.data.extent(0), CTable.data.extent(2)}),
      KOKKOS_LAMBDA(const size_t i, const size_t j) {
        const auto& [iz, fz] = get_index(CTable.redshift, redshift);
        Cz(i,j)  = (1 - fz) * CTable.data(i, iz, j) + fz * CTable.data(i, iz+1, j);
        Hz(i,j)  = (1 - fz) * HTable.data(i, iz, j) + fz * HTable.data(i, iz+1, j);
        muz(i,j) = (1 - fz) * muTable.data(i, iz, j) + fz * muTable.data(i, iz+1, j);
    });

    Table2DQuadT Htab(HTable.log_nH, HTable.temperature, Hz);
    Table2DQuadT Ctab(CTable.log_nH, CTable.temperature, Cz);
    Table2DQuadT mutab(muTable.log_nH, muTable.temperature, muz);


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

    int Nsteps_tot = 0;
    int Ncells_tot = 0;

    foreach_cell.reduce_cell( "Cooling::update", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell, int & Nsteps_tot, int & Ncells_tot) {
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

        // Newton-Raphson to find T from T/µ
        real_t T = find_T(log_nH, T_over_mu, mutab);

        // Do cooling timestep
        real_t Tend = evolve_rosenbrock(T, nH, log_nH, dt_tot_s, Htab, Ctab, mutab, dt_tot_s, Nsteps_tot);

        if ((iCell.iOct.iOct == 0) && iCell.i == 0 && iCell.j == 0 && iCell.k == 0)
          printf("T = %e, nH = %e K\n", T, nH);

        // Convert back to pressure
        T_over_mu = Tend / mutab.interp(log_nH, Tend, log10(Tend));
        P_physical = T_over_mu * K * rho_physical / mp_over_kb;
        q.p = Units::physical_to_supercomoving<Units::Pressure>(P_physical.convert_to(code_pressure), aexp);

        // Update state
        u = dyablo::primToCons<ndim>(q, gamma0);
        dyablo::setConservativeState<ndim>(Uout, iCell, u);
        Ncells_tot ++;
    }, Nsteps_tot, Ncells_tot);

    std::cout << "Ncells = " << Ncells_tot << ", Nsteps = " << Nsteps_tot
    << ", Nsteps/Ncells = " << (real_t)Nsteps_tot / Ncells_tot << std::endl;

    timers.get("Cooling simple").stop();
  }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    int ndim = foreach_cell.getDim();

    if (ndim == 2)
      throw "2D not implemented";
    else
      update_aux<3>(U, scalar_data);
  }

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::SourceUpdate_cooling_grackle_table,
                  "SourceUpdate_cooling_grackle_table");
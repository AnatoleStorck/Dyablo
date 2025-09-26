// #include "CoolingUpdate_base.h"
#include "utils/units/Units.h"
#include "SourceUpdate_base.h"
#include "states/State_hydro.h"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace dyablo {
  
  namespace {
  
  /***
   * @brief Get the index of the value in the axes
   *
   * Returns the index of the first value i such that
   * axes(i) <= value <= axes(i+1)
   * and the fractional distance between axes(i) and value
   * (i.e. (value - axes(i)) / (axes(i+1) - axes(i)) )
   */
  KOKKOS_INLINE_FUNCTION
  std::tuple<size_t, real_t> get_index(const Kokkos::View<real_t*>& axes, const real_t value)
  {
    size_t imax = axes.extent(0) - 1;
    size_t i = 0, j = imax;
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
      while (j > i + 1) {
        size_t m = (i + j) / 2;
        if (value < axes(m)) {
          j = m;
        } else {
          i = m;
        }
      }
      // Fractional distance
      f = (value - axes(i)) / (axes(i+1) - axes(i));
      f = FMIN(1, FMAX(0, f));
    }
    return {i, f};
  }
  
  KOKKOS_INLINE_FUNCTION
  real_t interpolate(
    const Kokkos::View<real_t**> table,
    const size_t i,
    const size_t j,
    const real_t fi,
    const real_t fj
  ) {
    return (1 - fi) * (1 - fj) * table(i, j)
         + (1 - fi) * fj * table(i, j+1)
         + fi * (1 - fj) * table(i+1, j)
         + fi * fj * table(i+1, j+1);
  }

  KOKKOS_INLINE_FUNCTION
  std::tuple<size_t, real_t> solve_mu(
    const Kokkos::View<real_t**> mu_table,
    const Kokkos::View<real_t*> T_grid,
    const real_t T_over_mu,
    const size_t inH,
    const real_t fnH
  ) {
    const size_t imax = T_grid.extent(0) - 1;

    // Handle out-of-bounds (clamp to µ_min and µ_max)
    real_t Tmu_low  = T_grid(0)    * mu_table(inH, 0);
    real_t Tmu_high = T_grid(imax) * mu_table(inH, imax);
  
    if (T_over_mu  <= Tmu_low) {
      return {0,      0.};
    } else if (T_over_mu >= Tmu_high) {
      return {imax-1, 1.};
    }
    size_t i = 0;
    for (i = 1; i <= imax; i++) {
      Tmu_high = T_grid(i) * (
        (1 - fnH) * mu_table(inH,   i) +
        fnH       * mu_table(inH+1, i)
      );
      if (Tmu_low < T_over_mu && T_over_mu <= Tmu_high) break;
      Tmu_low = Tmu_high;
    }
    return {i - 1, (T_over_mu - Tmu_low) / (Tmu_high - Tmu_low)};
  }

  struct GrackleTable {
    Kokkos::View<real_t***> data;
    Kokkos::View<real_t*> temperature;
    Kokkos::View<real_t*> log_nH;
    Kokkos::View<real_t*> redshift;
  };
  
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
    throw std::runtime_error("Error reading cooling table: " + name + " Parameter1 size mismatch");
    
    // Read "Parameter2" (array of redshifts)
    hid_t attr_redshift = H5Aopen(dataset, "Parameter2", H5P_DEFAULT);
    hsize_t adims_redshift[1];
    hid_t atype_redshift = H5Aget_type(attr_redshift);
    aspace = H5Aget_space(attr_redshift);
    H5Sget_simple_extent_dims(aspace, adims_redshift, NULL);
    if (adims_redshift[0] != dims[1])
    throw std::runtime_error("Error reading cooling table: " + name + " Parameter2 size mismatch");
    
    // Read "Temperature" (array of temperatures)
    hid_t attr_temperature = H5Aopen(dataset, "Temperature", H5P_DEFAULT);
    hsize_t adims_temperature[1];
    hid_t atype_temperature = H5Aget_type(attr_temperature);
    aspace = H5Aget_space(attr_temperature);
    H5Sget_simple_extent_dims(aspace, adims_temperature, NULL);
    if (adims_temperature[0] != dims[2])
    throw std::runtime_error("Error reading cooling table: " + name + " Parameter2 size mismatch");
    // Allocate memory
    GrackleTable table;
    table.data = Kokkos::View<real_t***>(name, dims[0], dims[1], dims[2]);
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
          {"rho",    Irho}, 
          {"e_tot",  IE_tot},
          {"rho_vx", Irho_vx},
          {"rho_vy", Irho_vy},
          {"rho_vz", Irho_vz},
    });

    const auto& CTable = this->C;
    const auto& HTable = this->H;
    const auto& muTable = this->mu;
    
    real_t redshift = 1/scalar_data.get<real_t>("aexp") - 1;

    // -----------------------------------------------------------
    // Interpolate on redshift space
    Kokkos::View<real_t**> Cz("cooling_rate_at_z", CTable.data.extent(0), CTable.data.extent(1));
    Kokkos::View<real_t**> Hz("heating_rate_at_z", HTable.data.extent(0), HTable.data.extent(1));
    Kokkos::View<real_t**> muz("mu_at_z", muTable.data.extent(0), muTable.data.extent(1));

    Kokkos::parallel_for("Cooling::interpolate_redshift", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {CTable.data.extent(0)-1, CTable.data.extent(1)-1}),
      KOKKOS_LAMBDA(const size_t i, const size_t j) {
        const auto& [iz, fz] = get_index(CTable.redshift, redshift);
        Cz(i,j) = ((1 - fz) * CTable.data(i, iz, j) + fz * CTable.data(i, iz+1, j));
        Hz(i,j) = ((1 - fz) * HTable.data(i, iz, j) + fz * HTable.data(i, iz+1, j));
        muz(i,j) = (1 - fz) * muTable.data(i, iz, j) + fz * muTable.data(i, iz+1, j);
    });

    // ----------------------------------------------------------
    // Cooling timeloop
   
    Units::Time code_time = Units::code_units().getUnit(Units::s());
    const real_t dt_tot_s = (scalar_data.get<real_t>("dt") * code_time).convert_to(Units::s());
    real_t XH = Units::XH().convert_to(Units::one());
    
    real_t gammam1 = gamma0 - 1;

    Units::Density mp_per_cc = Units::PROTON_MASS() / Units::cm3();
    auto mp_over_kb = Units::PROTON_MASS() / Units::KBOLTZ();
    Units::Density code_density = Units::code_units().getUnit(Units::kg()/Units::m3());
    Units::Pressure code_pressure = Units::code_units().getUnit(Units::Pa());
    auto code_P_over_rho = code_pressure / code_density;
    Units::Temperature K = Units::Kelvin();
    Units::Pressure erg_per_cc = Units::erg() / Units::cm3();

    foreach_cell.foreach_cell( "Cooling::update", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout) {
        dyablo::ConsHydroState u;
        dyablo::getConservativeState<ndim>(Uout, iCell_Uout, u);
        
        // Initial state
        real_t nH = (Uout.at(iCell_Uout, Irho) * XH * code_density).convert_to(mp_per_cc);
        real_t log_nH = log10(nH);
        
        // Subcycle cooling timesteps
        real_t t0_s = 0;
        PrimHydroState q = dyablo::consToPrim<ndim>(u, gamma0);
        
        while (t0_s < dt_tot_s) {
          real_t T_over_mu = (q.p / q.rho * code_P_over_rho * gammam1 * mp_over_kb).convert_to(K);
          real_t p_cgs = (q.p * code_pressure).convert_to(erg_per_cc);

          const auto& [inH, fnH] = get_index(CTable.log_nH, log_nH);
          const auto& [iT, fT] = solve_mu(
            muz,
            muTable.temperature,
            T_over_mu,
            inH, fnH
          );

          real_t C = interpolate(Cz, inH, iT, fnH, fT);
          real_t H = interpolate(Hz, inH, iT, fnH, fT);

          // Variation of pressure [erg/s/cm³]
          real_t dp_dt = (H - C) * nH * nH;

          real_t dt_sub_s = FMIN(
            0.1 * FABS(p_cgs / dp_dt), // 10% rule
            dt_tot_s - t0_s            // do not exceed dt_tot_s
          );
          p_cgs += dp_dt * dt_sub_s;
          t0_s  += dt_sub_s;

          q.p = (p_cgs * erg_per_cc).convert_to(code_pressure);
        }
    });

    timers.get("Cooling simple").stop();
  }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    int ndim = foreach_cell.getDim();

    if (ndim == 2)
      update_aux<2>(U, scalar_data);
    else
      update_aux<3>(U, scalar_data);
  }

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::SourceUpdate_cooling_grackle_table,
                  "SourceUpdate_cooling_grackle_table");
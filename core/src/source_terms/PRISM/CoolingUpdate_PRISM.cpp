#include <math.h>
#include "../SourceUpdate_base.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "utils/units/Units.h"

#include "gizmo_rtz/dyablo_api.hpp"


namespace PRISM {
  inline void parseIonInputs(
      const std::vector<std::string>& ions,
      std::array<int, MAX_ELEMENTS>& nions,
      std::array<int, MAX_ELEMENTS>& elems2passive,
      std::array<int, MAX_ELEMENTS>& ions2passive,
      std::map<std::string, int>& ion_counts
  ) {
      for (auto ion : ions) {
          // Trim whitespace
          ion.erase(ion.find_last_not_of(" \n\r\t") + 1);
          ion.erase(0, ion.find_first_not_of(" \n\r\t"));

          std::string element_name;
          if (ion == "H2") {
            // Special case for H2, we don't want to count it as an ion
            element_name = "H";
          } else if (ion == "CO") {
              // Special case for CO, we don't want to count it as an ion
              element_name = "C"; // TODO: Add to O?
          } else {
              element_name = ion.substr(0, ion.find_first_of("_"));
          }

          // Insert if missing
          if (ion_counts.find(element_name) == ion_counts.end()) {
            // If the element is not already in the map, initialize to 0
            ion_counts[element_name] = 0;
          }

          // Increment
          ion_counts[element_name]++;
      }

      // DYABLO_ASSERT_HOST_RELEASE(ion_counts.size() + ions.size() <= n_passive_scalars, "More ions than passive scalars");

      // Create mask of elements
      int iions = ion_counts.size(), ielems = 0;
      auto check_set = [&](
          const std::string& elem_name,
          std::array<int, MAX_ELEMENTS>& nions,
          std::array<int, MAX_ELEMENTS>& elems2passive,
          std::array<int, MAX_ELEMENTS>& ions2passive,
          const int index
      ) {
          if (ion_counts.find(elem_name) == ion_counts.end()) return;

          int nions_this_element = ion_counts.at(elem_name);
          nions[index] = nions_this_element;
          ions2passive[index] = iions;
          elems2passive[index] = ielems;
          iions += nions_this_element;
          ielems++;
      };

      for (auto i = 0; i < MAX_ELEMENTS; ++i) {
          nions[i] = 0;
          ions2passive[i] = -1;
      }

      check_set( "H", nions, elems2passive, ions2passive,  1);
      check_set("He", nions, elems2passive, ions2passive,  2);
      check_set( "C", nions, elems2passive, ions2passive,  6);
      check_set( "N", nions, elems2passive, ions2passive,  7);
      check_set( "O", nions, elems2passive, ions2passive,  8);
      check_set("Ne", nions, elems2passive, ions2passive, 10);
      check_set("Mg", nions, elems2passive, ions2passive, 12);
      check_set("Si", nions, elems2passive, ions2passive, 14);
      check_set( "S", nions, elems2passive, ions2passive, 16);
      check_set("Fe", nions, elems2passive, ions2passive, 26);

  }
}

namespace dyablo {
constexpr bool constant_temperature = false;
constexpr bool ramses_rt_t_scheme = true;
constexpr bool include_H2 = true;
constexpr bool include_CO = true;

using RTZ_type = RTZ<constant_temperature,ramses_rt_t_scheme,include_H2,include_CO>;
/** 
 * @brief PRISM cooling module (https://arxiv.org/abs/2211.04626)
 */
template< typename Policy >
class CoolingUpdate_PRISM : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  
  typename Policy::Params policy_params;

  real_t smallr;
  real_t smallc;
  real_t smallp;
  bool cosmo_run;
  std::string data_path;
  
  // Information about network
  std::vector<std::string> ions;
  std::map<std::string, int> ion_counts;
  std::array<int, MAX_ELEMENTS> nions{};
  std::array<int, MAX_ELEMENTS> elems2passive{};
  std::array<int, MAX_ELEMENTS> ions2passive{};
  int ion_counts_total = 0;

  real_t T_blackbody;
  real_t unit_time;
  real_t unit_density;
  real_t unit_length;
  real_t unit_photon_number;

  RTZ_type rtz_solver;

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;
  
  CoolingUpdate_PRISM(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers
  ) :
        foreach_cell(foreach_cell),
        timers(timers),
        policy_params(Policy::getParams(configMap)),
        cosmo_run(configMap.getValue<bool>("cosmology", "active", false)),
        data_path(configMap.getValue<std::string>("cooling", "data_path")),
        // HM12_UVB_data(PRISM::load_UVB_data(UVB_table_path)),
        ions( configMap.getValue<std::vector<std::string>>("cooling", "ions" ) ),
        T_blackbody( configMap.getValue<real_t>("cooling", "T_blackbody", 1e4) ),
        unit_time( configMap.getValue<real_t>("units", "time", 1.0) ),
        unit_density( configMap.getValue<real_t>("units", "density", 1.0) ),
        unit_length( configMap.getValue<real_t>("units", "length", 1.0) ),
        unit_photon_number( configMap.getValue<real_t>("units", "photon_number", 1.0) ),
        rtz_solver(data_path)
  {
    PRISM::parseIonInputs(ions, this->nions, this->elems2passive, this->ions2passive, this->ion_counts);
  };

  void update( UserData &U,
                ScalarSimulationData& scalar_data)
  {
    const Policy policy( this->policy_params, scalar_data );
    // Get simulation state
    const real_t aexp = cosmo_run ? scalar_data.get<real_t>("aexp") : 1;
    const real_t redshift = 1e0/aexp - 1e0;
    const real_t dt = scalar_data.get<real_t>("dt");

    // Update UV background if needed
    rtz_solver.need_to_update_UVB(redshift);

    // Hydro state accessors
    dyablo::UserData::FieldAccessor Uin = policy.getUout(U);

    // Ion abundances and ionization fractions accessors
    std::vector<UserData::FieldAccessor::FieldInfo> passive_inout;
    for (auto ipassive = 0; ipassive < int(ion_counts.size() + ions.size()); ++ipassive) {
        std::ostringstream oss;
        oss << "passive_scalar_" << ipassive << "_next";
        passive_inout.push_back({oss.str(), ipassive});
    }
    UserData::FieldAccessor Uinout_passive = U.getAccessor( passive_inout );

    // Create units
    real_t XH = Units::XH().convert_to(Units::one());

    auto mp_per_cc     = Units::PROTON_MASS() / Units::cm3();
    auto mp_over_kb    = Units::PROTON_MASS() / Units::KBOLTZ();
    auto K = Units::Kelvin();
    auto code_density  = Units::code_units().getUnit<Units::Density>();
    auto code_pressure = Units::code_units().getUnit<Units::Pressure>();
    auto code_time     = Units::code_units().getUnit<Units::Time>();

    real_t dt_s = (dt * code_time).convert_to(Units::second());


    // ------ Call PRISM cooling update on each cell ------
    foreach_cell.foreach_cell( "CoolingUpdate_PRISM", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell ) {
        // Get local hydro quantities
        ConsState u = policy.getConsState(Uin, iCell);
        PrimState q = policy.consToPrim(u);

        // Initial state
        auto rho_physical = Units::supercomoving_to_physical<Units::Density>(q.rho, aexp) * code_density;
        auto P_physical = Units::supercomoving_to_physical<Units::Pressure>(q.p, aexp) * code_pressure;
        real_t nH = (rho_physical * XH).convert_to(mp_per_cc);
        real_t log_nH = log10(nH);

        // Compute T/µ
        real_t T_over_mu = (P_physical / rho_physical * mp_over_kb).convert_to(K);

        // TODO: get metallicity
        real_t metallicity = 1e0;

        // Get element number densities
        std::array<double, MAX_ELEMENTS> nelements_loc{};
        nelements_loc[1] = nH;
        for (auto i = 2; i < MAX_ELEMENTS; ++i) {
          // TODO: only keep selected elements
          nelements_loc[i] = nH * Uinout_passive.at(iCell, elems2passive[i]);
        }

        // Get ionization fractions
        std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> xions_loc{};
        {
          int iion = 0;
          for (auto i = 1; i < MAX_ELEMENTS; ++i) {
            // TODO: only keep selected elements
            for (auto j = 0; j < i; ++j) {
              xions_loc[iion] = Uinout_passive.at_ivar(iCell, ions2passive[i] + j);
              iion++;
            }
          }
        }

        // TODO: CO
        real_t nCO = 0;
        real_t out_T_over_mu, out_mu, out_ddt;
        int out_its;

        // Physics flags
        PhysicsFlags flags {
           .include_collisional_ionization = true,
           .include_photoionization = true,
           .include_cosmic_ray_ionization = true,
           .include_HM12_UVB = true,
           .include_dust_recombination = true,
           .include_charge_exchange = true,
           .include_self_shielding = false,
        };
  
        rtz_solver.solve_chemistry_and_cooling(
          T_over_mu,
          metallicity,
          aexp,
          dt_s,
          nelements_loc,
          xions_loc,
          nCO,
          out_T_over_mu,
          out_mu,
          out_its,
          out_ddt,
          20'000,
          40'000,
          flags
        );

        // Set element number densities
        nH = nelements_loc[1];
        Uinout_passive.at(iCell, elems2passive[1]) = nH;
        for (auto i = 2; i < MAX_ELEMENTS; ++i) {
          // TODO: only keep selected elements
          Uinout_passive.at(iCell, elems2passive[i]) = nelements_loc[i] / nH;
        }

        // Get ionization fractions
        {
          int iion = 0;
          for (auto i = 1; i < MAX_ELEMENTS; ++i) {
            // TODO: only keep selected elements
            for (auto j = 0; j < i; ++j) {
              Uinout_passive.at_ivar(iCell, ions2passive[i] + j) = xions_loc[iion];
              iion++;
            }
          }
        }

        // Recompute conservative variables
        q.p = (out_T_over_mu * Units::Kelvin() * rho_physical / mp_over_kb).convert_to(code_pressure);
        
        u = policy.primToCons(q);
        policy.setConsState(Uin, iCell, u);
      }
    );
  }
};
} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::CoolingUpdate_PRISM<dyablo::HyperbolicPolicy_Hydro>,
                  "CoolingUpdate_PRISM");

// namespace dyablo {

// /**
//  * @brief Simple density-metallicity dependent cooling function.
//  */

// class CoolingUpdate_PRISM : public SourceUpdate
// {
// private:
//   ForeachCell& foreach_cell;
//   Timers& timers;
//   real_t gamma0;
//   real_t smallr;
//   real_t smallc;
//   real_t smallp;

// public:
//   CoolingUpdate_PRISM(
//         ConfigMap& configMap,
//         ForeachCell& foreach_cell,
//         Timers& timers )
//   : foreach_cell(foreach_cell),
//     timers(timers),
//     gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
//     smallr(configMap.getValue<real_t>("hydro", "smallr", 1e-10)),
//     smallc(configMap.getValue<real_t>("hydro", "smallc", 1e-10)),
//     smallp(smallc * smallc / gamma0),
//     cosmo_run(configMap.getValue<bool>("cosmology", "active", false)),
//     UVB_table_path(configMap.getValue<std::string>("cooling", "UVB_tables")),
//     ct_rates_path(configMap.getValue<std::string>("cooling", "charge_transfer_tables")),
//     high_temperature_metal_cooling_path(configMap.getValue<std::string>("cooling", "high_temperature_metal_cooling_tables")),
//     fine_structure_path(configMap.getValue<std::string>("cooling", "fine_structure_tables")),
//     n_passive_scalars( configMap.getValue<int>("run", "n_passive_scalars", 0) ),
//     HM12_UVB_data(PRISM::load_UVB_data(UVB_table_path)),
//     ions( configMap.getValue<std::vector<std::string>>("cooling", "ions" ) ),
//     T_blackbody( configMap.getValue<real_t>("cooling", "T_blackbody", 1e4) ),
//     unit_time( configMap.getValue<real_t>("units", "time", 1.0) ),
//     unit_density( configMap.getValue<real_t>("units", "density", 1.0) ),
//     unit_length( configMap.getValue<real_t>("units", "length", 1.0) ),
//     unit_photon_number( configMap.getValue<real_t>("units", "photon_number", 1.0) )
//   {

//     PRISM::parseIonInputs(ions, this->nions, this->elems2passive, this->ions2passive, this->ion_counts);

//     // Initialize cosmic ray rates
//     const auto& [cosmic_ray_ionization_rates, cosmic_ray_ionization_rates_induced_UV, cosmic_ray_ionization_rates_induced_UV_heat] = PRISM::initialize_cr_rates();

//     // Initialize the charge transfer rates
//     const auto& [CTRecomb, CTIon] = PRISM::load_ct_rates(ct_rates_path);

//     // Initialize high temperature cooling tables
//     const auto& [high_t_cooling_temp, high_t_cooling_rates, high_t_cooling_rates_tflag] = PRISM::initialize_high_temperature_metal_cooling(high_temperature_metal_cooling_path);

//     // Initialize the low temperature cooling tables
//     const auto& fs_cool_tab = PRISM::init_fine_structure_tables(fine_structure_path);

//     // Get hard-coded rates
//     const auto& dust_rec_coefs = PRISM::get_dust_rec_coefs();
//     const auto& G0_heating_rates = PRISM::get_G0_heating_rates();

//     // HM-related data
//     PRISM::copy_data_1D(G0_heating_rates, tabData.G0_heating_rates);

//     // Initialize rates
//     init_recombination_rates(tabData);
//     init_collisional_ionization(tabData);

//     // Initialize photon groups
//     // TODO: This should be configurable
//     group_E_min = {};
//     group_E_max = {};
//     group_E_min[0] = 13.6; // eV
//     group_E_max[0] = 500.0; // eV

//     // Cosmic ray and dust-related data
//     PRISM::copy_data_2D(cosmic_ray_ionization_rates, tabData.cosmic_ray_ionization_rates);
//     PRISM::copy_data_1D(cosmic_ray_ionization_rates_induced_UV, tabData.cosmic_ray_ionization_rates_induced_UV);
//     PRISM::copy_data_1D(cosmic_ray_ionization_rates_induced_UV_heat, tabData.cosmic_ray_ionization_rates_induced_UV_heat);
//     PRISM::copy_data_2D(dust_rec_coefs, tabData.dust_rec_coefs);

//     // Charge transfer
//     PRISM::copy_data_3D(CTRecomb, tabData.CTRecomb);
//     PRISM::copy_data_3D(CTIon, tabData.CTIon);

//     // Fine structure cooling
//     PRISM::copy_data_3D(fs_cool_tab, tabData.fs_cool_tab);

//     // High-T cooling rates
//     PRISM::copy_data_1D(high_t_cooling_temp, tabData.high_t_cooling_temp);
//     PRISM::copy_data_3D(high_t_cooling_rates, tabData.high_t_cooling_rates);
//     PRISM::copy_data_2D(high_t_cooling_rates_tflag, tabData.high_t_cooling_rates_tflag);

//     // Update cross sections
//     // NB: for a blackbody, this could be done once at initialization.
//     //     But we keep it here for flexibility.
//     timers.get("Cross sections").start();
//     const auto& cross_sections = PRISM::initialize_cross_sections();
//     const auto& cs_ph = PRISM::update_cross_sections(T_blackbody, cross_sections, group_E_min, group_E_max, N_groups);
//     timers.get("Cross sections").stop();

//     // Photo-heating cross sections
//     PRISM::copy_data_4D(cs_ph, tabData.cs_ph);

//   }

//   ~CoolingUpdate_PRISM() {}

//   void update( UserData &U,
//                ScalarSimulationData& scalar_data
//                )
//   {
//     enum VarIndex_rad {
//       Ie_rad, Ifx_rad, Ify_rad, Ifz_rad,       // radiation field
//     };
//     timers.get("Cooling PRISM").start();
//     // ---------------------------------------------------
//     // MODULE CONFIGURATOIN
//     constexpr bool include_H2 = false;               // Whether to include molecular hydrogen
//     constexpr bool include_CO = false;               // Whether to include CO
//     constexpr bool ramses_rt_T_scheme = true;        // Whether to use the ramses-rt temperature update
//     constexpr bool rosenbrock_T_scheme = false;      // Whether to use the rosenbrocat temperature update
//     constexpr bool constant_temperature = false;     // Whether to compute at constant temperature
//     // constexpr real_t metallicity = 0.1;              // Some assumed metallicity

//     // ---------------------------------------------------
//     // UPDATE TIME-DEPENDENT QUANTITIES
//     const real_t aexp = cosmo_run ? scalar_data.get<real_t>("aexp") : 1;
//     const real_t redshift = 1e0/aexp - 1e0;

//     const real_t dt = scalar_data.get<real_t>("dt");

//     // Update UVB
//     const auto& HM12_UVB_z = PRISM::update_UVB(redshift, this->HM12_UVB_data); // Interpolate to the correct redshift
//     PRISM::copy_data_3D(HM12_UVB_z, tabData.HM12_UVB_z);

//     // ---------------------------------------------------
//     // GET ACCESSORS
//     // Hydro fields accessors
//     UserData::FieldAccessor Uinout = U.getAccessor({
//         {"rho_next", ConsHydroState::VarIndex::Irho},
//         {"e_tot_next", ConsHydroState::VarIndex::Ie_tot},
//         {"rho_vx_next", ConsHydroState::VarIndex::Irho_vx},
//         {"rho_vy_next", ConsHydroState::VarIndex::Irho_vy},
//         {"rho_vz_next", ConsHydroState::VarIndex::Irho_vz},
//     });

//     // RT fields accessors
//     const UserData::FieldAccessor Uinout_rad = U.getAccessor({
//         {"e_rad_next", VarIndex_rad::Ie_rad},
//         {"fx_rad_next", VarIndex_rad::Ifx_rad},
//         {"fy_rad_next", VarIndex_rad::Ify_rad},
//         {"fz_rad_next", VarIndex_rad::Ifz_rad}
//     });

//     // Ion abundances and ionization fractions accessors
//     std::vector<UserData::FieldAccessor::FieldInfo> passive_inout;
//     for (auto ipassive = 0; ipassive < int(ion_counts.size() + ions.size()); ++ipassive) {
//         std::ostringstream oss;
//         oss << "passive_scalar_" << ipassive << "_next";
//         passive_inout.push_back({oss.str(), ipassive});
//     }
//     UserData::FieldAccessor Uinout_passive = U.getAccessor( passive_inout );
//     ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
//     uint32_t ndim = foreach_cell.getDim();

//     // ---------------------------------------------------
//     // GET CONSTANTS
//     // const real_t unit_length = this->unit_length; // code -> [m]
//     const real_t unit_time = this->unit_time;     // code -> [s]
//     const real_t unit_density = this->unit_density / (Units::PROTON_MASS * 1e6); 
//                                                   // code -> [kg/m^3] -> [mp/cm^3]
//     const real_t unit_T = SQR(this->unit_length / this->unit_time) * Units::PROTON_MASS / Units::KBOLTZ;
//                                                   // code -> [K]
    
//     const real_t unit_photon_number = this->unit_photon_number;

//     const real_t dt_s = dt * unit_time;
//     const real_t gamma0 = this->gamma0;
//     const auto& nions = this->nions;
//     const auto& elems2passive = this->elems2passive;
//     const auto& ions2passive = this->ions2passive;
//     const auto& tabData = this->tabData;
//     const int N_groups = this->N_groups;

//     // ---------------------------------------------------
//     // THERMOCHEMISTRY STEP
//     int Ncell = 0, Nstep_tot = 0;
//     foreach_cell.reduce_cell( "Cooling::update", Uinout.getShape(),
//       KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell, int& Nstep_tot, int& Ncell) {
//         dyablo::ConsHydroState u;
//         getConservativeState<3>(Uinout, iCell, u);
//         dyablo::PrimHydroState q = consToPrim<3>(u, gamma0);

//         auto cell_size = cells.getCellSize(iCell);

//         // Compute temperature
//         const real_t Tmu = q.p / q.rho * (gamma0 - 1) * unit_T;

//         // Extract ion data
//         PRISM::Element elements_loc[MAX_ELEMENTS];
//         PRISM::ParticleIonData n_and_ion_fracs_loc[MAX_ELEMENTS];
//         // Reset values
//         for (auto i = 0; i < MAX_ELEMENTS; ++i) {
//           n_and_ion_fracs_loc[i].n_element = 0.0;
//           for (auto j = 0; j < MAX_ELEMENTS; ++j) {
//             n_and_ion_fracs_loc[i].ion_fracs[j] = 0.0;
//             n_and_ion_fracs_loc[i].ion_fracs_new[j] = 0.0;
//           }
//         }

//         // Initialize the elements
//         PRISM::initialize_elements<include_H2, include_CO>(elements_loc, nions);

//         // Get nH
//         real_t nH = q.rho * unit_density * (1.0 - Units::YHE) / elements_loc[1].atomic_mass;
//         n_and_ion_fracs_loc[1].n_element = nH;

//         // Get other species abundances
//         for (auto i = 2; i < MAX_ELEMENTS; ++i) {
//           if (elements_loc[i].atomic_number < 0) continue;
//           n_and_ion_fracs_loc[i].n_element = nH * Uinout_passive.at(iCell, elems2passive[i]);
//         }

//         // Get ion fractions (incl. H)
//         for (auto i = 1; i < MAX_ELEMENTS; ++i) {
//           if (elements_loc[i].atomic_number < 0) continue;
//           real_t xtot = 0.0;
//           for (auto j = 0; j < elements_loc[i].n_ions + elements_loc[i].n_mol; ++j) {
//               n_and_ion_fracs_loc[i].ion_fracs[j] = Uinout_passive.at_ivar(iCell, ions2passive[i] + j);
//               xtot += n_and_ion_fracs_loc[i].ion_fracs[j];
//           }
//           // Normalize ion fractions
//           for (auto j = 0; j < elements_loc[i].n_ions + elements_loc[i].n_mol; ++j) {
//               n_and_ion_fracs_loc[i].ion_fracs[j] /= xtot;
//           }
//         }

//         constexpr double UV_background_G0 = 1E-6;
//         constexpr double generic_cosmic_ray_ionization_rate = 1E-25;
//         constexpr double dust_to_gas_mass_ratio_over_mw = 1;
//         const real_t vol = cell_size[IX] * cell_size[IY] * (ndim == 3 ? cell_size[IZ] : 1);
//         real_t N_phot[MAX_N_GROUPS] = {0};
//         real_t F_phot[MAX_N_GROUPS][3] = {};
//         for (auto igrp = 0; igrp < N_groups; ++igrp) {
//           N_phot[igrp] = Uinout_rad.at(iCell, VarIndex_rad::Ie_rad) * unit_photon_number;
//           F_phot[igrp][0] = Uinout_rad.at(iCell, VarIndex_rad::Ifx_rad) * unit_photon_number;
//           F_phot[igrp][1] = Uinout_rad.at(iCell, VarIndex_rad::Ify_rad) * unit_photon_number;
//           if (ndim == 3) F_phot[igrp][2] = Uinout_rad.at(iCell, VarIndex_rad::Ifz_rad) * unit_photon_number; 
//         }

//         const auto& [total_iterations, Tout] = PRISM::subcycle_chemistry<
//           constant_temperature, ramses_rt_T_scheme, rosenbrock_T_scheme, include_H2, include_CO,
//           true
//         >(elements_loc, n_and_ion_fracs_loc, Tmu, aexp, dt_s,
//           UV_background_G0, generic_cosmic_ray_ionization_rate, dust_to_gas_mass_ratio_over_mw,
//           N_phot, F_phot, N_groups,
//           20000, 100000,
//           tabData);

//         // Store new temperature
//         q.p = q.rho * Tout / (gamma0 - 1) / unit_T;
//         u = primToCons<3>(q, gamma0);
//         Uinout.at(iCell, dyablo::ConsHydroState::Ie_tot) = u.e_tot;

//         // Store new ionization fractions
//         for (auto i = 1; i < MAX_ELEMENTS; ++i) {
//           if (elements_loc[i].atomic_number < 0) continue;
//           for (auto j = 0; j < elements_loc[i].n_ions + elements_loc[i].n_mol; ++j) {
//               Uinout_passive.at(iCell, ions2passive[i] + j) = n_and_ion_fracs_loc[i].ion_fracs_new[j];
//           }
//         }

//         // Store new fluxes and photon numbers
//         for (auto igrp = 0; igrp < N_groups; ++igrp) {
//           Uinout_rad.at(iCell, VarIndex_rad::Ie_rad) = N_phot[igrp] / unit_photon_number;
//           Uinout_rad.at(iCell, VarIndex_rad::Ifx_rad) = F_phot[igrp][0] / unit_photon_number;
//           Uinout_rad.at(iCell, VarIndex_rad::Ify_rad) = F_phot[igrp][1] / unit_photon_number;
//           if (ndim == 3) Uinout_rad.at(iCell, VarIndex_rad::Ifz_rad) = F_phot[igrp][2] / unit_photon_number;
//         }

//         // printf("T = %e, rho = %e, N = %e, xHI = %e, xHII = %e, xHeI = %e, xHeII = %e, xHeIII = %e, iterations = %d\n",
//         //        Tout, q.rho, N_phot_old[0],
//         //        Uinout_passive.at(iCell, ions2passive[1]), Uinout_passive.at(iCell, ions2passive[1] + 1),
//         //        Uinout_passive.at(iCell, ions2passive[2]), Uinout_passive.at(iCell, ions2passive[2] + 1), Uinout_passive.at(iCell, ions2passive[2] + 2),
//         //        total_iterations);

//         Nstep_tot += total_iterations;
//         Ncell++;

//     }, Nstep_tot, Ncell);

//     std::cout << "Converged in " << real_t(Nstep_tot) / Ncell << " iterations on average." << std::endl;

//     timers.get("Cooling PRISM").stop();

//   }

//   PRISM::TabulatedData tabData;
//   bool cosmo_run;

//   std::string UVB_table_path;
//   std::string ct_rates_path;
//   std::string high_temperature_metal_cooling_path;
//   std::string fine_structure_path;

//   int n_passive_scalars;

//   // UV background data
//   PRISM::UVB_table_t HM12_UVB;
//   PRISM::UVB_data HM12_UVB_data;

//   // Information about network
//   std::vector<std::string> ions;
//   std::map<std::string, int> ion_counts;
//   std::array<int, MAX_ELEMENTS> nions;
//   std::array<int, MAX_ELEMENTS> elems2passive;
//   std::array<int, MAX_ELEMENTS> ions2passive;

//   // Photon groups
//   std::array<double, MAX_N_GROUPS> group_E_min;
//   std::array<double, MAX_N_GROUPS> group_E_max;
//   int N_groups = 1; // Number of photon groups, can be set in the constructor

//   // Blackbody temperature
//   real_t T_blackbody;

//   // Units
//   real_t unit_time;
//   real_t unit_density;
//   real_t unit_length;
//   real_t unit_photon_number;

// };

// } // namespace dyablo

// FACTORY_REGISTER( dyablo::SourceUpdateFactory,
//                   dyablo::CoolingUpdate_PRISM,
//                   "CoolingUpdate_PRISM");
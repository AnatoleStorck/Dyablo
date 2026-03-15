#include <math.h>
#include "../SourceUpdate_base.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "utils/units/Units.h"

#include "gizmo_rtz/dyablo_api.hpp"
// Include the implementation so that CUDA device functions (KOKKOS_FUNCTION)
// are visible within this translation unit, avoiding the need for
// relocatable device code (-rdc=true).
#include "gizmo_rtz/dyablo_api.cpp"


namespace PRISM {
  std::string int2roman(int num) {
      std::vector<std::pair<int, std::string>> value_symbols = {
          {1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"},
          {100, "C"}, {90, "XC"}, {50, "L"}, {40, "XL"},
          {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"},
          {1, "I"}
      };

      std::string roman;
      for (const auto& [value, symbol] : value_symbols) {
          while (num >= value) {
              roman += symbol;
              num -= value;
          }
      }
      return roman;
  }

  inline void parseIonInputs(
      const std::vector<std::string>& ions,
      std::array<int, MAX_ELEMENTS>& nions_and_molecules,
      std::array<int, MAX_ELEMENTS>& elems2passive,
      std::array<int, MAX_ELEMENTS>& ions2passive,
      std::map<std::string, int>& elem2atomicnum,
      std::map<std::string, int>& ion_counts,
      std::map<std::string, int>& molecule_counts,
      bool include_H2
  ) {
      for (auto ion : ions) {
          // Trim whitespace
          ion.erase(ion.find_last_not_of(" \n\r\t") + 1);
          ion.erase(0, ion.find_first_not_of(" \n\r\t"));

          std::string element_name;
          if (ion == "H2") {
            // Special case for H2, we don't want to count it as an ion
            element_name = "H";
            molecule_counts["H"] = 1;
          } else if (ion == "CO") {
              // Special case for CO, we don't want to count it as an ion
              element_name = "C"; // TODO: Add to O?
              // FIXME / TODO
          } else {
              element_name = ion.substr(0, ion.find_first_of("_"));
            // Insert if missing
            if (ion_counts.find(element_name) == ion_counts.end()) {
              // If the element is not already in the map, initialize to 0
              ion_counts[element_name] = 0;
            }

            // Increment
            ion_counts[element_name]++;
          }
      }

      // If running with H2, make sure it is included
      if (include_H2) {
        // Find index of "H2" in ions list
        int H2_index = -1;
        for (size_t i = 0; i < ions.size(); i++) {
            if (ions[i] == "H2") {
                H2_index = i;
                break;
            }
        }
        DYABLO_ASSERT_HOST_RELEASE(H2_index != -1, "H2 is included but not found in ions list");
      }

      // DYABLO_ASSERT_HOST_RELEASE(ion_counts.size() + ions.size() <= n_passive_scalars, "More ions than passive scalars");

      // Create mask of elements
      int iions = ion_counts.size(), ielems = 0;
      auto check_set = [&](
          const std::string& elem_name,
          const int atomic_number
      ) {
          if (ion_counts.find(elem_name) == ion_counts.end()) return;

          int nions_this_element = ion_counts.at(elem_name);
          int nmolecules_this_element = molecule_counts.find(elem_name) != molecule_counts.end() ? molecule_counts.at(elem_name) : 0;

          nions_and_molecules[atomic_number] = nions_this_element + nmolecules_this_element;
          ions2passive[atomic_number] = iions;
          elems2passive[atomic_number] = ielems;
          elem2atomicnum[elem_name] = atomic_number;
          iions += nions_this_element + nmolecules_this_element;
          ielems++;
      };

      for (auto i = 0; i < MAX_ELEMENTS; ++i) {
          nions_and_molecules[i] = 0;
          ions2passive[i] = -1;
      }

      check_set( "H",  1);
      check_set("He",  2);
      check_set( "C",  6);
      check_set( "N",  7);
      check_set( "O",  8);
      check_set("Ne", 10);
      check_set("Mg", 12);
      check_set("Si", 14);
      check_set( "S", 16);
      check_set("Fe", 26);
  }
}

namespace dyablo {
constexpr bool constant_temperature = true;
constexpr bool include_H2 = true;
constexpr bool include_CO = false;
constexpr bool rt_advect = true;
constexpr bool include_self_shielding = false;

using RTZ_type = RTZ<constant_temperature,include_H2,include_CO,rt_advect,include_self_shielding>;
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
  std::map<std::string, int> molecule_counts;
  std::array<int, MAX_ELEMENTS> nions_and_molecules{};
  std::array<int, MAX_ELEMENTS> elems2passive{};
  std::array<int, MAX_ELEMENTS> ions2passive{};
  std::map<std::string, int> elem2atomicnum{};
  int ion_counts_total = 0;

  real_t T_blackbody;

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
        T_blackbody( configMap.getValue<real_t>("cooling", "T_blackbody", 4e4) ),
        rtz_solver(data_path)
  {
    PRISM::parseIonInputs(ions, this->nions_and_molecules, this->elems2passive, this->ions2passive, this->elem2atomicnum, this->ion_counts, this->molecule_counts, include_H2);
    rtz_solver.set_photon_groups({13.6}, {500});
    for (int i = 0; i < MAX_ELEMENTS; ++i)
      ion_counts_total += nions_and_molecules[i];

    timers.get("CoolingUpdate_PRISM:cross_section").start();
    rtz_solver.need_to_update_cross_sections(T_blackbody);
    timers.get("CoolingUpdate_PRISM:cross_section").stop();

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
    std::vector<UserData::FieldAccessor::FieldInfo> passive_in;
    std::vector<UserData::FieldAccessor::FieldInfo> passive_out;
    {
      // Add in element abundances
      for (const auto& [elem, _val]: ion_counts) {
        int atomic_num = elem2atomicnum[elem];
        std::ostringstream oss;
        oss << "n" << elem;
        passive_in.push_back({oss.str(), elems2passive[atomic_num]});
        // oss << "_next";
        passive_out.push_back({oss.str(), elems2passive[atomic_num]});
      }
      // Add in xions
      for (const auto& [elem, _val]: ion_counts) {
        int atomic_num = elem2atomicnum[elem];
        for (int iion = 0; iion < ion_counts[elem]; iion++) {
          std::ostringstream oss;
          // Convert iion to roman numeral
          std::string iion_roman = PRISM::int2roman(iion+1);
          oss << "x" << elem << "_" << iion_roman;
          passive_in.push_back({oss.str(), ions2passive[atomic_num] + iion});
          // oss << "_next";
          passive_out.push_back({oss.str(), ions2passive[atomic_num] + iion});
        }
        if (elem == "H" && include_H2) {
          // Add H2
          std::string field_name = "xH2";
          passive_in.push_back({field_name.c_str(), ions2passive[atomic_num] + 2});
          passive_out.push_back({field_name.c_str(), ions2passive[atomic_num] + 2});
        }
      }

    }

    UserData::FieldAccessor Uin_passive = U.getAccessor( passive_in );
    UserData::FieldAccessor Uout_passive = U.getAccessor( passive_out );


    // RT accessors    std::vector<UserData::FieldAccessor::FieldInfo> rt_in;
    DYABLO_ASSERT_HOST_RELEASE(N_GROUPS == 1, "Only N_GROUPS=1 is currently supported");
    DYABLO_ASSERT_HOST_RELEASE(foreach_cell.getDim() == 3, "Only 3D is currently supported");

    UserData::FieldAccessor Uin_rt = U.getAccessor( {{"e_rad_next", 0}, {"fx_rad_next", 1}, {"fy_rad_next", 2}, {"fz_rad_next", 3}} );
    UserData::FieldAccessor Uout_rt = U.getAccessor( {{"e_rad_next", 0}, {"fx_rad_next", 1}, {"fy_rad_next", 2}, {"fz_rad_next", 3}} );

    // Create units
    real_t XH = Units::XH().convert_to(Units::one());

    auto mp_per_cc     = Units::PROTON_MASS() / Units::cm3();
    auto mp_over_kb    = Units::PROTON_MASS() / Units::KBOLTZ();
    auto K = Units::Kelvin();
    auto code_density  = Units::code_units().getUnit<Units::Density>();
    auto code_pressure = Units::code_units().getUnit<Units::Pressure>();
    auto code_time     = Units::code_units().getUnit<Units::Time>();

    real_t dt_s = (dt * code_time).convert_to(Units::second());

    const std::array<int, MAX_ELEMENTS> &nions_and_molecules = this->nions_and_molecules;
    const std::array<int, MAX_ELEMENTS> &elems2passive = this->elems2passive;
    const std::array<int, MAX_ELEMENTS> &ions2passive = this->ions2passive;
    const RTZ_type& rtz_solver = this->rtz_solver;

    timers.get("CoolingUpdate_PRISM").start();

    // Important note here:
    // GPUs have a limited amount of shared memory (typically on the order of
    // 1kB per thread). CompactIonData takes 2352 bytes, so we cannot store one
    // per thread on the stack. Instead, we use a shared pool of CompactIonData
    // objects, and each thread acquires one when needed. The UniqueToken
    // mechanism, by default, has a .size() equal to the hardware concurrency
    // (the number of simultaneous active threads), and allows generating unique
    // indices in the range [0, .size()) that can be used to index into shared
    // memory without conflicts.
    using exec_space = Kokkos::DefaultExecutionSpace;
    Kokkos::Experimental::UniqueToken<exec_space> token;
    Kokkos::View<CompactIonData*> compact_data("PRISM_compact_data", token.size());

    // ------ Call PRISM cooling update on each cell ------
    foreach_cell.foreach_cell( "CoolingUpdate_PRISM", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell ) {
        Kokkos::Experimental::AcquireUniqueToken<exec_space> slot(token);
        CompactIonData& n_and_ion_fracs_loc = compact_data(slot.value());

        // Get local hydro quantities
        ConsState u = policy.getConsState(Uin, iCell);
        PrimState q = policy.consToPrim(u);

        // Initial state
        auto rho_physical = Units::supercomoving_to_physical<Units::Density>(q.rho, aexp) * code_density;
        auto P_physical = Units::supercomoving_to_physical<Units::Pressure>(q.p, aexp) * code_pressure;
        real_t rho = rho_physical.convert_to(mp_per_cc);

        // Compute T/µ
        real_t T_over_mu = (P_physical / rho_physical * mp_over_kb).convert_to(K);

        // Initialize CompactIonData from field data
        const Element* elements = rtz_solver.elements_d.data();
        n_and_ion_fracs_loc = CompactIonData{};
        n_and_ion_fracs_loc.init_offsets(elements);

        for (int i = 1; i < MAX_ELEMENTS; ++i) {
          if (ions2passive[i] == -1) continue; // Skip elements not in network
          // Set element number density
          n_and_ion_fracs_loc.n_element[i] = Uin_passive.at(iCell, elems2passive[i]);
          // Set ion fractions
          for (int j = 0; j < nions_and_molecules[i]; ++j) {
            int index = ions2passive[i] + j;
            n_and_ion_fracs_loc[i].ion_fracs[j] = Uin_passive.at(iCell, index);
          }
        }

        // Get Photon Stuff
        std::array<double, N_GROUPS> N_PHOT{};
        std::array<std::array<double, 3>, N_GROUPS> F_PHOT{};

        for (auto i = 0; i < N_GROUPS; ++i) {
          int index = 4 * i; // TODO: don't hardcode this
          N_PHOT[i] = Uin_rt.at(iCell, index);
          for (auto j = 0; j < 3; ++j) {
            F_PHOT[i][j] = Uin_rt.at(iCell, index + j + 1);
          }
        }

        // TODO: CO
        real_t nCO = 0;
        real_t out_T_over_mu, out_mu;

        // Physics flags
        PhysicsFlags flags {
            .include_collisional_ionization = true,
            .include_photoionization        = true,
            .include_cosmic_ray_ionization  = true,
            .include_HM12_UVB               = true,
            .include_dust_recombination     = true,
            .include_charge_exchange        = true,
        };

        // TODO: get metallicity
        real_t metallicity = 1e-40;

        T_over_mu = 1e4;

        rtz_solver.solve_chemistry_and_cooling(
          T_over_mu,
          metallicity,
          aexp,
          dt_s,
          n_and_ion_fracs_loc,
          nCO,
          N_PHOT,
          F_PHOT,
          out_T_over_mu,
          out_mu,
          20'000,
          40'000,
          flags
        );

        // Write back element number densities and ion fractions
        for (int i = 1; i < MAX_ELEMENTS; ++i) {
          if (ions2passive[i] == -1) continue; // Skip elements not in network
          Uout_passive.at(iCell, elems2passive[i]) = n_and_ion_fracs_loc.n_element[i];
          for (int j = 0; j < nions_and_molecules[i]; ++j) {
            int index = ions2passive[i] + j;
            Uout_passive.at(iCell, index) = n_and_ion_fracs_loc[i].ion_fracs[j];
          }
        }

        for (auto i = 0; i < N_GROUPS; ++i) {
          int index = 4 * i; // TODO: don't hardcode this
          Uout_rt.at(iCell, index) = N_PHOT[i];
          for (auto j = 0; j < 3; ++j) {
            Uout_rt.at(iCell, index + j + 1) = F_PHOT[i][j];
          }
        }

        // Recompute conservative variables
        q.p = (out_T_over_mu * Units::Kelvin() * rho_physical / mp_over_kb).convert_to(code_pressure);

        u = policy.primToCons(q);
        policy.setConsState(Uin, iCell, u);
      }
    );

    timers.get("CoolingUpdate_PRISM").stop();
  }
};
} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::CoolingUpdate_PRISM<dyablo::HyperbolicPolicy_Hydro>,
                  "CoolingUpdate_PRISM_hydro");

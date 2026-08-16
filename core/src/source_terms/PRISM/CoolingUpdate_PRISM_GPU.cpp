
#include <math.h>
#include <iostream>
#include "../SourceUpdate_base.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "utils/units/Units.h"

#include "gizmo_rtz/dyablo_api_GPU.cpp"

using namespace prism_gpu;


namespace PRISM_GPU {
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

  inline void parsePhotonGroupInputs(
      int n_groups,
      const std::vector<real_t>& rt_groups_lower,
      const std::vector<real_t>& rt_groups_upper,
      std::vector<double>& E_min,
      std::vector<double>& E_max
  ) {
      DYABLO_ASSERT_HOST_RELEASE(n_groups > 0, "n_groups must be > 0");
      DYABLO_ASSERT_HOST_RELEASE(n_groups == N_GROUPS, "n_groups must be equal to N_GROUPS");
      DYABLO_ASSERT_HOST_RELEASE(rt_groups_lower.size() == static_cast<size_t>(n_groups), "rt_groups_lower size must be equal to n_groups");
      DYABLO_ASSERT_HOST_RELEASE(rt_groups_upper.size() == static_cast<size_t>(n_groups), "rt_groups_upper size must be equal to n_groups");

      E_min.assign(n_groups, 0.0);
      E_max.assign(n_groups, 0.0);

      for (int i = 0; i < n_groups; i++) {
        E_min[i] = rt_groups_lower[i];
        E_max[i] = rt_groups_upper[i];
          DYABLO_ASSERT_HOST_RELEASE(E_min[i] < E_max[i], "For photon group " << i << ", E_min must be less than E_max");
      }
  }
}

namespace dyablo {
constexpr bool constant_temperature = false;
constexpr bool include_H2 = true;
constexpr bool include_CO = true;
constexpr bool rt_advect = true;
constexpr bool include_self_shielding = true;

using RTZ_type = RTZ<constant_temperature,include_H2,include_CO,rt_advect,include_self_shielding>;

// Per-cell solver state persisted in global memory between the split
// step-kernels of the subcycle driver (one slot per cell of the current
// batch; the larger CompactIonData part lives in its own pool).
struct PrismCellState {
  SubcycleState sub;
  std::array<double, N_GROUPS> N_PHOT;     // committed photon densities
  std::array<double, N_GROUPS> N_PHOT0;    // photon densities at hydro-step start
                                           // (reference scale for rad_residual_floor)
  Array2D_RT F_PHOT;                       // committed photon fluxes
  std::array<double, N_GROUPS> N_phot_new; // radiation step -> commit
  std::array<double, N_GROUPS> phot_att;   // radiation step -> commit (flux attenuation)
  std::array<double, N_GROUPS> dNpdt;      // rt_smooth: operator-split RT increments,
  Array2D_RT dFpdt;                        // injected during the subcycle (zero when off)
  double nCO;        // committed CO number density
  double loc_rho;    // pre-depletion mass density [amu/cm^3]
  double dx_cm;      // cell size [cm]
  double dust_ratio; // dust-to-gas mass ratio over the MW value
  double ss_factor;  // self-shielding factor
  double cool_rate_pair[2]; // slots for the two all_cooling evaluations
                            // (cooling_rate_lane in the main loop; the
                            // team-shared cooling_step_team in the tail)
};

/**
 * @brief PRISM cooling module (https://arxiv.org/abs/2211.04626)
 *
 * Split-kernel variant: the thermochemistry subcycle loop runs on the host,
 * and each physics step (prep+radiation, cooling, molecules, ions, commit) is
 * launched as its own kernel over the still-active cells, so each step gets
 * compiled and optimized independently instead of as one monolithic kernel.
 * Each cell keeps its own adaptive subcycle timestep; cells that have
 * integrated to the full hydro dt are compacted out of the active list.
 */
template< typename Policy >
class CoolingUpdate_PRISM_GPU : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;

  typename Policy::Params policy_params;

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

  // RT groups
  int n_groups;
  std::vector<real_t> rt_groups_lower;
  std::vector<real_t> rt_groups_upper;

  std::vector<double> E_min;
  std::vector<double> E_max;

  real_t c_rad;
  real_t c_tilde;

  real_t T_blackbody;

  bool include_HM12_UVB;
  real_t UV_background_G0;

  bool sort_cells_by_cost;
  int max_substeps;
  int batch_size;
  real_t rad_residual_floor;
  bool rt_smooth;

  // Flattened (element << 8 | stage) work items for the team-parallel Jacobi
  // metal-ion update: one entry per true ion stage of the configured network
  Kokkos::View<uint16_t*> ion_map_d;
  int n_ion_work = 0;
  int ions_team_size = 0;

  // Solver-state pools for the split-kernel driver, cached across update() calls
  Kokkos::View<PrismCellState*> cell_state_pool;
  Kokkos::View<CompactIonData*> ion_state_pool;
  Kokkos::View<double*> ddt_pool, total_time_pool;
  Kokkos::View<int*> iterations_pool;
  Kokkos::View<uint8_t*> done_pool;
  Kokkos::View<uint32_t*> active_list_pool, active_list_next_pool;

  RTZ_type rtz_solver;

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

  CoolingUpdate_PRISM_GPU(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers
  ) :
        foreach_cell    ( foreach_cell),
        timers          ( timers),
        policy_params   ( Policy::getParams(configMap)),
        cosmo_run       ( configMap.getValue<bool>("cosmology", "active", false)),
        data_path       ( configMap.getValue<std::string>("cooling", "data_path")),
        ions            ( configMap.getValue<std::vector<std::string>>("cooling", "ions" ) ),
        rt_groups_lower ( configMap.getValue<std::vector<real_t>>("rad", "photon_groups_lower",
                          {13.6, 15.2, 24.59, 54.42}) ),
        rt_groups_upper ( configMap.getValue<std::vector<real_t>>("rad", "photon_groups_upper",
                          {15.2, 24.59, 54.42, 500.0}) ),
        rtz_solver(data_path)
  {
    auto code_velocity = Units::code_units().getUnit<Units::Velocity>();
    c_rad              = configMap.getValue_in_code_unit<Units::Velocity>("rad", "c_rad", "speedoflight");
    c_tilde = (c_rad * code_velocity).convert_to(Units::SPEEDOFLIGHT());
    n_groups           = configMap.getValue<int>("rad", "n_groups", 4);
    T_blackbody        = configMap.getValue<real_t>("cooling", "T_blackbody", 1e4);
    // Cells that took many iterations last step are scheduled first, so
    // expensive cells share batches instead of straggling behind cheap ones
    sort_cells_by_cost = configMap.getValue<bool>("cooling", "sort_cells_by_cost", true);
    // Safety cap on the number of subcycle iterations per cell
    max_substeps       = configMap.getValue<int>("cooling", "max_substeps", 100000);
    // Cells are processed in batches of at most this many, sized to bound the
    // per-cell solver-state pools (CompactIonData alone is ~1.2 kB/cell in the
    // reduced-precision float build, RTZ_REDUCED_PRECISION; ~2.4 kB in double)
    batch_size         = configMap.getValue<int>("cooling", "batch_size", 1048576);
    rad_residual_floor = configMap.getValue<real_t>("cooling", "rad_residual_floor", 1e-3);
    rt_smooth          = configMap.getValue<bool>("cooling", "rt_smooth", true);
    include_HM12_UVB   = configMap.getValue<bool>("cooling", "include_HM12_UVB", true);
    UV_background_G0   = configMap.getValue<real_t>("cooling", "UV_background_G0", 0.0070977);
    PRISM_GPU::parseIonInputs(
      ions,
      this->nions_and_molecules, this->elems2passive, this->ions2passive, this->elem2atomicnum, this->ion_counts, this->molecule_counts,
      include_H2
    );
    PRISM_GPU::parsePhotonGroupInputs(n_groups, rt_groups_lower, rt_groups_upper, this->E_min, this->E_max);
    rtz_solver.set_reduced_speed_of_light_factor(c_tilde);
    rtz_solver.set_UV_background_G0(UV_background_G0);
    std::array<double, N_GROUPS> E_min_tmp;
    std::array<double, N_GROUPS> E_max_tmp;
    for (int i = 0; i < n_groups; ++i) {
      E_min_tmp[i] = this->E_min[i];
      E_max_tmp[i] = this->E_max[i];
    }
    rtz_solver.set_photon_groups(E_min_tmp, E_max_tmp);

    // Work items for the team-parallel metal-ion update (metal_ions_step): every
    // true ion stage (n_ions = atomic number + 1, as in initialize_elements) of
    // every metal (Z >= 3) in the network. H and He are updated separately by
    // HandHe_ions_step; molecule slots (H2) are handled by molecular_step.
    {
      std::vector<uint16_t> ion_map_host;
      for (int i = 3; i < MAX_ELEMENTS; ++i) {
        if (ions2passive[i] == -1) continue;
        for (int j = 0; j <= i; ++j)
          ion_map_host.push_back( static_cast<uint16_t>((i << 8) | j) );
      }
      n_ion_work = ion_map_host.size();
      // One thread per ion stage, rounded up to full warps, at most 4 warps
      ions_team_size = std::min( ((n_ion_work + 31)/32)*32, 128 );
      ion_map_d = Kokkos::View<uint16_t*>("PRISM_ion_map", n_ion_work);
      auto ion_map_h = Kokkos::create_mirror_view(ion_map_d);
      for (int t = 0; t < n_ion_work; ++t)
        ion_map_h(t) = ion_map_host[t];
      Kokkos::deep_copy(ion_map_d, ion_map_h);
    }

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
          std::string iion_roman = PRISM_GPU::int2roman(iion+1);
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

    // CO
    std::vector<UserData::FieldAccessor::FieldInfo> CO_in;
    std::vector<UserData::FieldAccessor::FieldInfo> CO_out;
    if (include_CO) {
      std::string field_name = "nCO";
      CO_in.push_back({field_name.c_str(), 0});
      CO_out.push_back({field_name.c_str(), 0});
    }
    UserData::FieldAccessor Uin_CO = U.getAccessor( CO_in );
    UserData::FieldAccessor Uout_CO = U.getAccessor( CO_out );

    std::vector<UserData::FieldAccessor::FieldInfo> rt_fields;
    rt_fields.reserve(4 * n_groups);
    for (int g = 0; g < n_groups; ++g) {
      const int off = 4 * g;
      rt_fields.push_back({"e_rad_"  + std::to_string(g) + "_next", off + 0});
      rt_fields.push_back({"fx_rad_" + std::to_string(g) + "_next", off + 1});
      rt_fields.push_back({"fy_rad_" + std::to_string(g) + "_next", off + 2});
      rt_fields.push_back({"fz_rad_" + std::to_string(g) + "_next", off + 3});
    }
    UserData::FieldAccessor Uin_rt = U.getAccessor(rt_fields);
    UserData::FieldAccessor Uout_rt = U.getAccessor(rt_fields);

    // Pre-transport photon fields for rt_smooth
    std::vector<UserData::FieldAccessor::FieldInfo> rt_fields_old;
    rt_fields_old.reserve(4 * n_groups);
    for (int g = 0; g < n_groups; ++g) {
      const int off = 4 * g;
      rt_fields_old.push_back({"e_rad_"  + std::to_string(g), off + 0});
      rt_fields_old.push_back({"fx_rad_" + std::to_string(g), off + 1});
      rt_fields_old.push_back({"fy_rad_" + std::to_string(g), off + 2});
      rt_fields_old.push_back({"fz_rad_" + std::to_string(g), off + 3});
    }
    UserData::FieldAccessor Uin_rt_old = U.getAccessor(rt_fields_old);

    UserData::FieldAccessor Uout_debug = U.getAccessor({{"PRISM_iter", 0}});

    auto mp_over_kb    = Units::PROTON_MASS() / Units::KBOLTZ();
    auto K = Units::Kelvin();
    auto code_density  = Units::code_units().getUnit<Units::Density>();
    auto code_pressure = Units::code_units().getUnit<Units::Pressure>();
    auto code_time     = Units::code_units().getUnit<Units::Time>();
    auto code2cm = Units::supercomoving_to_physical<Units::Length>(
      (1 * Units::code_units().getUnit<Units::Length>()).convert_to(Units::cm()),
      aexp
    );

    real_t dt_s = (dt * code_time).convert_to(Units::second());

    const real_t gamma0 = this->policy_params.policy_params.gamma0;

    const std::array<int, MAX_ELEMENTS> &nions_and_molecules = this->nions_and_molecules;
    const std::array<int, MAX_ELEMENTS> &elems2passive = this->elems2passive;
    const std::array<int, MAX_ELEMENTS> &ions2passive = this->ions2passive;
    // const int n_groups = this->n_groups;
    const RTZ_type& rtz_solver = this->rtz_solver;
    const bool include_HM12_UVB = this->include_HM12_UVB;

    timers.get("CoolingUpdate_PRISM").start();

    using exec_space = Kokkos::DefaultExecutionSpace;

    Kokkos::View<int> max_iter_reached_device("PRISM_max_iter_reached");

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    // Solver data shared by all kernels
    const TabulatedData tabData = rtz_solver.get_tabData();
    const Element* elements = rtz_solver.elements_d.data();
    const double UV_G0 = rtz_solver.get_UV_background_G0();
    const double primary_cr_rate = rtz_solver.get_generic_cosmic_ray_ionization_rate();

    // Physics flags
    const PhysicsFlags flags {
        .include_collisional_ionization = true,
        .include_photoionization        = true,
        .include_cosmic_ray_ionization  = true,
        .include_HM12_UVB               = include_HM12_UVB,
        .include_dust_recombination     = true,
        .include_charge_exchange        = true,
    };

    // ------ Build the (optionally cost-sorted) list of local cells ------
    const auto shape = Uin.getShape();
    const uint32_t nbCells = shape.bx * shape.by * shape.bz * shape.nbOcts;

    Kokkos::View<ForeachCell::CellIndex*> all_cells(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_all_cells"), nbCells );
    {
      Kokkos::View<float*> keys(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_cost"), nbCells );
      foreach_cell.foreach_cell( "PRISM_index", shape,
        KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
      {
        uint32_t flat = iCell.iOct.iOct * (iCell.bx*iCell.by*iCell.bz)
                      + iCell.k * (iCell.bx*iCell.by) + iCell.j * iCell.bx + iCell.i;
        all_cells(flat) = iCell;
        // Cells that took many iterations last step get scheduled first, so
        // expensive cells share batches instead of straggling behind cheap ones
        keys(flat) = -static_cast<float>( Uout_debug.at(iCell, 0) );
      });
      if( sort_cells_by_cost )
        Kokkos::Experimental::sort_by_key( exec_space(), keys, all_cells );
    }

    // ------ Per-cell solver-state pools, persisted across the step kernels ------
    // CompactIonData alone is ~1.2 kB per cell in the reduced-precision float
    // build (RTZ_REDUCED_PRECISION; ~2.4 kB in double), so the pools are sized
    // for at most batch_size cells and the cell list is processed in batches.
    const uint32_t pool_size = std::min<uint32_t>( batch_size, nbCells );
    if( ion_state_pool.extent(0) < pool_size )
    {
      cell_state_pool       = Kokkos::View<PrismCellState*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_cell_state"), pool_size );
      ion_state_pool        = Kokkos::View<CompactIonData*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_ion_state"), pool_size );
      ddt_pool              = Kokkos::View<double*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_ddt"), pool_size );
      total_time_pool       = Kokkos::View<double*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_total_time"), pool_size );
      iterations_pool       = Kokkos::View<int*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_iterations"), pool_size );
      done_pool             = Kokkos::View<uint8_t*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_done"), pool_size );
      active_list_pool      = Kokkos::View<uint32_t*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_active"), pool_size );
      active_list_next_pool = Kokkos::View<uint32_t*>( Kokkos::view_alloc(Kokkos::WithoutInitializing, "PRISM_active_next"), pool_size );
    }
    // Local handles for device-lambda capture (KOKKOS_LAMBDA must not capture this)
    auto cell_state      = this->cell_state_pool;
    auto ion_state       = this->ion_state_pool;
    auto ddt_pool        = this->ddt_pool;
    auto total_time_pool = this->total_time_pool;
    auto iterations_pool = this->iterations_pool;
    auto done_pool       = this->done_pool;
    auto active          = this->active_list_pool;
    auto active_next     = this->active_list_next_pool;

    const uint16_t* ion_map = ion_map_d.data();
    const int n_ion_work = this->n_ion_work;
    const int ions_team_size = this->ions_team_size;
    const int max_substeps = this->max_substeps;
    const double rad_residual_floor = this->rad_residual_floor;
    const bool rt_smooth = this->rt_smooth;

    int ions_team_size_clamped = -1;

    for( uint32_t batch_start = 0; batch_start < nbCells; batch_start += pool_size )
    {
      const uint32_t n_batch = std::min( pool_size, nbCells - batch_start );
      auto batch_cells = Kokkos::subview( all_cells, std::make_pair(batch_start, batch_start+n_batch) );

      // ------ Load: initialize the solver state from field data ------
      Kokkos::parallel_for( "PRISM_load", Kokkos::RangePolicy<>(0, n_batch),
        KOKKOS_LAMBDA( uint32_t s )
      {
        const real_t gamma_m1 = gamma0 - 1.0;
        const ForeachCell::CellIndex iCell = batch_cells(s);
        CompactIonData& n_and_ion_fracs_loc = ion_state(s);
        PrismCellState& cs = cell_state(s);

        // Get local hydro quantities
        ConsState u = policy.getConsState(Uin, iCell);
        PrimState q = policy.consToPrim(u);

        real_t p_thermal = q.p;
        if constexpr ( Policy::has_dual_energy() )
          p_thermal = gamma_m1 * u.e_int;

        // Initial state
        auto rho_physical = Units::supercomoving_to_physical<Units::Density>(q.rho, aexp) * code_density;
        auto P_physical = Units::supercomoving_to_physical<Units::Pressure>(p_thermal, aexp) * code_pressure;

        auto cell_size = cells.getCellSize(iCell);
        cs.dx_cm = cell_size[IX] * code2cm;

        // Compute T/µ
        real_t T_over_mu = (P_physical / rho_physical * mp_over_kb).convert_to(K);

        if (T_over_mu < 2.727e0) {
          // Can't have gas cool below the CMB temperature floor, and
          // it will also heavily slow down the chemistry solver
          T_over_mu = 2.727e0;
        }

        // Initialize CompactIonData from field data
        n_and_ion_fracs_loc = CompactIonData{};
        n_and_ion_fracs_loc.init_offsets(elements);

        for (int i = 1; i < MAX_ELEMENTS; ++i) {
          if (ions2passive[i] == -1) continue; // Skip elements not in network
          // Element channels are transported as conserved number densities n_i.
          n_and_ion_fracs_loc.n_element[i] = Uin_passive.at(iCell, elems2passive[i]);
          // Set ion fractions
          for (int j = 0; j < nions_and_molecules[i]; ++j) {
            int index = ions2passive[i] + j;
            n_and_ion_fracs_loc[i].ion_fracs[j] = Uin_passive.at(iCell, index) / u.rho;
          }
        }

        cs.dNpdt = {};
        cs.dFpdt = {};
        for (int g = 0; g < N_GROUPS; ++g) {
          int index = 4 * g; // TODO: don't hardcode this
          cs.N_PHOT[g] = Uin_rt.at(iCell, index);
          for (int d = 0; d < 3; ++d) {
            cs.F_PHOT[g][d] = Uin_rt.at(iCell, index + d + 1);
          }
          if (rt_smooth) {
            // The subcycle sweeps between the pre- and post-transport values
            const double N_old = Uin_rt_old.at(iCell, index);
            cs.N_PHOT0[g] = fmax(N_old, cs.N_PHOT[g]);
            cs.dNpdt[g] = (cs.N_PHOT[g] - N_old) / dt_s;
            cs.N_PHOT[g] = N_old;
            for (int d = 0; d < 3; ++d) {
              const double F_old = Uin_rt_old.at(iCell, index + d + 1);
              cs.dFpdt[g][d] = (cs.F_PHOT[g][d] - F_old) / dt_s;
              cs.F_PHOT[g][d] = F_old;
            }
          } else {
            cs.N_PHOT0[g] = cs.N_PHOT[g];
          }
        }
        cs.N_phot_new = {};
        cs.phot_att = {};

        if (include_CO)
          cs.nCO = Uin_CO.at(iCell, 0);
        else
          cs.nCO = 0;

        // Preamble of solve_chemistry_and_cooling
        cs.dust_ratio = get_dust_mass_and_depletion(
          elements, n_and_ion_fracs_loc.n_element[8], n_and_ion_fracs_loc.n_element[1], cs.nCO );
        cs.loc_rho = get_rho(elements, n_and_ion_fracs_loc);
        apply_dust_depletion(elements, cs.dust_ratio, n_and_ion_fracs_loc, cs.nCO);

        // Entry of subcycle_chemistry
        cs.ss_factor = 1.0;
        if (include_self_shielding)
          cs.ss_factor = exp(-1.0 * n_and_ion_fracs_loc.n_element[1] / 1E-2);

        cs.sub = SubcycleState{};
        cs.sub.success_counter = 0;
        cs.sub.Tmu_old = T_over_mu;
        cs.sub.Tmu_new = T_over_mu;
        cs.sub.ne_initial = get_ne(elements, n_and_ion_fracs_loc, false);

        // Per-cell driver state
        ddt_pool(s) = dt_s; // first try a single step over the full dt
        total_time_pool(s) = 0.0;
        iterations_pool(s) = 0;
        done_pool(s) = 0;
        active(s) = s;
      });

      // ------ Host-side subcycle loop: one pass = one subcycle iteration per active cell ------
      // Each pass launches the physics steps as separate kernels over the
      // still-active cells (per-cell adaptive ddt, no host sync between
      // launches). The active list is compacted only after passes 1 and 10 —
      // the vast majority of cells finish in the first iteration — and after
      // tail_pass iterations the stiff stragglers are finished in one on-device
      // kernel instead of thousands more per-pass launches.

      constexpr bool is_gpu = !std::is_same_v<Kokkos::DefaultExecutionSpace,
                                              Kokkos::DefaultHostExecutionSpace>;
      constexpr int tail_pass = is_gpu ? 30 : 0;
      const size_t ion_scratch = MAX_TOTAL_IONS * sizeof(CompactIonData::xion_t);

      uint32_t n_active = n_batch;
      int pass = 0;

      while( n_active > 0 && pass < max_substeps )
      {
        // After tail_pass split passes (immediately on host builds), finish
        // the remaining cells in one on-device kernel: one team per cell runs
        // the whole remaining subcycle (team-parallel cooling + sequential-
        // H/He / parallel-metal ions)
        if( pass >= tail_pass )
        {
          auto active_tail = active;
          auto tail_functor = KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type& team )
          {
            const uint32_t s = active_tail(team.league_rank());
            if( done_pool(s) ) return;
            CompactIonData& n_and_ion_fracs_loc = ion_state(s);
            PrismCellState& cs = cell_state(s);

            // Snapshot buffer for metal_ions_step, allocated once per cell:
            // team scratch is a bump allocator, so a per-iteration get_shmem
            // would return nullptr from the second subcycle iteration on
            auto* x_snap = static_cast<CompactIonData::xion_t*>(
              team.team_scratch(0).get_shmem(ion_scratch) );

            // Every lane keeps its own iteration counter so the loop condition
            // stays team-uniform (the pool copy is bookkeeping, written by lane 0)
            int iterations = iterations_pool(s);

            while( iterations < max_substeps )
            {
              iterations += 1;
              const double total_time = total_time_pool(s);

              Kokkos::single( Kokkos::PerTeam(team), [&]()
              {
                iterations_pool(s) = iterations;

                double ddt = ddt_pool(s);
                subcycle_prep_iteration<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    elements, n_and_ion_fracs_loc, cs.nCO, dt_s, total_time,
                    primary_cr_rate, ddt, cs.sub );
                ddt_pool(s) = ddt; // prep may clamp ddt to the remaining time

                radiation_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    ddt, elements, n_and_ion_fracs_loc, tabData, cs.nCO, cs.dx_cm,
                    cs.dust_ratio, cs.N_PHOT, cs.N_phot_new, cs.phot_att, cs.sub,
                    rad_residual_floor, &cs.N_PHOT0,
                    rt_smooth, cs.dNpdt );
              });
              team.team_barrier();

              const double ddt = ddt_pool(s);

              cooling_step_team<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                  team, cs.cool_rate_pair, ddt, aexp, elements, n_and_ion_fracs_loc, tabData, flags,
                  cs.loc_rho, cs.nCO, cs.dust_ratio, primary_cr_rate, cs.ss_factor,
                  UV_G0, cs.N_phot_new, cs.sub );
              team.team_barrier();

              Kokkos::single( Kokkos::PerTeam(team), [&]()
              {
                molecular_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    ddt, elements, n_and_ion_fracs_loc, tabData, cs.nCO,
                    cs.dust_ratio, UV_G0, cs.N_phot_new, cs.sub );
              });
              team.team_barrier();

              HandHe_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                  team, ddt, elements, n_and_ion_fracs_loc, tabData, flags,
                  cs.dust_ratio, UV_G0, primary_cr_rate, cs.ss_factor,
                  cs.N_phot_new, cs.sub );
              metal_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                  team, ddt, elements, n_and_ion_fracs_loc, tabData, flags,
                  cs.dust_ratio, UV_G0, primary_cr_rate, cs.ss_factor,
                  cs.N_phot_new, ion_map, n_ion_work, x_snap, cs.sub );
              team.team_barrier();

              Kokkos::single( Kokkos::PerTeam(team), [&]()
              {
                double ddt_loc = ddt;
                double total_time_loc = total_time;
                subcycle_commit_and_control<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    elements, tabData, n_and_ion_fracs_loc, cs.nCO, cs.N_PHOT, cs.F_PHOT,
                    cs.N_phot_new, cs.phot_att, ddt_loc, total_time_loc, cs.sub,
                    rt_smooth, cs.dFpdt );
                ddt_pool(s) = ddt_loc;
                total_time_pool(s) = total_time_loc;
              });
              team.team_barrier();

              if( fabs(total_time_pool(s) - dt_s)/dt_s < 1e-6 )
                break;
            }

            Kokkos::single( Kokkos::PerTeam(team), [&]() { done_pool(s) = 1; });
          };

          // ions_team_size (network stages rounded to warps) on device, unless
          // the functor's resource use forces a smaller block; a single thread
          // per cell on host, where the fused kernel IS the monolithic loop
          // (barriers and singles collapse to straight serial execution)
          const int team_size = is_gpu
            ? std::min( ions_team_size,
                Kokkos::TeamPolicy<>(n_active, Kokkos::AUTO)
                  .set_scratch_size(0, Kokkos::PerTeam(ion_scratch))
                  .team_size_max(tail_functor, Kokkos::ParallelForTag()) )
            : 1;

          // Cost-sorted cells put the expensive stragglers first, so the host
          // league needs dynamic scheduling (a blocked static partition would
          // hand one thread all of them); CUDA hardware-schedules blocks and
          // ignores the tag
          Kokkos::parallel_for( "PRISM_tail",
            Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>(n_active, team_size)
              .set_scratch_size(0, Kokkos::PerTeam(ion_scratch)),
            tail_functor );

          break;
        }

        // (1) prep + radiation: one thread per cell
        Kokkos::parallel_for( "PRISM_prep_rad", Kokkos::RangePolicy<>(0, n_active),
          KOKKOS_LAMBDA( uint32_t idx )
        {
          const uint32_t s = active(idx);
          if( done_pool(s) ) return;
          CompactIonData& n_and_ion_fracs_loc = ion_state(s);
          PrismCellState& cs = cell_state(s);

          iterations_pool(s) += 1;

          double ddt = ddt_pool(s);
          subcycle_prep_iteration<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
              elements, n_and_ion_fracs_loc, cs.nCO, dt_s, total_time_pool(s),
              primary_cr_rate, ddt, cs.sub );
          ddt_pool(s) = ddt; // prep may clamp ddt to the remaining time

          radiation_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
              ddt, elements, n_and_ion_fracs_loc, tabData, cs.nCO, cs.dx_cm,
              cs.dust_ratio, cs.N_PHOT, cs.N_phot_new, cs.phot_att, cs.sub,
              rad_residual_floor, &cs.N_PHOT0,
              rt_smooth, cs.dNpdt );
        });

        // (2a) cooling rate pair: two threads per cell (lane 0 -> T, lane 1 ->
        // 1.001*T). Packed flat so a 32-lane warp evaluates 16 cells with no
        // idle lanes; a RangePolicy has no intra-warp barrier, so the Newton
        // step that consumes the pair is the separate kernel (2b).
        Kokkos::parallel_for( "PRISM_cooling_rates", Kokkos::RangePolicy<>(0, 2*n_active),
          KOKKOS_LAMBDA( uint32_t t )
        {
          const uint32_t s = active(t/2);
          if( done_pool(s) ) return;
          PrismCellState& cs = cell_state(s);

          cooling_rate_lane<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
              t%2, cs.cool_rate_pair, aexp, elements, ion_state(s), tabData, flags,
              cs.nCO, cs.dust_ratio, primary_cr_rate, cs.ss_factor, UV_G0,
              cs.N_phot_new, cs.sub );
        });

        // (2b) cooling Newton step from the pair: one thread per cell
        // (3) molecules: one thread per cell
        // (4a) H/He ions: one thread per cell. Hydrogen and helium updated sequentially
        Kokkos::parallel_for( "PRISM_molecules_HandHe", Kokkos::RangePolicy<>(0, n_active),
          KOKKOS_LAMBDA( uint32_t idx )
        {
          const uint32_t s = active(idx);
          if( done_pool(s) ) return;
          PrismCellState& cs = cell_state(s);

          cooling_update<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
              cs.cool_rate_pair, ddt_pool(s), cs.loc_rho, cs.sub );

          molecular_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
              ddt_pool(s), elements, ion_state(s), tabData, cs.nCO,
              cs.dust_ratio, UV_G0, cs.N_phot_new, cs.sub );

          HandHe_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
              ddt_pool(s), elements, ion_state(s), tabData, flags,
              cs.dust_ratio, UV_G0, primary_cr_rate, cs.ss_factor,
              cs.N_phot_new, cs.sub );
        });

        // (4b) metal ions: one team per cell, one thread per ion stage in parallel
        {
          auto ions_functor = KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type& team )
          {
            const uint32_t s = active(team.league_rank());
            if( done_pool(s) ) return;
            PrismCellState& cs = cell_state(s);

            auto* x_snap = static_cast<CompactIonData::xion_t*>(
              team.team_scratch(0).get_shmem(ion_scratch) );

            metal_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                team, ddt_pool(s), elements, ion_state(s), tabData, flags,
                cs.dust_ratio, UV_G0, primary_cr_rate, cs.ss_factor,
                cs.N_phot_new, ion_map, n_ion_work, x_snap, cs.sub );
          };

          // ions_team_size (network stages rounded to warps) unless the
          // functor's resource use forces a smaller block; fixed across passes,
          // so query it only once
          if( ions_team_size_clamped < 0 )
            ions_team_size_clamped = std::min( ions_team_size,
              Kokkos::TeamPolicy<>(n_active, Kokkos::AUTO)
                .set_scratch_size(0, Kokkos::PerTeam(ion_scratch))
                .team_size_max(ions_functor, Kokkos::ParallelForTag()) );

          Kokkos::parallel_for( "PRISM_ions_metals",
            Kokkos::TeamPolicy<>(n_active, ions_team_size_clamped)
              .set_scratch_size(0, Kokkos::PerTeam(ion_scratch)),
            ions_functor );
        }

        // (5) commit + timestep control: one thread per cell
        Kokkos::parallel_for( "PRISM_commit", Kokkos::RangePolicy<>(0, n_active),
          KOKKOS_LAMBDA( uint32_t idx )
        {
          const uint32_t s = active(idx);
          if( done_pool(s) ) return;
          CompactIonData& n_and_ion_fracs_loc = ion_state(s);
          PrismCellState& cs = cell_state(s);

          double ddt = ddt_pool(s);
          double total_time = total_time_pool(s);
          subcycle_commit_and_control<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
              elements, tabData, n_and_ion_fracs_loc, cs.nCO, cs.N_PHOT, cs.F_PHOT,
              cs.N_phot_new, cs.phot_att, ddt, total_time, cs.sub,
              rt_smooth, cs.dFpdt );
          ddt_pool(s) = ddt;
          total_time_pool(s) = total_time;

          // Stop subcycling once the full hydro timestep has been integrated
          if( fabs(total_time - dt_s)/dt_s < 1e-6 )
            done_pool(s) = 1;
        });

        ++pass;

        // Compact the active list after the 1st and 10th passes only (the
        // parallel_scan's returned count is the only host sync per pass; the
        // vast majority of cells finish in the first iteration)
        if( pass == 1 || pass == 3 || pass == 10 || pass == 30 )
        {
          uint32_t n_alive = 0;
          Kokkos::parallel_scan( "PRISM_compact", Kokkos::RangePolicy<>(0, n_active),
            KOKKOS_LAMBDA( uint32_t idx, uint32_t& count, bool final )
          {
            const uint32_t s = active(idx);
            if( !done_pool(s) )
            {
              if( final )
                active_next(count) = s;
              ++count;
            }
          }, n_alive );
          std::swap( active, active_next );
          n_active = n_alive;
        }

        // After tail_pass iterations, finish the remaining stiff cells in one
        // on-device kernel: one team per cell runs the whole remaining subcycle
        // (team-parallel cooling + sequential-H/He / parallel-metal ions), so
        // the per-pass launches vanish for the latency-bound straggler tail.
        if( pass == tail_pass && n_active > 0 )
        {
          auto active_tail = active;
          auto tail_functor = KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type& team )
          {
            const uint32_t s = active_tail(team.league_rank());
            if( done_pool(s) ) return;
            CompactIonData& n_and_ion_fracs_loc = ion_state(s);
            PrismCellState& cs = cell_state(s);

            // Every lane keeps its own iteration counter so the loop condition
            // stays team-uniform (the pool copy is bookkeeping, written by lane 0)
            int iterations = iterations_pool(s);

            while( iterations < max_substeps )
            {
              iterations += 1;
              const double total_time = total_time_pool(s);

              Kokkos::single( Kokkos::PerTeam(team), [&]()
              {
                iterations_pool(s) = iterations;

                double ddt = ddt_pool(s);
                subcycle_prep_iteration<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    elements, n_and_ion_fracs_loc, cs.nCO, dt_s, total_time,
                    primary_cr_rate, ddt, cs.sub );
                ddt_pool(s) = ddt; // prep may clamp ddt to the remaining time

                radiation_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    ddt, elements, n_and_ion_fracs_loc, tabData, cs.nCO, cs.dx_cm,
                    cs.dust_ratio, cs.N_PHOT, cs.N_phot_new, cs.phot_att, cs.sub,
                    rad_residual_floor, &cs.N_PHOT0 );
              });
              team.team_barrier();

              const double ddt = ddt_pool(s);

              cooling_step_team<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                  team, cs.cool_rate_pair, ddt, aexp, elements, n_and_ion_fracs_loc, tabData, flags,
                  cs.loc_rho, cs.nCO, cs.dust_ratio, primary_cr_rate, cs.ss_factor,
                  UV_G0, cs.N_phot_new, cs.sub );
              team.team_barrier();

              Kokkos::single( Kokkos::PerTeam(team), [&]()
              {
                molecular_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    ddt, elements, n_and_ion_fracs_loc, tabData, cs.nCO,
                    cs.dust_ratio, UV_G0, cs.N_phot_new, cs.sub );
              });
              team.team_barrier();

              HandHe_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                  team, ddt, elements, n_and_ion_fracs_loc, tabData, flags,
                  cs.dust_ratio, UV_G0, primary_cr_rate, cs.ss_factor,
                  cs.N_phot_new, cs.sub );
              metal_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                  team, ddt, elements, n_and_ion_fracs_loc, tabData, flags,
                  cs.dust_ratio, UV_G0, primary_cr_rate, cs.ss_factor,
                  cs.N_phot_new, ion_map, n_ion_work, cs.sub );
              team.team_barrier();

              Kokkos::single( Kokkos::PerTeam(team), [&]()
              {
                double ddt_loc = ddt;
                double total_time_loc = total_time;
                subcycle_commit_and_control<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
                    elements, tabData, n_and_ion_fracs_loc, cs.nCO, cs.N_PHOT, cs.F_PHOT,
                    cs.N_phot_new, cs.phot_att, ddt_loc, total_time_loc, cs.sub );
                ddt_pool(s) = ddt_loc;
                total_time_pool(s) = total_time_loc;
              });
              team.team_barrier();

              if( fabs(total_time_pool(s) - dt_s)/dt_s < 1e-6 )
                break;
            }

            Kokkos::single( Kokkos::PerTeam(team), [&]() { done_pool(s) = 1; });
          };

          const int team_size = std::min( ions_team_size,
            Kokkos::TeamPolicy<>(n_active, Kokkos::AUTO)
              .set_scratch_size(0, Kokkos::PerTeam(ion_scratch))
              .team_size_max(tail_functor, Kokkos::ParallelForTag()) );

          Kokkos::parallel_for( "PRISM_tail",
            Kokkos::TeamPolicy<>(n_active, team_size)
              .set_scratch_size(0, Kokkos::PerTeam(ion_scratch)),
            tail_functor );

          n_active = 0;
        }
      } // End subcycle pass loop

      // ------ Store: write the solver state back to field data ------
      Kokkos::parallel_for( "PRISM_store", Kokkos::RangePolicy<>(0, n_batch),
        KOKKOS_LAMBDA( uint32_t s )
      {
        const real_t gamma_m1 = gamma0 - 1.0;
        const ForeachCell::CellIndex iCell = batch_cells(s);
        CompactIonData& n_and_ion_fracs_loc = ion_state(s);
        PrismCellState& cs = cell_state(s);

        // Epilogue of solve_chemistry_and_cooling: the committed T/µ is the
        // result; undo the dust depletion applied by the load kernel
        const real_t out_T_over_mu = cs.sub.Tmu_old;
        remove_dust_depletion(elements, cs.dust_ratio, n_and_ion_fracs_loc, cs.nCO);

        const int iters = iterations_pool(s);
        Uout_debug.at(iCell, 0) = iters;

        if( iters > max_iter_reached_device() )
          Kokkos::atomic_max(&max_iter_reached_device(), iters);

        // Hydro fields are untouched while the solver runs, so p_thermal and
        // rho_physical can be recomputed exactly as in the load kernel
        ConsState u = policy.getConsState(Uin, iCell);
        PrimState q = policy.consToPrim(u);

        real_t p_thermal = q.p;
        if constexpr ( Policy::has_dual_energy() )
          p_thermal = gamma_m1 * u.e_int;

        auto rho_physical = Units::supercomoving_to_physical<Units::Density>(q.rho, aexp) * code_density;

        // Write back element number densities and ion fractions
        for (int i = 1; i < MAX_ELEMENTS; ++i) {
          if (ions2passive[i] == -1) continue; // Skip elements not in network
          Uout_passive.at(iCell, elems2passive[i]) = n_and_ion_fracs_loc.n_element[i];
          for (int j = 0; j < nions_and_molecules[i]; ++j) {
            int index = ions2passive[i] + j;
            Uout_passive.at(iCell, index) = n_and_ion_fracs_loc[i].ion_fracs[j] * u.rho;
          }
        }

        for (int g = 0; g < N_GROUPS; ++g) {
          int index = 4 * g; // TODO: don't hardcode this
          Uout_rt.at(iCell, index) = cs.N_PHOT[g];
          for (int d = 0; d < 3; ++d) {
            Uout_rt.at(iCell, index + d + 1) = cs.F_PHOT[g][d];
          }
        }

        if (include_CO) {
          Uout_CO.at(iCell, 0) = cs.nCO;
        }

        const real_t p_thermal_new = (out_T_over_mu * Units::Kelvin() * rho_physical / mp_over_kb).convert_to(code_pressure);
        const real_t delta_e_int = (p_thermal_new - p_thermal) / gamma_m1;

        u.e_tot += delta_e_int;
        if constexpr ( Policy::has_dual_energy() )
          u.e_int += delta_e_int;
        policy.setConsState(Uin, iCell, u);
      });
    } // End batch loop

    int max_iter_reached = 0;
    Kokkos::deep_copy(max_iter_reached, max_iter_reached_device);
    scalar_data.set("PRISM_max_iterations_reached", max_iter_reached);

    Kokkos::fence(); // Make sure all updates are finished before stopping timer
    timers.get("CoolingUpdate_PRISM").stop();
  }
};
} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::CoolingUpdate_PRISM_GPU<dyablo::HyperbolicPolicy_Hydro>,
                  "CoolingUpdate_PRISM_GPU_hydro");

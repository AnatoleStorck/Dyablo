#include "Compute_dt_base.h"

#include "utils_hydro.h"

namespace dyablo {


class Compute_dt_rad : public Compute_dt
{
public:
  Compute_dt_rad(   ConfigMap& configMap,
                        ForeachCell& foreach_cell,
                        Timers& timers )
  : foreach_cell(foreach_cell),
    c_rad( configMap.getValue_in_code_unit<Units::Velocity>("rad", "c_rad", "speedoflight") ),
    n_groups( configMap.getValue<int>("rad", "n_groups", 1) ),
    wait_for_radiation(configMap.getValue<bool>("dt", "wait_for_radiation", false))
  {
    real_t default_cfl = 0.5;
    if (configMap.hasValue("hydro", "cfl")) {
      std::cout << "WARNING : hydro/cfl is deprecated in .ini, use dt/hydro_cfl instead !" << std::endl;
      default_cfl = configMap.getValue<real_t>("hydro", "cfl");
    }
    this->cfl = configMap.getValue<real_t>("dt", "hydro_cfl", default_cfl);
  }

  void compute_dt( UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t aexp = scalar_data.get<real_t>("aexp");
    real_t ctilde = Units::physical_to_supercomoving<Units::Velocity>(this->c_rad , aexp);
    int n_groups = this->n_groups;
    bool wait_for_radiation = this->wait_for_radiation;

    auto cells = foreach_cell.getCellMetaData();

    if (wait_for_radiation) {
      // This is a heuristic to avoid very small time steps at the beginning of the simulation when no stars have formed.
      // loop over the cells and find the max value of e_rad_0 in the domain.
      // If e_rad_0 < 1e-19, we consider that there is no radiation and we can use the hydro/particle cfl conditions instead of the radiative one.
      UserData::FieldAccessor Urt_check = U.getAccessor({ {"e_rad_0", 0} });

      std::vector<UserData::FieldAccessor::FieldInfo> rt_clean_fields;
      rt_clean_fields.reserve(4 * n_groups);

      for (int g = 0; g < n_groups; ++g) {
        const int off = 4 * g;
        rt_clean_fields.push_back({"e_rad_"  + std::to_string(g), off + 0});
        rt_clean_fields.push_back({"fx_rad_" + std::to_string(g), off + 1});
        rt_clean_fields.push_back({"fy_rad_" + std::to_string(g), off + 2});
        rt_clean_fields.push_back({"fz_rad_" + std::to_string(g), off + 3});
      }

      UserData::FieldAccessor Urt_clean = U.getAccessor(rt_clean_fields);

      real_t max_e_rad = 0;
      foreach_cell.reduce_cell( "compute_dt_rad_max_e_rad", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& max_e_rad_update )
      {
        max_e_rad_update = FMAX( max_e_rad_update, Urt_check.at(iCell, 0) );
      }, Kokkos::Max<real_t>(max_e_rad) );
      if (max_e_rad < 1e-19) {
        // set the radiation to 1e-20 (there are numerical errors which can build up)
        foreach_cell.foreach_cell( "compute_dt_rad_set_e_rad", U.getShape(),
        KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
        {
          for (int i = 0; i < 4*n_groups; i++)
            Urt_clean.at(iCell, i) = (i%4 == 0) ? 1e-20 : 0;
        } );
        return;
      }
    }

    // TODO : We don't have to iterate on every cell for this
    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt_rad", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];

	    inv_dt_update = FMAX( inv_dt_update, ctilde/dx+ ctilde/dy + ctilde/dz );
      
    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt_local = cfl / inv_dt;

    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }
  

private:
  ForeachCell& foreach_cell;
  real_t c_rad;
  int n_groups;
  bool wait_for_radiation;
  real_t cfl;
  
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_rad, "Compute_dt_rad" );

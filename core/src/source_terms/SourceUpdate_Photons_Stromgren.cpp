#include "SourceUpdate_base.h"

namespace dyablo {

/**
 * @brief Photons source term for the Stromgren sphere
 */
class SourceUpdate_Photons_Stromgren : public SourceUpdate
{
private: 
  ForeachCell& foreach_cell;
  real_t source_position, spawn_rate_physical, a_stop_emission;

  using SpawnRate = decltype( 1/Units::s() );
 
public:
  SourceUpdate_Photons_Stromgren(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    source_position(configMap.getValue_in_code_unit<Units::Length>("stromgren", "source_position", 0.0)),
    // spawn_rate comes from the ini file in the non-cosmo case. In the cosmo case it is computed from physical parameters of the problem (in InitialConditions_stromgren.cpp)
    spawn_rate_physical(configMap.getValue_in_code_unit<SpawnRate>("stromgren", "spawn_rate", "1e56 /s")),
    a_stop_emission(configMap.getValue<real_t>( "rad", "a_stop_emission", 1.0 ))
  {}

  void update( UserData &U, ScalarSimulationData& scalar_data)
  {
    using pos_t = ForeachCell::CellMetaData::pos_t;

    real_t aexp = scalar_data.get<real_t>("aexp");
    real_t dt = scalar_data.get<real_t>("dt");
    real_t spawn_rate = Units::physical_to_supercomoving<SpawnRate>(this->spawn_rate_physical, aexp);

    // Stop the production of photons when a_stop_emission is reached
    if( aexp > a_stop_emission )
      spawn_rate = 0.0;

    pos_t source_position {this->source_position,this->source_position, this->source_position};

    enum VarIndex_Stromgren{In_rad};

    UserData::FieldAccessor Uout = U.getAccessor({ {"e_rad_next", In_rad} });

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    Kokkos::parallel_for( "SourceUpdate_Photons_Stromgren::spawn", 1,
      KOKKOS_LAMBDA( int )
    {
      ForeachCell::CellIndex iCell_spawn = cells.getCellFromPos( source_position );

      auto size = cells.getCellSize(iCell_spawn);
      real_t V = size[IX]*size[IY]*size[IZ];

      Uout.at(iCell_spawn, VarIndex_Stromgren::In_rad) += dt * spawn_rate / V;;
    });
  }
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_Photons_Stromgren, 
                  "SourceUpdate_Photons_Stromgren" );
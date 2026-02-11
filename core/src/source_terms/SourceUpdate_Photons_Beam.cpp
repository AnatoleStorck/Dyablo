#include "SourceUpdate_base.h"

namespace dyablo{

/**
 * @brief Photons source term for a single photons beam
 */
class SourceUpdate_Photons_Beam : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  real_t source_position;
  real_t spawn_rate;
  real_t c_rad;

  using SpawnRate = decltype( 1/Units::Time(0) );

public:
  SourceUpdate_Photons_Beam(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  :  foreach_cell(foreach_cell),
     source_position(configMap.getValue_in_code_unit<Units::Length>("rad", "source_position", 5.0)),
     spawn_rate(configMap.getValue_in_code_unit<SpawnRate>("rad", "spawn_rate", "1e56 /s")),
     c_rad( configMap.getValue_in_code_unit<Units::Velocity>("rad", "c_rad", "speedoflight") )
  { }

  void update( UserData &U, ScalarSimulationData& scalar_data)
  {
    using pos_t = ForeachCell::CellMetaData::pos_t;

    real_t dt = scalar_data.get<real_t>("dt");

    pos_t source_position {this->source_position/5.0, this->source_position, this->source_position};
    real_t aexp = scalar_data.get<real_t>("aexp");
    real_t ctilde = Units::physical_to_supercomoving<Units::Velocity>(this->c_rad , aexp);
    real_t spawn_rate = this->spawn_rate;

    enum VarIndex_Beam{In_rad, In_fx_rad, In_fy_rad, In_fz_rad};

    UserData::FieldAccessor Uout = U.getAccessor({ 
        {"e_rad_next",  In_rad},
        {"fx_rad_next", In_fx_rad},
        {"fy_rad_next", In_fy_rad},
        {"fz_rad_next", In_fz_rad}
      });

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    Kokkos::parallel_for( "SourceUpdate_Photons_Beam::spawn", 1, KOKKOS_LAMBDA( int )
    {
      ForeachCell::CellIndex iCell_spawn = cells.getCellFromPos( source_position );

      auto size = cells.getCellSize(iCell_spawn);
      real_t V = size[IX]*size[IY]*size[IZ];

      real_t n_rad = Uout.at(iCell_spawn, VarIndex_Beam::In_rad);
      n_rad += dt * spawn_rate / V;
      Uout.at(iCell_spawn, VarIndex_Beam::In_rad) = n_rad;
      Uout.at(iCell_spawn, VarIndex_Beam::In_fx_rad) = n_rad * ctilde;
      Uout.at(iCell_spawn, VarIndex_Beam::In_fy_rad) = 0.0;
      Uout.at(iCell_spawn, VarIndex_Beam::In_fz_rad) = 0.0;

    });
 }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_Photons_Beam, 
                  "SourceUpdate_Photons_Beam" );
#include "PassiveScalar_IC_base.h"

namespace dyablo{

struct PassiveScalar_IC_kelvin_helmholtz : public PassiveScalar_IC {
  ForeachCell &foreach_cell;

  std::string passive_name;

  const real_t z1;      // Lower limit for the transition
  const real_t z2;      // Upper limit for the transition
  const real_t a;       // Smoothing factor for the transition

  PassiveScalar_IC_kelvin_helmholtz(  ConfigMap& configMap,
                                      ForeachCell& foreach_cell,  
                                      Timers& timers,
                                      std::string passive_scalar_name ) : 
    foreach_cell(foreach_cell),
    passive_name(passive_scalar_name),
    z1(configMap.getValue<real_t>("KH", "z1", 0.5)),
    z2(configMap.getValue<real_t>("KH", "z2", 1.5)),
    a(configMap.getValue<real_t>("KH", "a", 0.05))
  {
  }

  void init( UserData &U ) {
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    const UserData::FieldAccessor Uout = U.getAccessor( {{passive_name, 0}} );

    const real_t z1 = this->z1;
    const real_t z2 = this->z2;
    const real_t a  = this->a;

    foreach_cell.foreach_cell( "PassiveScalar_IC_kelvin_helmholtz::fill_U", U.getShape(),
                KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
    {
      ForeachCell::CellMetaData::pos_t pos = cellmetadata.getCellCenter(iCell_U);
      const real_t y = pos[IY];
      const real_t q1 = tanh((y-z1)/a);
      const real_t q2 = tanh((y-z2)/a);
      const real_t c = 0.5 * (q2 - q1 + 2.0);
      Uout.at(iCell_U, 0) = c;
    });
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::PassiveScalar_IC_Factory, 
                 dyablo::PassiveScalar_IC_kelvin_helmholtz,
                 "kelvin_helmholtz");


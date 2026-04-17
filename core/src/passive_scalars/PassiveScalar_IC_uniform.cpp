#include "PassiveScalar_IC_base.h"

namespace dyablo{

/**
 * @brief Uniform initialization of passive scalars with a constant density 
 */
struct PassiveScalar_IC_uniform : public PassiveScalar_IC {
  ForeachCell &foreach_cell;

  std::string passive_name;

  const real_t rho0; // Density to apply on the grid

  PassiveScalar_IC_uniform(  ConfigMap& configMap,
                             ForeachCell& foreach_cell,  
                             Timers& timers,
                             std::string passive_scalar_name ) : 
    foreach_cell(foreach_cell),
    passive_name(passive_scalar_name),
    rho0(configMap.getValue<real_t>("passive_scalars", "IC_uniform/" + passive_scalar_name, 1.0)){}

  void init( UserData &U ) {
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    const UserData::FieldAccessor Uout = U.getAccessor( {{passive_name, 0}} );

    const real_t rho0 = this->rho0;

    foreach_cell.foreach_cell( "PassiveScalar_IC_uniform::fill_U", U.getShape(),
                KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
    {
      Uout.at(iCell_U, 0) = rho0;
    });
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::PassiveScalar_IC_Factory, 
                 dyablo::PassiveScalar_IC_uniform,
                 "uniform");


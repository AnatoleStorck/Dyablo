#include "PassiveScalar_IC_base.h"
#include "utils/units/Units.h"

namespace dyablo{

/**
 * @brief Density-based (for now) initialization of a refinement-tag passive scalar.
 *
 * Sets the scalar to a fraction of 1 in cells whose gas density is above a
 * threshold and 0 elsewhere, so it tags a region for RefineCondition_passive_scalar.
 *
 * NOTE: Passive scalars are stored conserved (rho_scalar = rho * fraction) so that
 * hydro transports them with the mass flux.
 */
struct PassiveScalar_IC_refinement : public PassiveScalar_IC {
  ForeachCell &foreach_cell;

  std::string target_name; // passive scalar field to tag (set_refinement/scalar)

  const real_t density_threshold; // gas density above which the tag is set to 1

  PassiveScalar_IC_refinement(  ConfigMap& configMap,
                                ForeachCell& foreach_cell,
                                Timers& timers,
                                std::string passive_scalar_name ) :
    foreach_cell(foreach_cell),
    target_name(configMap.getValue<std::string>("passive_scalars", "refinement_scalar_name", passive_scalar_name)),
    density_threshold(configMap.getValue_in_code_unit<Units::Density>("passive_scalars", "set_refinement_density_threshold", "1e3 proton_mass / cm**3")){}

  void init( UserData &U ) {
    enum VarIndex { ISCALAR, IRHO };
    const UserData::FieldAccessor Uout = U.getAccessor( {{target_name, ISCALAR}, {"rho", IRHO}} );

    const real_t density_threshold = this->density_threshold;

    printf("PassiveScalar_IC_refinement : initializing scalar '%s' with density threshold %e code units\n",
           target_name.c_str(), density_threshold);

    foreach_cell.foreach_cell( "PassiveScalar_IC_refinement::fill_U", U.getShape(),
                KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
    {
      const real_t rho = Uout.at(iCell_U, IRHO);
      // Store the conserved tag : fraction 1 (=rho) above the threshold, 0 below.
      Uout.at(iCell_U, ISCALAR) = (rho > density_threshold) ? rho : real_t(0);
    });
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::PassiveScalar_IC_Factory,
                 dyablo::PassiveScalar_IC_refinement,
                 "set_refinement");

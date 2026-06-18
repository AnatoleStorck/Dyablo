#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "utils/misc/Dyablo_assert.h"

#include <string>
#include <vector>

namespace dyablo {

/**
 * @brief Refinement region mask based on a passive scalar.
 *
 * Remove REFINE markers produced by the other conditions wherever the cell's
 * refinement passive scalar is <= threshold (0.5 by default).
 *
 * NOTE: Passive scalars are stored conserved (rho_scalar = rho * fraction) so that
 * hydro transports them with the mass flux.
 */
class RefineCondition_passive_scalar : public RefineCondition
{
public:
  RefineCondition_passive_scalar( ConfigMap& configMap,
                                  ForeachCell& foreach_cell,
                                  Timers& timers )
    : foreach_cell( foreach_cell ),
      threshold( configMap.getValue<real_t>("passive_scalars", "refine_passive_scalar_threshold", 0.5) )
  {
    std::string scalar_name = configMap.getValue<std::string>("passive_scalars", "refinement_scalar_name", "refinement");
    std::vector<std::string> passive_scalars_names =
      configMap.getValue<std::vector<std::string>>("passive_scalars", "passive_scalars_names", {});

    scalar_index = -1;
    for( size_t i=0; i<passive_scalars_names.size(); ++i )
      if( passive_scalars_names[i] == scalar_name )
        scalar_index = (int)i;

    DYABLO_ASSERT_HOST_RELEASE( scalar_index >= 0,
      "RefineCondition_passive_scalar : passive_scalars/refinement_scalar_name='" << scalar_name
      << "' was not found in passive_scalars/passive_scalars_names" );
  }

  bool is_refinement_mask() const override { return true; }

  void mark_cells( UserData& U, ScalarSimulationData& /*scalar_data*/ ) override
  {
    using CellIndex = ForeachCell::CellIndex;

    AMRmesh& pmesh = foreach_cell.get_amr_mesh();
    uint32_t nbOcts = pmesh.getNumOctants();

    std::string field = "rho_scalar_" + std::to_string( scalar_index );
    UserData::FieldAccessor Uin = U.getAccessor( {{field, ISCALAR}, {"rho", IRHO}} );

    // Per octant : maximum advected scalar fraction over the octant's cells.
    // Refinement is allowed if *any* cell of the octant is inside the region (frac > threshold).
    Kokkos::View<real_t*> oct_frac_max("RefineCondition_passive_scalar::frac", nbOcts);
    Kokkos::deep_copy( oct_frac_max, real_t(-1.0) );
    foreach_cell.foreach_cell( "RefineCondition_passive_scalar::reduce", U.getShape(),
      KOKKOS_LAMBDA( const CellIndex& iCell )
    {
      real_t rho = Uin.at(iCell, IRHO);
      real_t frac = (rho > 0) ? Uin.at(iCell, ISCALAR) / rho : real_t(0);
      Kokkos::atomic_fetch_max( &oct_frac_max( iCell.getOct() ), frac );
    });

    // Read the markers requested by the other conditions and veto refinement
    // (REFINE -> NOCHANGE) wherever the octant is outside the region.
    /** 
     * NOTE: This does not affect COARSEN or NOCHANGE markers.
     */
    Kokkos::View<int*> markers("RefineCondition_passive_scalar::markers", nbOcts);
    Kokkos::deep_copy( markers, pmesh.getMarkers() );
    const real_t threshold_loc = this->threshold;
    Kokkos::parallel_for( "RefineCondition_passive_scalar::veto", nbOcts,
      KOKKOS_LAMBDA( uint32_t iOct )
    {
      if( markers(iOct) == RefineCondition::REFINE && oct_frac_max(iOct) <= threshold_loc )
        markers(iOct) = RefineCondition::NOCHANGE;
    });

    RefineCondition_utils::set_markers( pmesh, markers );
  }

private:
  enum VarIndex { ISCALAR, IRHO };

  ForeachCell& foreach_cell;
  real_t threshold;
  int scalar_index;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_passive_scalar, "RefineCondition_passive_scalar" );

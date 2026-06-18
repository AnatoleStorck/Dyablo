#include "refine_condition/RefineCondition_helper.h"
#include "refine_condition/RefineCondition.h" // RefineConditionFactory::init() specialization (needed to instanciate sub-conditions)

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {

/**
 * @brief Combines several refine conditions into a single one.
 *
 * Each sub-condition is run in turn and its per-octant markers are merged with
 * an "OR for refinement" rule : an octant is refined if *any* sub-condition asks
 * for refinement, and only coarsened when *all* sub-conditions agree to coarsen.
 * This is achieved by taking the element-wise maximum of the markers produced by
 * each condition (REFINE=1 > NOCHANGE=0 > COARSEN=-1).
 *
 * The list of sub-conditions is read from the same `amr/markers_kernel` key as the
 * single-condition case, e.g. :
 *   markers_kernel = RefineCondition_mass,RefineCondition_second_derivative_error
 */
class RefineCondition_multiple : public RefineCondition
{
public:
  RefineCondition_multiple( ConfigMap& configMap,
                            ForeachCell& foreach_cell,
                            Timers& timers )
    : foreach_cell(foreach_cell)
  {
    std::vector<std::string> condition_ids =
      configMap.getValue<std::vector<std::string>>("amr", "markers_kernel",
        {"RefineCondition_second_derivative_error"});

    for( const std::string& condition_id : condition_ids )
    {
      DYABLO_ASSERT_HOST_RELEASE( condition_id != "RefineCondition_multiple",
        "RefineCondition_multiple cannot list itself in amr/markers_kernel" );
      conditions.push_back( RefineConditionFactory::make_instance( condition_id,
        configMap,
        foreach_cell,
        timers
      ));
    }

    bool has_additive = false;
    for( const auto& condition : conditions )
      if( !condition->is_refinement_mask() )
        has_additive = true;
    DYABLO_ASSERT_HOST_RELEASE( has_additive,
      "amr/markers_kernel must contain at least one non-mask refine condition "
      "(mask conditions such as RefineCondition_passive_scalar only veto refinement)" );
  }

  void mark_cells( UserData& U, ScalarSimulationData& scalar_data ) override
  {
    AMRmesh& pmesh = foreach_cell.get_amr_mesh();
    uint32_t nbOcts = pmesh.getNumOctants();

    // Accumulator initialized to the lowest marker so that max() merging works.
    Kokkos::View<int*> combined_markers("RefineCondition_multiple::markers", nbOcts);
    Kokkos::deep_copy( combined_markers, (int)RefineCondition::COARSEN );

    for( auto& condition : conditions )
    {
      if( condition->is_refinement_mask() )
        continue;

      // Each condition writes its markers into the mesh ; read them back and merge.
      condition->mark_cells( U, scalar_data );
      Kokkos::View<int*> markers = pmesh.getMarkers();

      Kokkos::parallel_for( "RefineCondition_multiple::merge", nbOcts,
        KOKKOS_LAMBDA( uint32_t iOct )
      {
        if( markers(iOct) > combined_markers(iOct) )
          combined_markers(iOct) = markers(iOct);
      });
    }

    RefineCondition_utils::set_markers( pmesh, combined_markers );

    // Mask pass
    for( auto& condition : conditions )
    {
      if( condition->is_refinement_mask() )
        condition->mark_cells( U, scalar_data );
    }
  }

private:
  ForeachCell& foreach_cell;
  std::vector<std::unique_ptr<RefineCondition>> conditions;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_multiple, "RefineCondition_multiple" );

#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "states/State_hydro.h"

namespace dyablo {

/**
 * @brief Refines the mesh wherever the local Jeans length is under-resolved.
 *
 * The Jeans length is
 *      lambda_J = sqrt( pi * c_s^2 / (G * rho) ),
 * with the sound speed
 *      c_s = sqrt( gamma * k_B * T / m_p ).
 * The EOS is p = rho * k_B * T / m_p, so
 *      c_s^2 = gamma * k_B * T / m_p = gamma * p / rho,
 *
 * Resolution is measured as the number of cells per Jeans length
 * N = lambda_J / dx. A cell is :
 *   - refined    if N <= amr/jeans_refine    (default 4 : "cell size is at least a
 *                                           quarter of the local Jeans length")
 *   - coarsened if N >  amr/jeans_coarsen  (default 8)
 *   - left unchanged otherwise.
 * jeans_coarsen >= 2*jeans_refine for now in case of refine/coarsen oscillations.
 * (to be checked)
 */
class RefineCondition_jeans_formula
{
public:
  struct Params
  {
    Params( ConfigMap& configMap )
    : gamma0( configMap.getValue<real_t>("hydro", "gamma0", 1.4) ),
      four_pi_G( configMap.getValue<real_t>("gravity", "4_Pi_G") ),
      jeans_refine( configMap.getValue<real_t>("amr", "jeans_refine", 4.0) ),
      jeans_coarsen( configMap.getValue<real_t>("amr", "jeans_coarsen", 8.0) )
    {}

    real_t gamma0;
    real_t four_pi_G;
    real_t jeans_refine;    // refine   when the Jeans length spans <= this many cells
    real_t jeans_coarsen;  // coarsen when the Jeans length spans >  this many cells
  };

  RefineCondition_jeans_formula( const Params& params, const UserData& U, ScalarSimulationData& /*scalar_data*/ )
    : gamma0( params.gamma0 ),
      jeans_refine( params.jeans_refine ),
      jeans_coarsen( params.jeans_coarsen ),
      four_pi_G( params.four_pi_G ),
      Uin( U.getAccessor( ConsHydroState::getFieldsInfo() ) )
  {}

  template< int ndim >
  KOKKOS_INLINE_FUNCTION
  int getMarker( const ForeachCell::CellIndex& iCell, const ForeachCell::CellMetaData& cells ) const
  {
    ConsHydroState uLoc{};
    getConservativeState<ndim>( Uin, iCell, uLoc );
    PrimHydroState qLoc = consToPrim<ndim>( uLoc, gamma0 );

    real_t rho = qLoc.rho;
    real_t p   = qLoc.p;
    real_t cs2 = gamma0 * p / rho;

    // Compare the squares to avoid sqrt
    // lambda_J^2 = pi * c_s^2 / (G * rho)
    real_t lambdaJ2 = 4 * cs2 / ( four_pi_G * rho );

    auto size = cells.getCellSize(iCell);
    // TODO: generalize to non-cubic cells (dx != dy != dz)
    DYABLO_ASSERT_KOKKOS_DEBUG( size[IX] == size[IY] && (ndim == 2 || size[IX] == size[IZ]),
      "RefineCondition_jeans only supports cubic cells, but cell size is " << size );
    // Assume octree cells are cubic : dx == dy == dz
    real_t dx = size[IX];

    real_t refine_len  = jeans_refine  * dx;
    real_t coarsen_len = jeans_coarsen * dx;

    int criterion;
    if( lambdaJ2 <= refine_len * refine_len )
      criterion = RefineCondition::REFINE;
    else if( lambdaJ2 > coarsen_len * coarsen_len )
      criterion = RefineCondition::COARSEN;
    else
      criterion = RefineCondition::NOCHANGE;

    return criterion;
  }

private:
  real_t gamma0;
  real_t jeans_refine, jeans_coarsen;
  real_t four_pi_G;
  UserData::FieldAccessor Uin;
};

/// Alias for template specialization
class RefineCondition_jeans
  : public RefineCondition_helper<RefineCondition_jeans_formula>
{
public:
  using RefineCondition_helper<RefineCondition_jeans_formula>::RefineCondition_helper;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_jeans, "RefineCondition_jeans" );

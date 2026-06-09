#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ParticleUpdate.h"
#include "utils/config/named_enum.h"

namespace dyablo {

/// Which density field(s) are used to compute the cell mass for refinement
enum MassRefineType {
  MASS_REFINE_GAS,        //!< gas density only ("rho")
  MASS_REFINE_PARTICLES,  //!< particle-deposited density only ("rho_g" - "rho")
  MASS_REFINE_ALL,        //!< gas + particles ("rho_g")
};

} // namespace dyablo

template<>
inline named_enum<dyablo::MassRefineType>::init_list named_enum<dyablo::MassRefineType>::names()
{
  return {
    {dyablo::MassRefineType::MASS_REFINE_GAS,       "gas"},
    {dyablo::MassRefineType::MASS_REFINE_PARTICLES, "particles"},
    {dyablo::MassRefineType::MASS_REFINE_ALL,       "all"},
  };
}

namespace dyablo {

class RefineCondition_mass_formula
{
public:
  // "rho" holds the gas density, "rho_g" holds gas + particle-deposited density
  // (the particle density update copies "rho" into "rho_g" before projecting particles).
  enum VarIndex { Irho, Irho_g };

  struct Params
  {
    Params( ConfigMap& configMap )
    : mass_coarsen( configMap.getValue<real_t>("amr", "mass_coarsen", 0.1)),
      mass_refine( configMap.getValue<real_t>("amr", "mass_refine", 1.0)),
      mass_type( configMap.getValue<MassRefineType>("amr", "mass_type", MASS_REFINE_ALL))
    {}

    real_t mass_coarsen, mass_refine;
    MassRefineType mass_type;
  };

  RefineCondition_mass_formula( const Params& params, const UserData& U, ScalarSimulationData& scalar_data )
    : mass_coarsen( params.mass_coarsen ),
      mass_refine( params.mass_refine ),
      mass_type( params.mass_type ),
      Uin( U.getAccessor( get_fields_info( params.mass_type ) ) )
  {}

  template< int ndim >
  KOKKOS_INLINE_FUNCTION
  int getMarker( const ForeachCell::CellIndex& iCell, const ForeachCell::CellMetaData& cells ) const
  {
    auto size = cells.getCellSize(iCell);
    real_t volume = size[IX]*size[IY]*(ndim == 3 ? size[IZ] : 1.0);

    real_t density = 0.0;
    switch( mass_type )
    {
      case MASS_REFINE_GAS:       density = Uin.at(iCell, Irho); break;
      case MASS_REFINE_PARTICLES: density = Uin.at(iCell, Irho_g) - Uin.at(iCell, Irho); break;
      case MASS_REFINE_ALL:       density = Uin.at(iCell, Irho_g); break;
    }
    real_t local_mass = density * volume;

    int criterion;
    if( local_mass > mass_refine )
      criterion = RefineCondition::REFINE;
    else if( local_mass <= mass_coarsen )
      criterion = RefineCondition::COARSEN;
    else
      criterion = RefineCondition::NOCHANGE;

    return criterion;
  }

  real_t mass_coarsen, mass_refine;
  MassRefineType mass_type;
  UserData::FieldAccessor Uin;

private:
  /// Fields needed by the accessor for each mass type (only request what getMarker reads)
  static std::vector<UserData::FieldAccessor_FieldInfo> get_fields_info( MassRefineType mass_type )
  {
    switch( mass_type )
    {
      case MASS_REFINE_PARTICLES: return {{"rho", Irho}, {"rho_g", Irho_g}};
      case MASS_REFINE_ALL:       return {{"rho_g", Irho_g}};
      case MASS_REFINE_GAS:
      default:                    return {{"rho", Irho}};
    }
  }
};

/// Alias for template specialization
class RefineCondition_mass
  : public RefineCondition_helper<RefineCondition_mass_formula>
{
protected:
  std::unique_ptr<ParticleUpdate> particle_update_density;

public:
  RefineCondition_mass( ConfigMap& configMap,
                        ForeachCell& foreach_cell,
                        Timers& timers )
  : RefineCondition_helper( configMap, foreach_cell, timers )
  {
    std::string particle_update_density_id = configMap.getValue<std::string>("particles", "update_density", "none");
    particle_update_density = ParticleUpdateFactory::make_instance( particle_update_density_id,
      configMap,
      foreach_cell,
      timers
    );
  }

  void mark_cells( UserData& U, ScalarSimulationData& scalar_data )
  {
    MassRefineType mass_type = refineCondition_formula_params.mass_type;
    bool needs_particles = (mass_type == MASS_REFINE_PARTICLES || mass_type == MASS_REFINE_ALL);

    // For "particles" / "all" the particle-deposited density ("rho_g") must be available.
    // It is a temporary field here : the particle density update fills it with rho + particles.
    bool created_temp_field = false;
    if( needs_particles )
    {
      DYABLO_ASSERT_HOST_RELEASE( particle_update_density,
        "amr/mass_type=" << named_enum<MassRefineType>::to_string(mass_type)
        << " requires particles/update_density to be set (it is 'none')" );

      if( !U.has_field("rho_g") )
      {
        U.new_fields({"rho_g"});
        created_temp_field = true;
      }
      particle_update_density->update( U, scalar_data );
    }

    RefineCondition_helper::mark_cells(U, scalar_data);

    // Only delete the field if it is a temporary one we created here.
    // Never delete a "rho_g" owned elsewhere (e.g. by the time loop).
    if( created_temp_field )
      U.delete_field( "rho_g" );
  }

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_mass, "RefineCondition_mass" );

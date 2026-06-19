#include "Compute_dt_base.h"

#include "utils_hydro.h"
#include "particles/ParticleFamilies.h"

namespace dyablo {


/**
 * @brief Timestep limiter for particles.
 * 
 * This class computes the limitation of the timestep for the particles
 * in the same way as the "traditional" CFL for hydro works. 
*/
class Compute_dt_particle_velocity : public Compute_dt
{
public:
  Compute_dt_particle_velocity( ConfigMap& configMap,
                                ForeachCell& foreach_cell,
                                Timers& timers )
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    cfl( configMap.getValue<real_t>("dt", "particle_cfl", 0.5) ),
    families( getParticleFamilies(configMap) )
  {}

  void compute_dt( UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t dt_local = compute_dt_aux(U);

    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }

  double compute_dt_aux( UserData& U )
  {
    int ndim = foreach_cell.getDim();
    
    enum VarIndex_particle{
      IVX,IVY,IVZ
    };
    
    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    constexpr real_t small_v = 1.0e-10;
    using pos_t = Kokkos::Array<real_t, 3>;

    // Reduce inv_dt over every particle family
    // TODO: Think about tracers, they shouldn't be included
    real_t inv_dt = 0.0;
    for( const std::string& family : families )
    {
      if( !U.has_ParticleArray( family ) ) continue;

      const ForeachParticle::ParticleArray& Ppos = U.getParticleArray( family );
      UserData::ParticleAccessor Pdata = U.getParticleAccessor( family, {{"vx", IVX},{"vy", IVY},{"vz", IVZ}} );

      real_t inv_dt_family;
      foreach_particle.reduce_particle( "compute_dt", Pdata.getShape(),
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex &iPart, real_t& inv_dt_update )
      {
        pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
        ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

        pos_t cell_size = cells.getCellSize( iCell );

        real_t dx = cell_size[IX];
        real_t dy = cell_size[IY];
        real_t dz = cell_size[IZ];

        real_t vx = small_v + Pdata.at( iPart, IVX );
        real_t vy = small_v + Pdata.at( iPart, IVY );
        real_t vz = (ndim == 3 ? small_v + Pdata.at( iPart, IVZ ) : 0.0);
        real_t v  = sqrt(vx*vx+vy*vy+vz*vz);
        real_t dmin = FMIN(dx,dy);
        if (ndim == 3)
          dmin = FMIN(dmin, dz);

        inv_dt_update = FMAX( inv_dt_update, v/dmin);
      }, Kokkos::Max<real_t>(inv_dt_family));

      inv_dt = FMAX( inv_dt, inv_dt_family );
    }

    real_t dt = (inv_dt > 0.0) ? cfl / inv_dt : 1.0e10;

    // Temporary fix for when an MPI process has no particles.
    if (dt != dt || dt == 0.0 || dt == -0.0) {
      dt = 1.0e10;
    }

    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);
    return dt;
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;

  real_t cfl;
  std::vector<std::string> families;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_particle_velocity, "Compute_dt_particle_velocity" );

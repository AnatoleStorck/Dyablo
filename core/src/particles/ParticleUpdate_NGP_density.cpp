#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "ParticleFamilies.h"

namespace dyablo {

class ParticleUpdate_NGP_density : public ParticleUpdate {
public:
  ParticleUpdate_NGP_density(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    parr_names( configMap.getValue<std::vector<std::string>>("particles", "density_projected_arrays", getParticleFamilies(configMap)) )
  {}

  ~ParticleUpdate_NGP_density() {}

  void update( UserData& U, ScalarSimulationData& scalar_data)
  {
    int ndim = foreach_cell.getDim();

    timers.get("ParticleUpdate_NGP_density").start();

    enum VarIndex_g{
      IRho, IRhoG
    };
    enum VarIndex_particle{
      IMass
    };

    auto Uin = U.getAccessor( {{"rho", IRho}, {"rho_g", IRhoG}} );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    // Initialize rho_g with the gas density, then accumulate every family.
    foreach_cell.foreach_cell( "ParticleUpdate_NGP_density::copy_density", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      Uin.at(iCell, IRhoG) = Uin.at(iCell, IRho);
    });

    for( const std::string& particle_array_name : parr_names )
    {
      if( !U.has_ParticleArray(particle_array_name) ) continue;

      const ForeachParticle::ParticleArray& Ppos = U.getParticleArray( particle_array_name );
      UserData::ParticleAccessor Pdata = U.getParticleAccessor( particle_array_name, {{"mass", IMass}} );

      foreach_particle.foreach_particle( "ParticleUpdate_NGP_density::projection", Ppos,
        KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
      {
        ForeachCell::CellIndex iCell = cells.getCellFromPos( {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)} );
        auto size = cells.getCellSize( iCell );
        size[IZ] = (ndim == 2) ? 1.0 : size[IZ];
        real_t rho_contrib = Pdata.at( iPart, IMass ) / (size[IX]*size[IY]*size[IZ]);

        Kokkos::atomic_add( &Uin.at( iCell, IRhoG ), rho_contrib ) ;
      });
    }

    timers.get("ParticleUpdate_NGP_density").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;

  std::vector<std::string> parr_names;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory, 
                  dyablo::ParticleUpdate_NGP_density, 
                  "ParticleUpdate_NGP_density")

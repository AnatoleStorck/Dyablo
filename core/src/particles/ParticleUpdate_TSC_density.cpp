#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "ParticleFamilies.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "mpi/GhostCommunicator.h"

namespace dyablo {

class ParticleUpdate_TSC_density : public ParticleUpdate {
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  ParticleUpdate_TSC_density(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    parr_names( configMap.getValue<std::vector<std::string>>("particles", "density_projected_arrays", getParticleFamilies(configMap)) )
  {}

  ~ParticleUpdate_TSC_density() {}

  void update( UserData& U, ScalarSimulationData& scalar_data ) 
  {
    if( foreach_cell.getDim() == 2 )
      update_aux<2>(U, scalar_data);
    else
      update_aux<3>(U, scalar_data);
  }

  template< int ndim>
  void update_aux( UserData& U, ScalarSimulationData& scalar_data ) 
  {
    timers.get("ParticleUpdate_TSC_density").start();

    enum VarIndex_g{
      IRho, IRhoG
    };
    enum VarIndex_particle{
      IMass
    };

    auto Uin = U.getAccessor( {{"rho", IRho}, {"rho_g", IRhoG}} );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    // Initialize rho_g with the gas density, then accumulate every family.
    foreach_cell.foreach_cell( "ParticleUpdate_TSC_density::copy_density", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      Uin.at(iCell, IRhoG) = Uin.at(iCell, IRho);
    });

    for( const std::string& particle_array_name : parr_names )
    {
      if( !U.has_ParticleArray(particle_array_name) ) continue;

      const ForeachParticle::ParticleArray& Ppos = U.getParticleArray( particle_array_name );
      UserData::ParticleAccessor Pdata = U.getParticleAccessor( particle_array_name, {{"mass", IMass}} );

      foreach_particle.foreach_particle( "ParticleUpdate_TSC_density::projection", Ppos,
        KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
      {
      real_t part_mass = Pdata.at( iPart, IMass );
      pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
      ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );
      ForeachCell::SearchMode_neighbor search_neighbor( cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

      pos_t cell_size = cells.getCellSize( iCell );
      cell_size[IZ] = (ndim == 2) ? 1.0 : cell_size[IZ];
      pos_t cell_pos = cells.getCellCenter( iCell );
      real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];

      pos_t p = { // Local position relative to center [-0.5, 0.5]^3
        (part_pos[IX] - cell_pos[IX])/cell_size[IX], 
        (part_pos[IY] - cell_pos[IY])/cell_size[IY], 
        (part_pos[IZ] - cell_pos[IZ])/cell_size[IZ] 
      };
      
      pos_t v_in =  {0.75-p[IX]*p[IX], 0.75-p[IY]*p[IY], 0.75-p[IZ]*p[IZ]};  // volume fraction in local cell [-0.5, 0.5]
      v_in[IZ] = (ndim == 2) ? 1.0 : v_in[IZ];
      pos_t v_out_minus = { 0.5*(0.5 - p[IX])*(0.5 - p[IX]), 
                            0.5*(0.5 - p[IY])*(0.5 - p[IY]),
                            0.5*(0.5 - p[IZ])*(0.5 - p[IZ])};     // volume fraction in neighbor cells [-1.5, -0.5]
      pos_t v_out_plus = {  0.5*(0.5 + p[IX])*(0.5 + p[IX]), 
                            0.5*(0.5 + p[IY])*(0.5 + p[IY]),
                            0.5*(0.5 + p[IZ])*(0.5 + p[IZ])};     // volume fraction in neighbor cells [0.5, 1.5]

      auto apply_rho_contrib = [&]( const ForeachCell::CellIndex::offset_t& offset )
      {
        ForeachCell::CellIndex iCell_neighbor = iCell.getNeighbor(offset, search_neighbor);
        real_t cx = (offset[IX]==0)?v_in[IX]:(offset[IX]==-1)?v_out_minus[IX]: v_out_plus[IX];
        real_t cy = (offset[IY]==0)?v_in[IY]:(offset[IY]==-1)?v_out_minus[IY]: v_out_plus[IY];
        real_t cz = (offset[IZ]==0)?v_in[IZ]:(offset[IZ]==-1)?v_out_minus[IZ]: v_out_plus[IZ];
        real_t volume_fraction = cx*cy*cz;

        assert( abs( iCell_neighbor.level_diff() ) <= 1 );

        if( iCell_neighbor.level_diff() >= 0 )
        { // bigger or same size : only one cell to write
          real_t rho_contrib = (part_mass * volume_fraction) / Vcell;
          if( iCell_neighbor.level_diff() == 1 )
          { 
            // Same volume fraction, but Vcell_neighbor is 2^ndim times bigger
            rho_contrib = rho_contrib/( 2*2*(ndim-1) );
          }         
          Kokkos::atomic_add( &Uin.at( iCell_neighbor, IRhoG ), rho_contrib) ;
        }
        else 
        {
          // Smaller : write to all smaller neighbors
          //real_t volume_fraction_smaller = volume_fraction/(2*(ndim-1)); // Mass distributed accross neighbors
          //real_t Vcell_smaller = Vcell/( 2*2*(ndim-1) ) // Volume 2^ndim times smaller
          //real_t rho_contrib_smaller = (part_mass * volume_fraction_smaller) / Vcell_smaller;
          int di_count = (offset[IX]==0)?2:1;
          int dj_count = (offset[IY]==0)?2:1;
          int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
          real_t rho_contrib = 2*2*(ndim - 1)/(di_count*dj_count*dk_count) * (part_mass * volume_fraction) / Vcell;
          foreach_smaller_neighbor<ndim>( iCell_neighbor, offset, search_neighbor,
            [&]( const ForeachCell::CellIndex& iCell_sn )
          {
            Kokkos::atomic_add( &Uin.at( iCell_sn, IRhoG ), rho_contrib) ;
          });
        }
      };

      for (int16_t ix=-1; ix<=1; ix++)
      for (int16_t iy=-1; iy<=1; iy++)
      for (int16_t iz=(ndim==3)?-1:0; iz<=(ndim==3)?1:0; iz++)
        apply_rho_contrib( {ix, iy, iz} );

      });
    }

    GhostCommunicator ghost_communicator( foreach_cell.get_amr_mesh(), U.getShape(), 1 );
    auto Urhog = U.getAccessor( {{"rho_g", IRhoG}} );
    ghost_communicator.reduce_ghosts(Urhog);

    timers.get("ParticleUpdate_TSC_density").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;

  std::vector<std::string> parr_names;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory, 
                  dyablo::ParticleUpdate_TSC_density, 
                  "ParticleUpdate_TSC_density")

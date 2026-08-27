#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "SinkUtils.h"
#include "foreach_cell/ForeachCell.h"

#include <cmath>

namespace dyablo {

/**
 * @brief Sink particle merging (RAMSES sink_particle.f90 update_sink criterion) :
 * sinks closer than merge_distance_cells * dx_min (default 2 cells at level_max)
 * merge, conserving mass, momentum and center of mass; the smallest id survives.
 *
 * Runs in the [particles] source_term hook, after ParticleUpdate_sink_accretion.
 * All decisions are taken in an identical serial computation on the replicated
 * GlobalSinkTable (chains of close sinks are resolved with union-find), so every
 * rank reaches the same outcome whatever the domain decomposition; each rank then
 * only rewrites its own particles. Absorbed sinks get mass=0 and are removed by
 * compacting the family through a scratch array.
 */
class ParticleUpdate_sink_merging : public ParticleUpdate {
public:
  ParticleUpdate_sink_merging(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    params( SinkParams::from_configMap(configMap, foreach_cell) )
  {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {
    timers.get("ParticleUpdate_sink_merging").start();

    const MpiComm comm = foreach_cell.get_amr_mesh().getMpiComm();
    GlobalSinkTable table = build_global_sink_table( U, foreach_particle, comm, params.family );

    if( table.n_tot < 2 )
    {
      timers.get("ParticleUpdate_sink_merging").stop();
      return;
    }

    const int n = table.n_tot;
    const real_t Lx = params.xmax - params.xmin;
    const real_t Ly = params.ymax - params.ymin;
    const real_t Lz = params.zmax - params.zmin;
    const real_t merge_dist2 = ( params.merge_distance_cells * params.dx_min )
                             * ( params.merge_distance_cells * params.dx_min );

    // -------------------------------------------------------------------
    // Union-find over close pairs : chains of close sinks form one component
    std::vector<int> parent( n );
    for( int i=0; i<n; i++ ) parent[i] = i;
    auto find = [&]( int i ) { while( parent[i] != i ) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };

    for( int i=0; i<n; i++ )
    for( int j=i+1; j<n; j++ )
    {
      const real_t rx = sink_wrapped_delta( table.at(i,GlobalSinkTable::X), table.at(j,GlobalSinkTable::X), Lx, params.periodic[IX] );
      const real_t ry = sink_wrapped_delta( table.at(i,GlobalSinkTable::Y), table.at(j,GlobalSinkTable::Y), Ly, params.periodic[IY] );
      const real_t rz = sink_wrapped_delta( table.at(i,GlobalSinkTable::Z), table.at(j,GlobalSinkTable::Z), Lz, params.periodic[IZ] );
      if( rx*rx + ry*ry + rz*rz < merge_dist2 )
        parent[ find(i) ] = find(j);
    }

    // -------------------------------------------------------------------
    // Resolve each multi-member component into its smallest-id member
    std::vector<int> survivor_of( n );   // row index of the component survivor
    int n_merged = 0;
    for( int i=0; i<n; i++ ) survivor_of[i] = -1;
    for( int i=0; i<n; i++ )
    {
      const int r = find(i);
      if( survivor_of[r] < 0 || table.at(i,GlobalSinkTable::ID) < table.at(survivor_of[r],GlobalSinkTable::ID) )
        survivor_of[r] = i;
    }
    for( int i=0; i<n; i++ )
    {
      survivor_of[i] = survivor_of[ find(i) ];
      if( survivor_of[i] != i ) n_merged++;
    }

    if( n_merged == 0 )
    {
      timers.get("ParticleUpdate_sink_merging").stop();
      return;
    }

    // New mass / center of mass / velocity per row (unchanged for singletons)
    enum ResCol : int { RES_MASS=0, RES_X, RES_Y, RES_Z, RES_VX, RES_VY, RES_VZ, RES_NCOL };
    std::vector<real_t> result( (size_t)n * RES_NCOL );
    for( int i=0; i<n; i++ )
    {
      result[ (size_t)i*RES_NCOL + RES_MASS ] = table.at(i,GlobalSinkTable::MASS);
      result[ (size_t)i*RES_NCOL + RES_X ]  = table.at(i,GlobalSinkTable::X);
      result[ (size_t)i*RES_NCOL + RES_Y ]  = table.at(i,GlobalSinkTable::Y);
      result[ (size_t)i*RES_NCOL + RES_Z ]  = table.at(i,GlobalSinkTable::Z);
      result[ (size_t)i*RES_NCOL + RES_VX ] = table.at(i,GlobalSinkTable::VX);
      result[ (size_t)i*RES_NCOL + RES_VY ] = table.at(i,GlobalSinkTable::VY);
      result[ (size_t)i*RES_NCOL + RES_VZ ] = table.at(i,GlobalSinkTable::VZ);
    }
    for( int s=0; s<n; s++ )
    {
      if( survivor_of[s] != s ) // absorbed
      {
        result[ (size_t)s*RES_NCOL + RES_MASS ] = 0;
        continue;
      }
      // survivor : sum its component (deltas taken minimum-image relative to the survivor)
      real_t M = 0, dx = 0, dy = 0, dz = 0, px = 0, py = 0, pz = 0;
      const real_t xs = table.at(s,GlobalSinkTable::X);
      const real_t ys = table.at(s,GlobalSinkTable::Y);
      const real_t zs = table.at(s,GlobalSinkTable::Z);
      for( int i=0; i<n; i++ )
      {
        if( survivor_of[i] != s ) continue;
        const real_t m = table.at(i,GlobalSinkTable::MASS);
        M  += m;
        dx += m * sink_wrapped_delta( table.at(i,GlobalSinkTable::X), xs, Lx, params.periodic[IX] );
        dy += m * sink_wrapped_delta( table.at(i,GlobalSinkTable::Y), ys, Ly, params.periodic[IY] );
        dz += m * sink_wrapped_delta( table.at(i,GlobalSinkTable::Z), zs, Lz, params.periodic[IZ] );
        px += m * table.at(i,GlobalSinkTable::VX);
        py += m * table.at(i,GlobalSinkTable::VY);
        pz += m * table.at(i,GlobalSinkTable::VZ);
      }
      result[ (size_t)s*RES_NCOL + RES_MASS ] = M;
      result[ (size_t)s*RES_NCOL + RES_X ]  = fmod( ( xs + dx/M - params.xmin ) + Lx, Lx ) + params.xmin;
      result[ (size_t)s*RES_NCOL + RES_Y ]  = fmod( ( ys + dy/M - params.ymin ) + Ly, Ly ) + params.ymin;
      result[ (size_t)s*RES_NCOL + RES_Z ]  = fmod( ( zs + dz/M - params.zmin ) + Lz, Lz ) + params.zmin;
      result[ (size_t)s*RES_NCOL + RES_VX ] = px / M;
      result[ (size_t)s*RES_NCOL + RES_VY ] = py / M;
      result[ (size_t)s*RES_NCOL + RES_VZ ] = pz / M;
    }

    // -------------------------------------------------------------------
    // Apply : local particles are exactly table rows [offset, offset+n_loc)
    if( table.n_loc > 0 )
    {
      Kokkos::View<real_t**> result_d( "sink_merge_result", table.n_loc, (int)RES_NCOL );
      auto result_h = Kokkos::create_mirror_view( result_d );
      for( int i=0; i<table.n_loc; i++ )
        for( int c=0; c<(int)RES_NCOL; c++ )
          result_h(i,c) = result[ (size_t)(table.offset + i)*RES_NCOL + c ];
      Kokkos::deep_copy( result_d, result_h );

      enum VarIndex_particle{ IMASS, IVX, IVY, IVZ };
      auto Ppos = U.getParticleArray( params.family );
      auto Pdata = U.getParticleAccessor( params.family,
        { {"mass",IMASS}, {"vx",IVX}, {"vy",IVY}, {"vz",IVZ} } );

      foreach_particle.foreach_particle( "sink_merging_apply", Ppos,
        KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
      {
        Pdata.at(iPart, IMASS) = result_d(iPart, RES_MASS);
        Pdata.at(iPart, IVX)   = result_d(iPart, RES_VX);
        Pdata.at(iPart, IVY)   = result_d(iPart, RES_VY);
        Pdata.at(iPart, IVZ)   = result_d(iPart, RES_VZ);
        Ppos.pos(iPart, IX) = result_d(iPart, RES_X);
        Ppos.pos(iPart, IY) = result_d(iPart, RES_Y);
        Ppos.pos(iPart, IZ) = result_d(iPart, RES_Z);
      });
    }

    // -------------------------------------------------------------------
    // Compaction : drop the absorbed (mass=0) sinks
    const std::string tmp = params.family + "_compact";
    U.new_ParticleArray( tmp, 0 );
    for( const std::string& attr : U.getEnabledParticleAttributes( params.family ) )
      U.new_ParticleAttribute( tmp, attr );
    U.merge_particles_if( tmp, params.family, "mass" );   // deletes the old family array
    U.move_ParticleArray( params.family, tmp );

    // Survivors moved to their component's center of mass, possibly across ranks
    U.distributeParticles( params.family );

    if( comm.MPI_Comm_rank() == 0 )
      std::cout << "Sink merging : " << n_merged << " sink(s) absorbed, "
                << ( n - n_merged ) << " remaining" << std::endl;

    timers.get("ParticleUpdate_sink_merging").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;
  SinkParams params;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_sink_merging,
                  "ParticleUpdate_sink_merging")

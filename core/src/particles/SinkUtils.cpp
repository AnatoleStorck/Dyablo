#include "ParticleUpdate_base.h"

#include "SinkUtils.h"
#include "user_data/ParticleAccessor.h"

namespace dyablo {

/***
 * Single definition of the sink table builder : the Kokkos kernel below must be
 * compiled in exactly one translation unit (see the note in SinkUtils.h).
 ***/
GlobalSinkTable build_global_sink_table(
    const UserData& U, const ForeachParticle& foreach_particle,
    const MpiComm& comm, const std::string& family )
{
  GlobalSinkTable table;

  int n_loc = 0;
  if( U.has_ParticleArray(family) )
    n_loc = U.getParticleArray(family).getNumParticles();

  int n_incl = 0;
  comm.MPI_Scan( &n_loc, &n_incl, 1, MpiComm::MPI_Op_t::SUM );
  int n_tot = 0;
  comm.MPI_Allreduce( &n_loc, &n_tot, 1, MpiComm::MPI_Op_t::SUM );

  table.n_loc = n_loc;
  table.offset = n_incl - n_loc;
  table.n_tot = n_tot;
  table.host.assign( (size_t)n_tot * GlobalSinkTable::NCOL, 0.0 );
  table.device = Kokkos::View<real_t**>( "sink_table", n_tot, (int)GlobalSinkTable::NCOL );

  if( n_tot == 0 )
    return table;

  if( n_loc > 0 )
  {
    enum VarIndex_sink{ IMASS, IVX, IVY, IVZ, IID };
    auto Ppos = U.getParticleArray( family );
    auto Pdata = U.getParticleAccessor( family,
      { {"mass",IMASS}, {"vx",IVX}, {"vy",IVY}, {"vz",IVZ}, {"id",IID} } );

    Kokkos::View<real_t**> pack( "sink_table_pack", n_loc, (int)GlobalSinkTable::NCOL );
    foreach_particle.foreach_particle( "sink_table_pack", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      pack(iPart, GlobalSinkTable::ID)   = Pdata.at(iPart, IID);
      pack(iPart, GlobalSinkTable::MASS) = Pdata.at(iPart, IMASS);
      pack(iPart, GlobalSinkTable::X)    = Ppos.pos(iPart, IX);
      pack(iPart, GlobalSinkTable::Y)    = Ppos.pos(iPart, IY);
      pack(iPart, GlobalSinkTable::Z)    = Ppos.pos(iPart, IZ);
      pack(iPart, GlobalSinkTable::VX)   = Pdata.at(iPart, IVX);
      pack(iPart, GlobalSinkTable::VY)   = Pdata.at(iPart, IVY);
      pack(iPart, GlobalSinkTable::VZ)   = Pdata.at(iPart, IVZ);
    });

    auto pack_host = Kokkos::create_mirror_view( pack );
    Kokkos::deep_copy( pack_host, pack );
    for( int i=0; i<n_loc; i++ )
      for( int c=0; c<(int)GlobalSinkTable::NCOL; c++ )
        table.host[ (size_t)(table.offset + i)*GlobalSinkTable::NCOL + c ] = pack_host(i,c);
  }

  comm.MPI_Allgatherv_inplace( table.host.data(), n_loc * (int)GlobalSinkTable::NCOL );

  auto device_host = Kokkos::create_mirror_view( table.device );
  for( int i=0; i<n_tot; i++ )
    for( int c=0; c<(int)GlobalSinkTable::NCOL; c++ )
      device_host(i,c) = table.host[ (size_t)i*GlobalSinkTable::NCOL + c ];
  Kokkos::deep_copy( table.device, device_host );

  return table;
}

} // namespace dyablo

#include "gtest/gtest.h"

#include <cmath>

#include "amr/AMRmesh.h"
#include "particles/ForeachParticle.h"
#include "particles/ParticleUpdate.h"
#include "particles/SinkUtils.h"

namespace dyablo
{

namespace {

using pos_t = Kokkos::Array<real_t, 3>;

/// Uniform periodic 3D mesh at level `level` (8^3 blocks) + the sink test config.
struct SinkTestSetup
{
  std::shared_ptr<AMRmesh> amr_mesh;
  std::unique_ptr<ConfigMap> configMap;
  std::unique_ptr<ForeachCell> foreach_cell;
  std::unique_ptr<UserData> U;
  Timers timers;

  SinkTestSetup( int level, const std::string& extra_config = "" )
  {
    amr_mesh = std::make_shared<AMRmesh>( 3, std::array<bool,3>{true,true,true}, level, level );
    amr_mesh->loadBalance();

    std::string config_str =
      "[amr]\n"
      "use_block_data=true\n"
      "bx=8\n"
      "by=8\n"
      "bz=8\n"
      "level_max=" + std::to_string(level) + "\n"
      "[mesh]\n"
      "ndim=3\n"
      "[hydro]\n"
      "gamma0=1.4\n"
      "[gravity]\n"
      "4_Pi_G=12.566370614359172\n"     // G = 1 in code units
      "[sink]\n"
      "rho_sink=2.0\n"
      "mass_sink_seed=0.0\n"
      "ir_cloud=4\n"
      "c_acc=0.75\n"
      "check_energies=false\n"          // no gravity solver in the tests : Jeans-proxy gate
      + extra_config;
    configMap = std::make_unique<ConfigMap>(config_str);

    foreach_cell = std::make_unique<ForeachCell>( *amr_mesh, *configMap );
    U = std::make_unique<UserData>( *configMap, *foreach_cell );
  }
};

constexpr const char* sink_attributes[] = { "mass", "vx", "vy", "vz", "id", "birth_time", "dm_acc" };

void create_sink_family( UserData& U, const std::string& family, uint32_t n_loc )
{
  U.new_ParticleArray( family, n_loc );
  for( const char* attr : sink_attributes )
    U.new_ParticleAttribute( family, attr );
}

enum VarIndex_hydro{ IRho, IE_tot, IE_int, IRho_vx, IRho_vy, IRho_vz };

UserData::FieldAccessor hydro_accessor( UserData& U )
{
  U.new_fields( {"rho", "e_tot", "e_int", "rho_vx", "rho_vy", "rho_vz"} );
  return U.getAccessor( { {"rho",IRho}, {"e_tot",IE_tot}, {"e_int",IE_int},
                          {"rho_vx",IRho_vx}, {"rho_vy",IRho_vy}, {"rho_vz",IRho_vz} } );
}

/// Global sums of gas mass and x-momentum
void global_gas_totals( ForeachCell& foreach_cell, const UserData::FieldAccessor& Uin,
                        real_t& mass_tot, real_t& px_tot )
{
  ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
  real_t mass_loc = 0, px_loc = 0;
  foreach_cell.reduce_cell( "gas_totals", Uin.getShape(),
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& m, real_t& px )
  {
    pos_t size = cells.getCellSize( iCell );
    const real_t V = size[IX]*size[IY]*size[IZ];
    m  += Uin.at(iCell, IRho) * V;
    px += Uin.at(iCell, IRho_vx) * V;
  }, mass_loc, px_loc);
  GlobalMpiSession::get_comm_world().MPI_Allreduce(&mass_loc, &mass_tot, 1, MpiComm::MPI_Op_t::SUM);
  GlobalMpiSession::get_comm_world().MPI_Allreduce(&px_loc, &px_tot, 1, MpiComm::MPI_Op_t::SUM);
}

} // anonymous namespace

// ===========================================================================
TEST(Test_ParticleUpdate_sink, move_ParticleArray)
{
  SinkTestSetup s( 2 );
  UserData& U = *s.U;
  ForeachParticle foreach_particle( *s.amr_mesh, *s.configMap );

  const int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
  const uint32_t n_loc = (rank == 0) ? 5 : 0;
  create_sink_family( U, "sinks_tmp", n_loc );

  {
    auto Ppos = U.getParticleArray( "sinks_tmp" );
    auto Pdata = U.getParticleAccessor( "sinks_tmp", {{"mass",0},{"id",1}} );
    foreach_particle.foreach_particle( "init_parts", Ppos,
      PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart )
    {
      Ppos.pos(iPart, IX) = 0.1 + 0.15*iPart;
      Ppos.pos(iPart, IY) = 0.5;
      Ppos.pos(iPart, IZ) = 0.5;
      Pdata.at(iPart, 0) = 1.0 + iPart;
      Pdata.at(iPart, 1) = 10.0 + iPart;
    });
  }

  U.move_ParticleArray( "sinks", "sinks_tmp" );

  EXPECT_FALSE( U.has_ParticleArray("sinks_tmp") );
  ASSERT_TRUE( U.has_ParticleArray("sinks") );
  for( const char* attr : sink_attributes )
    EXPECT_TRUE( U.has_ParticleAttribute("sinks", attr) );

  // Renamed array must still distribute correctly across ranks
  U.distributeParticles( "sinks" );

  int n_tot = 0, n_new = U.getParticleArray("sinks").getNumParticles();
  GlobalMpiSession::get_comm_world().MPI_Allreduce(&n_new, &n_tot, 1, MpiComm::MPI_Op_t::SUM);
  EXPECT_EQ( n_tot, 5 );

  // Attribute data survived the rename+distribute : mass sum = 1+2+3+4+5
  real_t mass_loc = 0, mass_tot = 0;
  {
    auto Ppos = U.getParticleArray( "sinks" );
    auto Pdata = U.getParticleAccessor( "sinks", {{"mass",0}} );
    foreach_particle.reduce_particle( "sum_mass", Ppos,
      PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart, real_t& sum )
    {
      sum += Pdata.at(iPart, 0);
    }, mass_loc);
  }
  GlobalMpiSession::get_comm_world().MPI_Allreduce(&mass_loc, &mass_tot, 1, MpiComm::MPI_Op_t::SUM);
  EXPECT_NEAR( mass_tot, 15.0, 1e-12 );
}

// ===========================================================================
TEST(Test_ParticleUpdate_sink, GlobalSinkTable_replicated)
{
  SinkTestSetup s( 2 );
  UserData& U = *s.U;
  ForeachParticle foreach_particle( *s.amr_mesh, *s.configMap );
  const MpiComm comm = s.amr_mesh->getMpiComm();
  const int rank = comm.MPI_Comm_rank();
  const int size = comm.MPI_Comm_size();

  // 2 sinks per rank, attributes derived from the global index
  const uint32_t n_loc = 2;
  create_sink_family( U, "sinks", n_loc );
  {
    auto Ppos = U.getParticleArray( "sinks" );
    auto Pdata = U.getParticleAccessor( "sinks", {{"mass",0},{"id",1},{"vx",2}} );
    foreach_particle.foreach_particle( "init_parts", Ppos,
      PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart )
    {
      const int gid = 2*rank + iPart;
      Ppos.pos(iPart, IX) = 0.25 + 0.5*(iPart%2);
      Ppos.pos(iPart, IY) = 0.25;
      Ppos.pos(iPart, IZ) = 0.25;
      Pdata.at(iPart, 0) = 1.0 + gid;
      Pdata.at(iPart, 1) = 100.0 + gid;
      Pdata.at(iPart, 2) = 0.01*gid;
    });
  }

  GlobalSinkTable table = build_global_sink_table( U, foreach_particle, comm, "sinks" );

  EXPECT_EQ( table.n_tot, 2*size );
  EXPECT_EQ( table.n_loc, 2 );
  EXPECT_EQ( table.offset, 2*rank );

  // Rows [offset, offset+n_loc) hold this rank's particles in iPart order
  for( int i=0; i<table.n_loc; i++ )
  {
    const int gid = 2*rank + i;
    EXPECT_NEAR( table.at( table.offset+i, GlobalSinkTable::MASS ), 1.0 + gid, 1e-14 );
    EXPECT_NEAR( table.at( table.offset+i, GlobalSinkTable::ID ), 100.0 + gid, 1e-14 );
  }
  // ... and the full table is identical on every rank : the global mass sum matches
  real_t mass_sum = 0;
  for( int i=0; i<table.n_tot; i++ )
    mass_sum += table.at( i, GlobalSinkTable::MASS );
  real_t expected = 0;
  for( int gid=0; gid<2*size; gid++ )
    expected += 1.0 + gid;
  EXPECT_NEAR( mass_sum, expected, 1e-12 );
}

// ===========================================================================
TEST(Test_ParticleUpdate_sink, formation_single_peak)
{
  const int level = 2;                       // uniform 32^3 cells
  SinkTestSetup s( level );
  UserData& U = *s.U;
  ForeachCell& foreach_cell = *s.foreach_cell;
  ForeachParticle foreach_particle( *s.amr_mesh, *s.configMap );
  const MpiComm comm = s.amr_mesh->getMpiComm();

  auto Uin = hydro_accessor( U );
  ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

  const real_t dx = 1.0/32;
  const real_t gamma0 = 1.4;
  const real_t P0 = 1e-4;                    // low pressure : Jeans length unresolved at the peak
  const pos_t peak = { 16.5*dx, 16.5*dx, 16.5*dx };   // a cell center
  const real_t sigma2 = 2.0 * (2*dx)*(2*dx);

  // Gaussian density peak (max 10, background 1) in a converging flow v = -0.05*(r-peak)
  foreach_cell.foreach_cell( "init_hydro", Uin.getShape(),
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    const pos_t c = cells.getCellCenter( iCell );
    const real_t r2 = (c[IX]-peak[IX])*(c[IX]-peak[IX])
                    + (c[IY]-peak[IY])*(c[IY]-peak[IY])
                    + (c[IZ]-peak[IZ])*(c[IZ]-peak[IZ]);
    const real_t rho = 1.0 + 9.0 * Kokkos::exp( -r2 / sigma2 );
    const real_t u = -0.05*(c[IX]-peak[IX]);
    const real_t v = -0.05*(c[IY]-peak[IY]);
    const real_t w = -0.05*(c[IZ]-peak[IZ]);
    Uin.at(iCell, IRho) = rho;
    Uin.at(iCell, IRho_vx) = rho*u;
    Uin.at(iCell, IRho_vy) = rho*v;
    Uin.at(iCell, IRho_vz) = rho*w;
    Uin.at(iCell, IE_int) = P0/(gamma0-1);
    Uin.at(iCell, IE_tot) = P0/(gamma0-1) + 0.5*rho*(u*u+v*v+w*w);
  });

  real_t gas_mass_before = 0, gas_px_before = 0;
  global_gas_totals( foreach_cell, Uin, gas_mass_before, gas_px_before );

  Timers timers;
  auto formation = ParticleUpdateFactory::make_instance( "ParticleUpdate_sink_formation",
    *s.configMap, foreach_cell, timers );

  ScalarSimulationData scalar_data;
  scalar_data.set<real_t>( "time", 0.0 );
  scalar_data.set<real_t>( "dt", 1e-3 );
  formation->update( U, scalar_data );

  // Exactly one sink, at the peak cell, with the expected seed mass
  GlobalSinkTable table = build_global_sink_table( U, foreach_particle, comm, "sinks" );
  ASSERT_EQ( table.n_tot, 1 );
  EXPECT_NEAR( table.at(0, GlobalSinkTable::X), peak[IX], 1e-12 );
  EXPECT_NEAR( table.at(0, GlobalSinkTable::Y), peak[IY], 1e-12 );
  EXPECT_NEAR( table.at(0, GlobalSinkTable::Z), peak[IZ], 1e-12 );
  EXPECT_NEAR( table.at(0, GlobalSinkTable::ID), 1.0, 1e-14 );
  const real_t expected_seed = 0.75 * (10.0 - 2.0) * dx*dx*dx;
  EXPECT_NEAR( table.at(0, GlobalSinkTable::MASS), expected_seed, 1e-12*expected_seed );
  // seed velocity = peak cell gas velocity = 0 at the peak
  EXPECT_NEAR( table.at(0, GlobalSinkTable::VX), 0.0, 1e-12 );

  // Gas + sink mass is conserved
  real_t gas_mass_after = 0, gas_px_after = 0;
  global_gas_totals( foreach_cell, Uin, gas_mass_after, gas_px_after );
  EXPECT_NEAR( gas_mass_after + table.at(0, GlobalSinkTable::MASS), gas_mass_before, 1e-12*gas_mass_before );

  // A second call forms nothing : the whole neighborhood is within 2*r_acc of the sink
  formation->update( U, scalar_data );
  GlobalSinkTable table2 = build_global_sink_table( U, foreach_particle, comm, "sinks" );
  EXPECT_EQ( table2.n_tot, 1 );
}

// ===========================================================================
TEST(Test_ParticleUpdate_sink, accretion_uniform)
{
  const int level = 2;
  SinkTestSetup s( level );
  UserData& U = *s.U;
  ForeachCell& foreach_cell = *s.foreach_cell;
  ForeachParticle foreach_particle( *s.amr_mesh, *s.configMap );
  const MpiComm comm = s.amr_mesh->getMpiComm();
  const int rank = comm.MPI_Comm_rank();

  auto Uin = hydro_accessor( U );

  const real_t dx = 1.0/32;
  const real_t gamma0 = 1.4;
  const real_t rho0 = 4.0, P0 = 1e-4, v0 = 0.1;   // uniform gas moving in +x

  foreach_cell.foreach_cell( "init_hydro", Uin.getShape(),
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    Uin.at(iCell, IRho) = rho0;
    Uin.at(iCell, IRho_vx) = rho0*v0;
    Uin.at(iCell, IRho_vy) = 0;
    Uin.at(iCell, IRho_vz) = 0;
    Uin.at(iCell, IE_int) = P0/(gamma0-1);
    Uin.at(iCell, IE_tot) = P0/(gamma0-1) + 0.5*rho0*v0*v0;
  });

  // One sink of mass 1 at rest, at a cell center
  const real_t m0 = 1.0;
  const pos_t sink_pos = { 16.5*dx, 16.5*dx, 16.5*dx };
  create_sink_family( U, "sinks", (rank==0) ? 1u : 0u );
  {
    auto Ppos = U.getParticleArray( "sinks" );
    auto Pdata = U.getParticleAccessor( "sinks", {{"mass",0},{"id",1}} );
    foreach_particle.foreach_particle( "init_sink", Ppos,
      PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart )
    {
      Ppos.pos(iPart, IX) = sink_pos[IX];
      Ppos.pos(iPart, IY) = sink_pos[IY];
      Ppos.pos(iPart, IZ) = sink_pos[IZ];
      Pdata.at(iPart, 0) = m0;
      Pdata.at(iPart, 1) = 1.0;
    });
  }
  U.distributeParticles( "sinks" );

  real_t gas_mass_before = 0, gas_px_before = 0;
  global_gas_totals( foreach_cell, Uin, gas_mass_before, gas_px_before );

  Timers timers;
  auto accretion = ParticleUpdateFactory::make_instance( "ParticleUpdate_sink_accretion",
    *s.configMap, foreach_cell, timers );

  ScalarSimulationData scalar_data;
  scalar_data.set<real_t>( "time", 0.0 );
  scalar_data.set<real_t>( "dt", 1e-3 );
  accretion->update( U, scalar_data );

  // Analytic expectation : every slot of the uniform sphere contributes
  // c_acc * (rho0 - rho_sink) * dx^3
  int n_slots = 0;
  for( int dk=-4; dk<=4; dk++ )
  for( int dj=-4; dj<=4; dj++ )
  for( int di=-4; di<=4; di++ )
    if( di*di + dj*dj + dk*dk <= 16 ) n_slots++;
  const real_t expected_dM = 0.75 * (rho0 - 2.0) * n_slots * dx*dx*dx;

  GlobalSinkTable table = build_global_sink_table( U, foreach_particle, comm, "sinks" );
  ASSERT_EQ( table.n_tot, 1 );
  EXPECT_NEAR( table.at(0, GlobalSinkTable::MASS), m0 + expected_dM, 1e-12*(m0+expected_dM) );
  // momentum conservation : the sink picked up the accreted gas momentum
  EXPECT_NEAR( table.at(0, GlobalSinkTable::VX), expected_dM*v0/(m0+expected_dM), 1e-12 );
  EXPECT_NEAR( table.at(0, GlobalSinkTable::VY), 0.0, 1e-14 );
  // uniform density : no center-of-mass shift
  EXPECT_NEAR( table.at(0, GlobalSinkTable::X), sink_pos[IX], 1e-12 );

  real_t gas_mass_after = 0, gas_px_after = 0;
  global_gas_totals( foreach_cell, Uin, gas_mass_after, gas_px_after );
  EXPECT_NEAR( gas_mass_after + table.at(0, GlobalSinkTable::MASS),
               gas_mass_before + m0, 1e-12*gas_mass_before );
  EXPECT_NEAR( gas_px_after + table.at(0, GlobalSinkTable::MASS)*table.at(0, GlobalSinkTable::VX),
               gas_px_before, 1e-12*fabs(gas_px_before) );
}

// ===========================================================================
TEST(Test_ParticleUpdate_sink, merging_pair)
{
  const int level = 2;
  SinkTestSetup s( level );
  UserData& U = *s.U;
  ForeachCell& foreach_cell = *s.foreach_cell;
  ForeachParticle foreach_particle( *s.amr_mesh, *s.configMap );
  const MpiComm comm = s.amr_mesh->getMpiComm();
  const int rank = comm.MPI_Comm_rank();

  // Sinks 0 and 1 are 0.03 < 2*dx_min = 0.0625 apart : they merge.
  // Sink 2 is far away : it survives untouched.
  const real_t x0 = 0.5, sep = 0.03;
  create_sink_family( U, "sinks", (rank==0) ? 3u : 0u );
  {
    auto Ppos = U.getParticleArray( "sinks" );
    auto Pdata = U.getParticleAccessor( "sinks", {{"mass",0},{"id",1},{"vx",2},{"vy",3}} );
    foreach_particle.foreach_particle( "init_sinks", Ppos,
      PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart )
    {
      if( iPart == 0 )      { Ppos.pos(iPart,IX)=x0;     Pdata.at(iPart,0)=1; Pdata.at(iPart,1)=1; Pdata.at(iPart,2)=1; Pdata.at(iPart,3)=0; }
      else if( iPart == 1 ) { Ppos.pos(iPart,IX)=x0+sep; Pdata.at(iPart,0)=3; Pdata.at(iPart,1)=2; Pdata.at(iPart,2)=0; Pdata.at(iPart,3)=1; }
      else                  { Ppos.pos(iPart,IX)=0.2;    Pdata.at(iPart,0)=5; Pdata.at(iPart,1)=7; Pdata.at(iPart,2)=0; Pdata.at(iPart,3)=0; }
      Ppos.pos(iPart, IY) = (iPart==2) ? 0.2 : 0.5;
      Ppos.pos(iPart, IZ) = (iPart==2) ? 0.2 : 0.5;
    });
  }
  U.distributeParticles( "sinks" );

  Timers timers;
  auto merging = ParticleUpdateFactory::make_instance( "ParticleUpdate_sink_merging",
    *s.configMap, foreach_cell, timers );

  ScalarSimulationData scalar_data;
  merging->update( U, scalar_data );

  GlobalSinkTable table = build_global_sink_table( U, foreach_particle, comm, "sinks" );
  ASSERT_EQ( table.n_tot, 2 );

  int i_merged = ( table.at(0, GlobalSinkTable::ID) == 1.0 ) ? 0 : 1;
  int i_far    = 1 - i_merged;
  ASSERT_EQ( table.at(i_merged, GlobalSinkTable::ID), 1.0 );   // smallest id survives
  ASSERT_EQ( table.at(i_far, GlobalSinkTable::ID), 7.0 );

  // Merged : mass 1+3, center of mass, momentum-conserving velocity
  EXPECT_NEAR( table.at(i_merged, GlobalSinkTable::MASS), 4.0, 1e-12 );
  EXPECT_NEAR( table.at(i_merged, GlobalSinkTable::X), x0 + 0.75*sep, 1e-12 );
  EXPECT_NEAR( table.at(i_merged, GlobalSinkTable::Y), 0.5, 1e-12 );
  EXPECT_NEAR( table.at(i_merged, GlobalSinkTable::VX), 0.25, 1e-12 );
  EXPECT_NEAR( table.at(i_merged, GlobalSinkTable::VY), 0.75, 1e-12 );

  // Far sink untouched
  EXPECT_NEAR( table.at(i_far, GlobalSinkTable::MASS), 5.0, 1e-12 );
  EXPECT_NEAR( table.at(i_far, GlobalSinkTable::X), 0.2, 1e-12 );
}

} // namespace dyablo

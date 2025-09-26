#include "gtest/gtest.h"

#include "amr/AMRmesh.h"
#include "amr/LightOctree.h"
#include "io/IOManager.h"

#include "particles/ForeachParticle.h"
#include "particles/ParticleUpdate.h"

namespace dyablo
{

using pos_t = Kokkos::Array<real_t, 3>;

void run_test(int ndim, std::string particle_update_density_id)
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing Mass Assignment Schemes ..\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;

  const int mpi_rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
  const int mpi_size = GlobalMpiSession::get_comm_world().MPI_Comm_size();

  const int level_min = 2;
  const int level_max = 8;
  const int oct_to_refine = 15; // Refine the 15th octant in the initial grid
  const int nbOcts_tot = 1U<<(ndim*level_min);
  DYABLO_ASSERT_HOST_RELEASE( mpi_size <= nbOcts_tot, "Untested with more tasks than octs"  );
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, std::array<bool,3>{true,true,true}, level_min, level_max );
    std::vector<int> octs_per_rank(mpi_size, 0), octs_sum_per_rank(mpi_size+1, 0);
    octs_per_rank[mpi_rank] = amr_mesh->getNumOctants();
    GlobalMpiSession::get_comm_world().MPI_Allreduce(octs_per_rank.data(), octs_per_rank.data(), mpi_size, MpiComm::MPI_Op_t::SUM);
    std::inclusive_scan(octs_per_rank.begin(), octs_per_rank.end(), octs_sum_per_rank.begin() + 1);
    if (0 == mpi_rank)
      amr_mesh->setMarker(0,1);
    if (octs_sum_per_rank[mpi_rank]<= oct_to_refine && oct_to_refine < octs_sum_per_rank[mpi_rank+1])
      amr_mesh->setMarker(oct_to_refine - octs_sum_per_rank[mpi_rank], 1);

    amr_mesh->adapt();
    amr_mesh->loadBalance();
  }

  //Timers timers;

  std::string outputPrefix = (ndim == 2 ? "2D_" : "3D_") + particle_update_density_id;
  std::string config_str = 
    "[output]\n"
    "hdf5_enabled=true\n"
    "write_mesh_info=true\n"
    "write_variables=rho_g\n"
    "write_iOct=false\n"
    "outputPrefix="+outputPrefix+"\n"
    "outputDir=./\n"
    "[amr]\n"
    "use_block_data=true\n"
    "bx=8\n"
    "by=8\n";
  ConfigMap configMap(config_str);

  configMap.getValue<int>("mesh", "ndim", ndim);
  configMap.getValue<int>("amr", "bz", ndim==2?1:8);
  
  ForeachCell foreach_cell( *amr_mesh, configMap );
  UserData U(configMap, foreach_cell);
  
  enum VarIndex_test{IRHO, IRHOG, IMASS};
  U.new_fields({"rho", "rho_g"}); 
  auto Uin = U.getAccessor( {{"rho", IRHO}, {"rho_g", IRHOG}} );
  const ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();


  // Initialize U
  foreach_cell.foreach_cell( "Init_U", Uin.getShape(),
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    Uin.at(iCell, IRHO) = 0;
    Uin.at(iCell, IRHOG) = 0;
  });

  std::string name = (ndim == 2 ? "2D_" : "3D_") + particle_update_density_id;
  std::string iomanager_id = "IOManager_hdf5";
  Timers timers;
  std::unique_ptr<IOManager> io_manager = IOManagerFactory::make_instance( iomanager_id,
    configMap,
    foreach_cell,
    timers
  );
  std::unique_ptr<ParticleUpdate> particle_update_density;
  particle_update_density = ParticleUpdateFactory::make_instance( particle_update_density_id,
    configMap,
    foreach_cell,
    timers
  );

  uint32_t px=10, py=10, pz=10;
  uint32_t nParticles_tot = px*py*pz;
  ForeachParticle foreach_particle( *amr_mesh, configMap);

  int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
  uint32_t nParticles = (rank == 0) ? nParticles_tot : 0;

  U.new_ParticleArray("particles", nParticles);
  U.new_ParticleAttribute( "particles", "mass" );

  UserData::ParticleArray_t Ppos = U.getParticleArray( "particles" );
  UserData::ParticleAccessor Pdata = U.getParticleAccessor( "particles", {{"mass", IMASS}} );

  // Set particle positions to form a grid inside the domain
  constexpr real_t eps = 1e-8;
  foreach_particle.foreach_particle( "set_particle_pos", Ppos,
    PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart )
  {
      uint32_t ix =  iPart%px;
      uint32_t iy = (iPart/px)%py;
      uint32_t iz =  iPart/(px*py);

      Pdata.at( iPart, IMASS ) = 1.0/nParticles;
      Ppos.pos( iPart, IX ) = real_t(ix)/px + eps;
      Ppos.pos( iPart, IY ) = real_t(iy)/py + eps;
      Ppos.pos( iPart, IZ ) = (ndim-2)*((iz + eps)/pz);
  });

  // Exchange particles between MPI domains to match local AMR mesh
  U.distributeAllParticles();
  const UserData::ParticleArray_t& Ppos2 = U.getParticleArray( "particles" );
  printf("Rank %d after distributeAllParticles, nParticles = %d\n", rank, Ppos2.getNumParticles());
  
  { // Check particle count
    int nParticles_tot_old = nParticles_tot;
    int nParticles_tot_new = 0;
    int nParticles_new = Ppos2.getNumParticles();
    GlobalMpiSession::get_comm_world().MPI_Allreduce(&nParticles_new, &nParticles_tot_new, 1, MpiComm::MPI_Op_t::SUM);
    EXPECT_EQ(nParticles_tot_new, nParticles_tot_old);
  }

  ScalarSimulationData dummy_scalar_data;
  particle_update_density->update( U, dummy_scalar_data );


  real_t rho_mean_local = 0.; 
  const auto iter_space = Uin.getShape(); 
  foreach_cell.reduce_cell("Compute rho_mean", iter_space,
    KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell, real_t & update_rhomean)
  {
    pos_t size = cells.getCellSize( iCell );
    size[IZ] = (ndim == 2) ? 1.0 : size[IZ];    
    const real_t rhoi = Uin.at(iCell, IRHOG);
    update_rhomean += rhoi * size[IX] * size[IY] * size[IZ];
  }, Kokkos::Sum<real_t>(rho_mean_local));

  real_t rho_mean = 0.;
  GlobalMpiSession::get_comm_world().MPI_Allreduce(&rho_mean_local, &rho_mean, 1., MpiComm::MPI_Op_t::SUM);
  printf("Rank: %d, rhomean = %.5e, rhomean_local = %.5e\n", mpi_rank, rho_mean, rho_mean_local);

  EXPECT_NEAR( rho_mean, 1.0 , 1e-10);

  int iter = 0;
  int time = 0;
  dummy_scalar_data.set<int>("iter", iter++);
  dummy_scalar_data.set<real_t>("time",time++);
  io_manager->save_snapshot(U, dummy_scalar_data); 
} // run_test

} // namespace dyablo

class Test_ParticleUpdate_density
  : public testing::TestWithParam<std::tuple<int, std::string>> 
{};

TEST_P(Test_ParticleUpdate_density, rho_mean_is_correct)
{
  int ndim = std::get<0>(GetParam());
  std::string density_name = std::get<1>(GetParam());
  dyablo::run_test(ndim, density_name);
}

INSTANTIATE_TEST_SUITE_P(
    Test_ParticleUpdate_density, Test_ParticleUpdate_density,
    testing::Combine(
        testing::Values(2,3),
        testing::Values("ParticleUpdate_NGP_density", "ParticleUpdate_CIC_density", "ParticleUpdate_TSC_density")
    ),
    [](const testing::TestParamInfo<Test_ParticleUpdate_density::ParamType>& info) {
      int ndim = std::get<0>(info.param);
      std::string scheme = std::get<1>(info.param);
      std::string name = (ndim == 2 ? "2D_" : "3D_") + scheme;
      return name;
    }
);
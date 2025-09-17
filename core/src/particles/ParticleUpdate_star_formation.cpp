#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "foreach_cell/ForeachCell.h"
#include "utils/units/Units.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

#include <Kokkos_Random.hpp>

namespace dyablo {

namespace rand {
    using RNGPool = Kokkos::Random_XorShift64_Pool<>;
    using RNGType = RNGPool::generator_type;

    // ! Draw from a Poisson distribution
    template<typename real_t>
    KOKKOS_INLINE_FUNCTION
    uint32_t poisson(const real_t lambda, const RNGPool& rand_pool) {
        // Adapted from https://github.com/ramses-organisation/ramses/blob/3ef7f32e8a194cb73d337d27dcef759bf0a2277d/amr/random.f90#L61
        
        RNGType rand_gen = rand_pool.get_state();

        const uint32_t NPoissonLimit = 10;
        const uint32_t NpoissonLimitx10 = 10 * NPoissonLimit;

        uint32_t PoissNum;

        if (lambda <= NPoissonLimit) {
            const real_t Norm = exp(-lambda);
            real_t Repar = 1;
            real_t Proba = 1;
            PoissNum = 0;

            const real_t RandNum = rand_gen.drand();

            while ((Repar * Norm <= RandNum) & (PoissNum <= NpoissonLimitx10)) {
                ++PoissNum;
                Proba += lambda / PoissNum;
                Repar += Proba;
            }
        } else {
            const real_t GaussNum = FMAX(
                rand_gen.normal() * SQRT(lambda) - 0.5 + lambda,
                0.
            );
            PoissNum = std::round(GaussNum);
        }

        rand_pool.free_state(rand_gen);

        return PoissNum;
    }
}

class ParticleUpdate_star_formation : public ParticleUpdate {
public:
  using Policy = HyperbolicPolicy_State_Hydro;
  using Policy_Params = HyperbolicPolicy_Hydro_Params;

  ParticleUpdate_star_formation(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    policy_params(Policy_Params::from_configMap(configMap)),
    rho_threshold_physical( configMap.getValue_in_code_unit<Units::Density>("star_formation", "density_threshold", "10 proton_mass/cm**3") ),
    P_over_rho_threshold_physical(
      configMap.getValue_in_code_unit<Units::Temperature>("star_formation", "temperature_threshold", "1e4 K") *
      Units::constant_to_code_units(Units::KBOLTZ() / Units::PROTON_MASS())
    ),
    epsilon_star    ( configMap.getValue<real_t>("star_formation", "epsilon_star") ),    
    seed            ( 100 ),
    rand_pool       ( seed*GlobalMpiSession::get_comm_world().MPI_Comm_rank()+1)
  {
    const uint32_t level_max = configMap.getValue<uint32_t>("amr", "level_max");

    const real_t xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0);
    const real_t xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0);
    const real_t ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
    const real_t ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
    const real_t zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);
    const real_t zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);

    // Get length units
    const real_t Lx = (xmax - xmin);
    const real_t Ly = (ymax - ymin);
    const real_t Lz = (zmax - zmin);

    const uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 0);
    const uint32_t by = configMap.getValue<uint32_t>("amr", "by", 0);
    const uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", 1);

    const real_t min_dx = Lx / ((1 << level_max) * bx);
    const real_t min_dy = Ly / ((1 << level_max) * by);
    const real_t min_dz = Lz / ((1 << level_max) * bz);

    this->vol_min = min_dx * min_dy * min_dz;
  }

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {
    timers.get("ParticleUpdate_star_formation").start();

    const Policy policy( this->policy_params ); 

    UserData::FieldAccessor Uin = policy.getUin(U);

    const real_t dt = scalar_data.get<real_t>("dt");
    const real_t aexp = scalar_data.hasValue<real_t>("aexp") ? scalar_data.get<real_t>("aexp") : 1;

    const real_t time = scalar_data.hasValue<real_t>("time_physical") ?
      scalar_data.get<real_t>("time_physical")
      : scalar_data.get<real_t>("time");

    const real_t rho_threshold = Units::physical_to_supercomoving<Units::Density>(this->rho_threshold_physical, aexp);
    using P_over_rho_u = decltype(Units::m2() / Units::s2());
    const real_t P_over_rho_threshold = Units::physical_to_supercomoving<P_over_rho_u>(this->P_over_rho_threshold_physical, aexp);
    
    const real_t Mstar = rho_threshold * this->vol_min;
    const real_t epsilon_star = this->epsilon_star;
    using G_unit = decltype(Units::NEWTON_G());
    const real_t G = Units::physical_to_supercomoving<G_unit>(Units::constant_to_code_units(Units::NEWTON_G()), aexp);

    auto isStarFormingCell = KOKKOS_LAMBDA( real_t rho, real_t P )
    {
      return (rho > rho_threshold) & ((P / rho) < P_over_rho_threshold);
    };

    uint32_t n_star_forming_cells = 0;

    // First pass, count number of star forming cells
    foreach_cell.reduce_cell( "count_star_forming_cells", U.getShape(),
      CELL_LAMBDA(const ForeachCell::CellIndex& iCell, uint32_t& count)
    {
      auto u = policy.getConsState( Uin, iCell );
      auto q = policy.consToPrim( u );

      bool starForming = isStarFormingCell(q.rho, q.p);
      count += starForming ? 1 : 0;
    }, n_star_forming_cells);

    Kokkos::View< ForeachCell::CellIndex* > star_forming_cells("star_forming_cells", n_star_forming_cells);
    Kokkos::View<uint32_t> star_forming_cell_id("star_forming_cell_id");

    // Second pass, store star forming cell ids
    foreach_cell.foreach_cell( "store_star_forming_cells", U.getShape(),
      CELL_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
      auto u = policy.getConsState( Uin, iCell );
      auto q = policy.consToPrim( u );

      bool starForming = isStarFormingCell(q.rho, q.p);
      
      if( starForming )
      {
        uint32_t id = Kokkos::atomic_fetch_add(&star_forming_cell_id(), 1);
        star_forming_cells(id) = iCell;
      }
    });

    U.new_ParticleArray("spawned_particles", n_star_forming_cells);
    U.new_ParticleAttribute("spawned_particles", "mass");
    U.new_ParticleAttribute("spawned_particles", "vx");
    U.new_ParticleAttribute("spawned_particles", "vy");
    U.new_ParticleAttribute("spawned_particles", "vz");

    { // scope guard important to avoid keeping references to "spawned_particles" array
      enum VarIndex_particle{
        IMASS, IVX, IVY, IVZ, IBIRTH_TIME
      };
      UserData::ParticleAccessor Pnew_data = U.getParticleAccessor( "spawned_particles", {
        {"mass", IMASS},
        {"vx", IVX},
        {"vy", IVY},
        {"vz", IVZ},
        {"birth_time", IBIRTH_TIME}
      });

      auto Pnew = U.getParticleArray( "spawned_particles" );

      ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

      const auto& rand_pool = this->rand_pool;

      foreach_particle.foreach_particle( "fill_spawned_particles", Pnew,
        KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) 
      {
        auto iCell = star_forming_cells(iPart);

        const auto cell_size = cells.getCellSize( iCell );
        const auto cell_pos = cells.getCellCenter( iCell );

        const real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];

        auto u_in = policy.getConsState( Uin, iCell );
        auto q = policy.consToPrim( u_in );

        Pnew.pos(iPart, IX) = cell_pos[IX];
        Pnew.pos(iPart, IY) = cell_pos[IY];
        Pnew.pos(iPart, IZ) = cell_pos[IZ];
        Pnew_data.at(iPart, IVX) = q.u; 
        Pnew_data.at(iPart, IVY) = q.v; 
        Pnew_data.at(iPart, IVZ) = q.w; 
        Pnew_data.at(iPart, IBIRTH_TIME) = time;

        real_t Mparticle = 0;
        {
          real_t Mcell = Vcell * q.rho;

          // Free fall time of an homogenous sphere
          const real_t tstar = 0.5427 * SQRT(1 / (G * q.rho));
          // Gas mass to be converted into stars
          real_t Mgas = dt * epsilon_star * Mcell / tstar;

          const real_t Nstar_mean = Mgas / Mstar;
          const uint32_t Nstar = rand::poisson(Nstar_mean, rand_pool);


          Mparticle = FMIN(Nstar * Mstar, 0.9 * Mcell);
        }
        Pnew_data.at(iPart, IMASS) = Mparticle;

        q.rho -= Mparticle / Vcell;
        auto u_out = policy.primToCons( q );
        policy.setConsState( Uin, iCell, u_out );

      });
    }

    // Merge spawned_particles to particles ignoring particles with mass=0
    U.merge_particles_if( "particles", "spawned_particles", "mass" );    

    timers.get("ParticleUpdate_star_formation").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;
  Policy_Params policy_params;
  real_t rho_threshold_physical, P_over_rho_threshold_physical;
  real_t epsilon_star;
  real_t vol_min;
  int seed;
  rand::RNGPool rand_pool;

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_star_formation,
                  "ParticleUpdate_star_formation")



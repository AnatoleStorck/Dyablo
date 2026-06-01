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

class ParticleUpdate_star_formation_alt : public ParticleUpdate {
public:
  using Policy = HyperbolicPolicy_State_Hydro;
  using Policy_Params = HyperbolicPolicy_Hydro_Params;

  ParticleUpdate_star_formation_alt(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    policy_params(Policy_Params::from_configMap(configMap)),
    cosmology( configMap.getValue<bool>("cosmology", "active", false) ),
    rho_threshold_physical( configMap.getValue_in_code_unit<Units::Density>("star_formation", "density_threshold", "10 proton_mass/cm**3") ),
    P_over_rho_threshold_physical(
      configMap.getValue_in_code_unit<Units::Temperature>("star_formation", "temperature_threshold", "1e4 K") *
      Units::constant_to_code_units(Units::KBOLTZ() / Units::PROTON_MASS())
    ),
    m_particle( configMap.getValue_in_code_unit<Units::Mass>("star_formation", "particle_mass", "100 solar_mass") ),
    rho_m([&]() {
      using Inv_Time = decltype(1 / Units::s());
      using G_units = decltype(Units::NEWTON_G());
      // Set it to zero if not cosmological run
      if (!cosmology)
        return 0.0;

      // How much above the mean matter density should the threshold be?
      const real_t Delta = configMap.getValue<real_t>("star_formation", "mean_overdensity_threshold", 200);

      // Compute mean matter density
      const auto H0 = configMap.getValue_in_code_unit<Inv_Time>("cosmology", "H0");
      const auto four_pi_G = configMap.getValue_in_code_unit<G_units>( "gravity", "4_Pi_G" );
      const auto rhoc = 3.0 * H0 * H0 /( 2 * four_pi_G );
      const real_t omegam = configMap.getValue<real_t>("cosmology", "omegam", 0.3);

      return rhoc * omegam * Delta;
    }()),
    epsilon_star    ( configMap.getValue<real_t>("star_formation", "epsilon_star") ),
    seed            ( 100 ),
    rand_pool       ( seed*GlobalMpiSession::get_comm_world().MPI_Comm_rank()+1)
  {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {
    timers.get("ParticleUpdate_star_formation_alt").start();

    const Policy policy( this->policy_params );

    UserData::FieldAccessor Uin = policy.getUin(U);

    const real_t dt = scalar_data.get<real_t>("dt");
    const real_t aexp = scalar_data.hasValue<real_t>("aexp") ? scalar_data.get<real_t>("aexp") : 1;

    // auto solar_mass_unit = Units::SOLAR_MASS();
    // auto code_mass = Units::code_units().getUnit<Units::Mass>();

    const real_t time = cosmology ?
      scalar_data.get<real_t>("time_physical")
      : scalar_data.get<real_t>("time");

    // Star formation parameters
    const real_t rho_threshold = FMAX(
      Units::physical_to_supercomoving<Units::Density>(this->rho_threshold_physical, aexp),
      this->rho_m
    );
    using P_over_rho_u = decltype(Units::m2() / Units::s2());
    const real_t P_over_rho_threshold = Units::physical_to_supercomoving<P_over_rho_u>(this->P_over_rho_threshold_physical, aexp);

    const real_t Mstar = this->m_particle;
    const real_t epsilon_star = this->epsilon_star;

    // Gravitational constant in supercomoving units
    using G_unit = decltype(Units::NEWTON_G());
    const real_t G = Units::physical_to_supercomoving<G_unit>(Units::constant_to_code_units(Units::NEWTON_G()), aexp);

    // Optionally use the metallicity (if present)
    bool has_metallicity = U.has_field("metallicity");
    UserData::FieldAccessor UinZ;
    if (has_metallicity)
      UinZ = U.getAccessor( {{"metallicity", 0}} );

    auto isStarFormingCell = KOKKOS_LAMBDA( real_t rho, real_t P )
    {
      return (rho > rho_threshold) & ((P / rho) < P_over_rho_threshold);
    };

    uint32_t n_star_forming_cells = 0;

    // -------------------------------------------------------------------
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

    // -------------------------------------------------------------------
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

    // -------------------------------------------------------------------
    // Third pass, spawn star particles in star forming cells
    U.new_ParticleArray("spawned_particles", n_star_forming_cells);
    U.new_ParticleAttribute("spawned_particles", "mass");
    U.new_ParticleAttribute("spawned_particles", "birth_mass");
    U.new_ParticleAttribute("spawned_particles", "vx");
    U.new_ParticleAttribute("spawned_particles", "vy");
    U.new_ParticleAttribute("spawned_particles", "vz");
    U.new_ParticleAttribute("spawned_particles", "birth_time");
    U.new_ParticleAttribute("spawned_particles", "id");
    U.new_ParticleAttribute("spawned_particles", "metallicity");

    { // scope guard important to avoid keeping references to "spawned_particles" array
      enum VarIndex_particle{
        IMASS, IBIRTHMASS, IVX, IVY, IVZ, IBIRTH_TIME, IID, IMETALLICITY
      };
      UserData::ParticleAccessor Pnew_data = U.getParticleAccessor( "spawned_particles", {
        {"mass", IMASS},
        {"birth_mass", IBIRTHMASS},
        {"vx", IVX},
        {"vy", IVY},
        {"vz", IVZ},
        {"birth_time", IBIRTH_TIME},
        {"id", IID},
        {"metallicity", IMETALLICITY}
      } );

      auto Pnew = U.getParticleArray( "spawned_particles" );

      ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

      const auto& rand_pool = this->rand_pool;

      int Nstar_formed = 0;

      foreach_particle.reduce_particle( "fill_spawned_particles", Pnew,
        KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart, int& Nstar_formed)
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
          const real_t Mcell = Vcell * q.rho;

          // Free fall time of a homogeneous sphere
          const real_t tstar = 0.5427 * SQRT(1 / (G * q.rho));
          // Expected gas mass converted to stars over dt
          const real_t Mgas = dt * epsilon_star * Mcell / tstar;

          const real_t Nstar_mean = Mgas / Mstar;
          // Poisson sample number of star particles; cap so we never exceed available gas
          const uint32_t Nstar_max = static_cast<uint32_t>(Mcell / Mstar);
          const uint32_t Nstar = FMIN(rand::poisson(Nstar_mean, rand_pool), Nstar_max);

          Nstar_formed += Nstar > 0 ? 1 : 0;

          Mparticle = Nstar * Mstar;
        }

        Pnew_data.at(iPart, IMASS) = Mparticle;
        Pnew_data.at(iPart, IBIRTHMASS) = Mparticle;
        Pnew_data.at(iPart, IID) = iPart; // assign unique id based on particle index in this array (is this right?)
        if (has_metallicity) {
          real_t Zcell = UinZ.at_ivar(iCell, 0) / q.rho;
          Pnew_data.at(iPart, IMETALLICITY) = Zcell;
          // Need to update cell metallicity to account for change in density
          UinZ.at_ivar(iCell, 0) -= Zcell * Mparticle / Vcell;
        }

        q.rho -= Mparticle / Vcell;
        auto u_out = policy.primToCons( q );
        policy.setConsState( Uin, iCell, u_out );

      }, Nstar_formed);

      if (Nstar_formed > 0)
        std::cout << "Formed " << Nstar_formed << " star particles" << std::endl;
    }

    // Merge spawned_particles to particles ignoring particles with mass=0
    U.merge_particles_if( "particles", "spawned_particles", "mass" );

    timers.get("ParticleUpdate_star_formation_alt").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;
  Policy_Params policy_params;
  bool cosmology;

  real_t rho_threshold_physical, P_over_rho_threshold_physical;
  real_t m_particle;
  real_t rho_m;
  real_t epsilon_star;

  int seed;
  rand::RNGPool rand_pool;

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_star_formation_alt,
                  "ParticleUpdate_star_formation_alt")



#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "mpi/GhostCommunicator.h"
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
                Proba *= lambda / PoissNum;
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

namespace {

constexpr real_t PI = 3.14159265358979323846;

/// Density and velocity read from a (possibly refined) neighbour cell.
struct NeighborState {
  real_t rho;
  real_t u, v, w;
};

/// Runtime parameters of the turbulent star-formation model (code units).
struct SFParams {
  real_t G;               //!< gravitational constant (supercomoving units)
  real_t gamma0;          //!< adiabatic index
  real_t rho_threshold;   //!< density threshold for star formation
};

/// Result of evaluating a cell for turbulent star formation.
struct SFProps {
  bool   eligible;   //!< does the cell pass all four criteria ?
  real_t eps_star;   //!< star-formation efficiency per free-fall time
  real_t t_ff;       //!< gas free-fall time (code units)
};

/**
 * @brief Read density + velocity of the neighbour reached by @p offset.
 *
 * Handles the three cases the way GravitySolver_cg::get_value() does:
 *  - domain boundary / invalid : fall back to the central cell (zero gradient),
 *  - same level or coarser      : a single cell holds the value,
 *  - finer                      : average the smaller cells touching the face.
 */
template< typename Policy_t >
KOKKOS_INLINE_FUNCTION
NeighborState read_neighbor_state(
    const UserData::FieldAccessor& Uin,
    const Policy_t& policy,
    int ndim,
    const ForeachCell::CellIndex& iCell,
    const ForeachCell::CellIndex::offset_t& offset,
    const ForeachCell::SearchMode_neighbor& search_neighbor,
    const NeighborState& central )
{
  const ForeachCell::CellIndex iN = iCell.getNeighbor( offset, search_neighbor );

  if( !iN.is_valid() || iN.is_boundary() )
    return central;

  if( iN.level_diff() >= 0 )
  {
    auto q = policy.consToPrim( policy.getConsState( Uin, iN ) );
    return { q.rho, q.u, q.v, q.w };
  }

  // Finer neighbour : average the smaller cells in contact with this face.
  real_t rho = 0, mu = 0, mv = 0, mw = 0;
  const int nbCells = foreach_smaller_neighbor( ndim, iN, offset, search_neighbor,
    [&]( const ForeachCell::CellIndex& iSub )
  {
    auto q = policy.consToPrim( policy.getConsState( Uin, iSub ) );
    rho += q.rho; mu += q.u; mv += q.v; mw += q.w;
  });
  return { rho/nbCells, mu/nbCells, mv/nbCells, mw/nbCells };
}

/**
 * @brief Evaluate the four eligibility criteria and the SF efficiency for one cell.
 *
 * Eligibility (Following Katz+2024) :
 *   1. gas density above threshold,
 *   2. turbulent Jeans length unresolved (lambda_J,turb < dx),
 *   3. cell is a local density maximum w.r.t. its face neighbours,
 *   4. the flow is locally converging (div(v) < 0).
 */
template< typename Policy_t >
KOKKOS_INLINE_FUNCTION
SFProps compute_sf_props(
    const UserData::FieldAccessor& Uin,
    const Policy_t& policy,
    int ndim,
    const ForeachCell::CellMetaData& cells,
    const ForeachCell::CellIndex& iCell,
    const SFParams& p )
{
  const auto size = cells.getCellSize( iCell );

  // TODO: generalize to non-cubic cells (dx != dy != dz)
  DYABLO_ASSERT_KOKKOS_DEBUG( size[IX] == size[IY] && (ndim == 2 || size[IX] == size[IZ]),
    "ParticleUpdate_star_formation_turbulent only supports cubic cells, but cell size is " << size );
  const real_t dx = size[IX];   // octree cells are cubic : dx == dy == dz

  const auto qc = policy.consToPrim( policy.getConsState( Uin, iCell ) );
  const real_t rho = qc.rho;
  const NeighborState central{ qc.rho, qc.u, qc.v, qc.w };

  SFProps out{};   // value-initialised : eligible = false, eps_star = 0, t_ff = 0

  // The vast majority of cells are ineligible, so we check the cheapest criterion first to quickly skip them.
  if( rho <= p.rho_threshold ) {
    out.eligible = false;
    return out;
  }

  ForeachCell::SearchMode_neighbor search_neighbor(
      cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

  // Velocity-gradient tensor A[i][j] = d v_i / d x_j, density maximum and divergence.
  const real_t inv2dx = 1.0 / ( 2.0 * dx );
  real_t A[3][3] = { {0,0,0}, {0,0,0}, {0,0,0} };
  bool   is_density_max = true;
  real_t div_v = 0;
  for( int j = 0; j < ndim; ++j )
  {
    ForeachCell::CellIndex::offset_t off_L{}, off_R{};
    off_L[j] = -1;
    off_R[j] = +1;
    const NeighborState L = read_neighbor_state( Uin, policy, ndim, iCell, off_L, search_neighbor, central );
    const NeighborState R = read_neighbor_state( Uin, policy, ndim, iCell, off_R, search_neighbor, central );

    if( L.rho >= rho || R.rho >= rho )
      is_density_max = false;

    A[IX][j] = ( R.u - L.u ) * inv2dx;
    A[IY][j] = ( R.v - L.v ) * inv2dx;
    A[IZ][j] = ( R.w - L.w ) * inv2dx;
    div_v += A[j][j];
  }

  const bool converging = ( div_v < 0 );

  // The turbulent velocity dispersion is estimated from the divergence-removed
  // velocity-gradient tensor of the face neighbours :
  //   sigma^2 = dx^2 * sum_ij ( dv_i/dx_j - (div v) delta_ij / 3 )^2 .
  // TODO : Check this is the correct approach

  // Turbulent velocity dispersion (non-thermal)
  real_t sigma2 = 0;
  for( int i = 0; i < 3; ++i )
  for( int j = 0; j < 3; ++j )
  {
    real_t a = A[i][j];
    if( i == j )
      a -= div_v / 3.0;
    sigma2 += a * a;
  }
  sigma2 *= dx * dx;

  const real_t cs2 = p.gamma0 * qc.p / rho;   // sound speed squared

  // Turbulent Jeans length following Rosdahl+2018:
  // positive root of 3 G rho dx lambda^2 - pi sigma^2 lambda - 3 pi cs^2 dx = 0,
  // i.e. lambda_J = [ pi sigma^2 + sqrt(36 pi G cs^2 dx^2 rho + pi^2 sigma^4) ] / (6 G rho dx).
  const real_t sigma4 = sigma2 * sigma2;
  const real_t lambda_J =
      ( PI * sigma2 + SQRT( 36.0 * PI * p.G * cs2 * dx * dx * rho + PI * PI * sigma4 ) )
      / ( 6.0 * p.G * rho * dx );
  const bool jeans_unresolved = ( lambda_J < dx );

  out.eligible = ( rho > p.rho_threshold ) & jeans_unresolved & is_density_max & converging;
  if( !out.eligible )
    return out;

  // Free-fall time of a homogeneous sphere : t_ff = sqrt(3 pi / (32 G rho)).
  out.t_ff = SQRT( 3.0 * PI / ( 32.0 * p.G * rho ) );

  // --- Multi-free-fall star-formation efficiency ---
  // Federrath & Klessen 2012, their eq. (32) --> Padoan&Nordlund 2011
  constexpr real_t b              = 0.4;   // turbulence forcing parameter (compressive/solenoidal mix)
  constexpr real_t theta          = 0.33;  // post-shock thickness factor entering s_crit
  constexpr real_t eps_acc        = 0.5;   // accretion fraction of gas onto stars
  constexpr real_t one_over_phi_t = 0.57;  // 1/phi_t (free-fall-time correction factor)

  const real_t mach2 = sigma2 / FMAX( cs2, 1e-30 );    // turbulent Mach number squared
  const real_t sigma_s2 = log( 1.0 + b * b * mach2 );  // variance of the lognormal density PDF

  if( sigma_s2 <= 0 )   // no turbulence -> this channel forms no stars
    return out;

  // Virial parameter alpha_vir = 2 E_kin / |E_grav|, turbulent only, uniform sphere of radius dx/2.
  const real_t alpha_vir = 5.0 * sigma2 / ( PI * p.G * rho * dx * dx );

  // Critical density contrast for collapse.
  const real_t s_crit = log( 0.067 / ( theta * theta ) * alpha_vir * mach2 );

  const real_t prefactor = eps_acc * one_over_phi_t / 2.0;   // eps_acc / (2 phi_t)
  out.eps_star = prefactor * exp( 3.0 * sigma_s2 / 8.0 )
               * ( 1.0 + Kokkos::erf( ( sigma_s2 - s_crit ) / SQRT( 2.0 * sigma_s2 ) ) );

  return out;
}

} // anonymous namespace

/**
 * @brief Turbulence-regulated star formation following Katz+2024 (MEGATRON calibration paper).
 *
 * A cell is eligible for star formation when (i) its density is above a
 * threshold, (ii) its turbulent Jeans length is unresolved, (iii) it is a local
 * density maximum and (iv) the surrounding flow is converging. The local
 * star-formation rate density follows the multi-free-fall prescription
 *   rho_dot_star = eps_star * rho / t_ff ,
 * with a turbulence-dependent efficiency eps_star.
 * Star particles of fixed mass M_star are then drawn from a Poisson distribution.
 */
class ParticleUpdate_star_formation_turbulent : public ParticleUpdate {
public:
  using Policy = HyperbolicPolicy_State_Hydro;
  using Policy_Params = HyperbolicPolicy_Hydro_Params;

  ParticleUpdate_star_formation_turbulent(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    policy_params(Policy_Params::from_configMap(configMap)),
    cosmology( configMap.getValue<bool>("cosmology", "active", false) ),
    rho_threshold_physical( configMap.getValue_in_code_unit<Units::Density>("star_formation", "density_threshold", "1e3 proton_mass/cm**3") ),
    m_particle( configMap.getValue_in_code_unit<Units::Mass>("star_formation", "particle_mass", "500 solar_mass") ),
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
    gamma0          ( configMap.getValue<real_t>("hydro", "gamma0", 1.4) ),
    n_passive_scalars( configMap.getValue<int>("run", "n_passive_scalars",
        configMap.getValue<std::vector<std::string>>("run", "passive_scalars_names", {}).size()) ),
    seed            ( 100 ),
    rand_pool       ( seed*GlobalMpiSession::get_comm_world().MPI_Comm_rank()+1)
  {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {
    timers.get("ParticleUpdate_star_formation_turbulent").start();

    const Policy policy( this->policy_params );

    UserData::FieldAccessor Uin = policy.getUin(U);

    const real_t dt = scalar_data.get<real_t>("dt");
    const real_t aexp = scalar_data.hasValue<real_t>("aexp") ? scalar_data.get<real_t>("aexp") : 1;

    const real_t time = cosmology ?
      scalar_data.get<real_t>("time_physical")
      : scalar_data.get<real_t>("time");

    // Star formation density threshold (in code units, never below the cosmological floor)
    const real_t rho_threshold = FMAX(
      Units::physical_to_supercomoving<Units::Density>(this->rho_threshold_physical, aexp),
      this->rho_m
    );

    const real_t Mstar = this->m_particle;

    // Gravitational constant in supercomoving units
    using G_unit = decltype(Units::NEWTON_G());
    const real_t G = Units::physical_to_supercomoving<G_unit>(Units::constant_to_code_units(Units::NEWTON_G()), aexp);

    const int ndim = foreach_cell.getDim();

    const SFParams sfp{ G, this->gamma0, rho_threshold };

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    // The eligibility criteria read the six face neighbours : make sure the
    // hydro field ghosts are up to date across MPI domains before reading them.
    GhostCommunicator ghost_comm( foreach_cell.get_amr_mesh(), U.getShape(), 1 );
    ghost_comm.exchange_ghosts( Uin );

    // Optionally use the metallicity (if present)
    bool has_metallicity = U.has_field("metallicity");
    UserData::FieldAccessor UinZ;
    if (has_metallicity)
      UinZ = U.getAccessor( {{"metallicity", 0}} );

    std::vector<UserData::FieldAccessor::FieldInfo> passive_fields;
    for (int i = 0; i < n_passive_scalars; ++i)
      passive_fields.push_back( {"rho_scalar_" + std::to_string(i), i} );
    UserData::FieldAccessor Uin_passive;
    if (!passive_fields.empty())
      Uin_passive = U.getAccessor( passive_fields );

    uint32_t n_star_forming_cells = 0;

    // -------------------------------------------------------------------
    // First pass, count number of star forming cells
    foreach_cell.reduce_cell( "count_star_forming_cells", U.getShape(),
      CELL_LAMBDA(const ForeachCell::CellIndex& iCell, uint32_t& count)
    {
      const SFProps props = compute_sf_props( Uin, policy, ndim, cells, iCell, sfp );
      count += props.eligible ? 1 : 0;
    }, n_star_forming_cells);

    Kokkos::View< ForeachCell::CellIndex* > star_forming_cells("star_forming_cells", n_star_forming_cells);
    Kokkos::View<uint32_t> star_forming_cell_id("star_forming_cell_id");

    // -------------------------------------------------------------------
    // Second pass, store star forming cell ids
    foreach_cell.foreach_cell( "store_star_forming_cells", U.getShape(),
      CELL_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
      const SFProps props = compute_sf_props( Uin, policy, ndim, cells, iCell, sfp );
      if( props.eligible )
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

      const auto& rand_pool = this->rand_pool;

      int Nstar_formed = 0;

      foreach_particle.reduce_particle( "fill_spawned_particles", Pnew,
        KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart, int& Nstar_formed)
      {
        auto iCell = star_forming_cells(iPart);

        const auto cell_size = cells.getCellSize( iCell );
        const auto cell_pos = cells.getCellCenter( iCell );

        const real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];

        auto q = policy.consToPrim( policy.getConsState( Uin, iCell ) );

        Pnew.pos(iPart, IX) = cell_pos[IX];
        Pnew.pos(iPart, IY) = cell_pos[IY];
        Pnew.pos(iPart, IZ) = cell_pos[IZ];
        Pnew_data.at(iPart, IVX) = q.u;
        Pnew_data.at(iPart, IVY) = q.v;
        Pnew_data.at(iPart, IVZ) = q.w;
        Pnew_data.at(iPart, IBIRTH_TIME) = time;

        real_t Mparticle = 0;
        {
          // Turbulent star-formation efficiency and free-fall time for this cell.
          const SFProps props = compute_sf_props( Uin, policy, ndim, cells, iCell, sfp );

          const real_t Mcell = Vcell * q.rho;

          // Expected gas mass converted to stars over dt :
          //   rho_dot_star * dt * Vcell = eps_star * rho / t_ff * dt * Vcell.
          const real_t Mgas = dt * props.eps_star * Mcell / props.t_ff;

          const real_t Nstar_mean = Mgas / Mstar;
          const uint32_t Nstar = rand::poisson(Nstar_mean, rand_pool);

          Nstar_formed += Nstar > 0 ? 1 : 0;

          // Cap the converted mass so at least 10% of the cell gas remains.
          Mparticle = FMIN(Nstar * Mstar, 0.9 * Mcell);
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

        // Deplete the gas and passive scalars following SF
        const real_t rho_old = q.rho;
        q.rho -= Mparticle / Vcell;
        if (Mparticle > 0) {
          const real_t rho_factor = q.rho / rho_old;
          for (int ivar = 0; ivar < Uin_passive.nbFields(); ++ivar)
            Uin_passive.at_ivar(iCell, ivar) *= rho_factor;
        }

        auto u_out = policy.primToCons( q );
        policy.setConsState( Uin, iCell, u_out );

      }, Nstar_formed);

      if (Nstar_formed > 0)
        std::cout << "Formed " << Nstar_formed << " star particles" << std::endl;
    }

    // Merge spawned_particles to particles ignoring particles with mass=0
    U.merge_particles_if( "particles", "spawned_particles", "mass" );

    timers.get("ParticleUpdate_star_formation_turbulent").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;
  Policy_Params policy_params;
  bool cosmology;

  real_t rho_threshold_physical;
  real_t m_particle;
  real_t rho_m;
  real_t gamma0;
  int n_passive_scalars;
  int seed;
  rand::RNGPool rand_pool;

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_star_formation_turbulent,
                  "ParticleUpdate_star_formation_turbulent")

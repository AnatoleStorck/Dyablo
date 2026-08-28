#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "SinkUtils.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "mpi/GhostCommunicator.h"
#include "utils/units/Units.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

namespace dyablo {

/**
 * @brief Sink particle accretion (RAMSES sink_particle.f90 accrete_sink). Each step a
 * sink removes gas from its accretion sphere of radius ir_cloud cells, at the local
 * gas velocity and specific energy; mass, momentum and the center-of-mass shift are
 * added to the sink. [sink] accretion_scheme selects how much is taken :
 *
 *   'threshold' : m = c_acc * (rho - rho_sink) * V, per cell, independently of dt.
 *   'flux'      : the sink swallows the net mass inflow through the accretion sphere,
 *                 Mdot = -Int div(rho (v - v_sink)) dV * f_a  (Bleuler & Teyssier 2014),
 *                 with f_a = 0.1*(log10(rho_mean) - log10(rho_sink)) + 1 a weak
 *                 regulator holding the zone near the threshold density. Mdot*dt is
 *                 then shared over the sphere by volume.
 *
 * Whichever scheme sets the request, the per-cell limiter below is the same, so gas
 * never drops below rho_sink and a cell can never be emptied.
 *
 * Runs in the [particles] source_term hook (before ParticleUpdate_sink_merging).
 * MPI/multi-sink correctness uses a request -> limit -> settle pattern on two
 * scratch fields :
 *   P1  every sink atomically adds its per-cell request (ghost cells included),
 *   P2  reduce_ghosts sums the requests on the owner, which computes the settle
 *       factor f = min(1, available/requested) - N competing sinks share the
 *       available excess proportionally and a cell can never drop below rho_sink,
 *   P3  the factors are published back to the ghosts,
 *   P4  every sink re-walks its sphere and gains f * request per slot,
 *   P5  the owner cells remove exactly what the sinks gained (density-proportional
 *       scaling of momentum, energies and passive scalars = removal at the local
 *       gas state).
 * The result is independent of the domain decomposition, and gas loss equals sink
 * gain to floating-point roundoff.
 */
class ParticleUpdate_sink_accretion : public ParticleUpdate {
public:
  using Policy = HyperbolicPolicy_State_Hydro;
  using Policy_Params = HyperbolicPolicy_Hydro_Params;
  using pos_t = Kokkos::Array<real_t, 3>;
  using ConsState = HyperbolicPolicy_ConsHydroState;

  ParticleUpdate_sink_accretion(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    policy_params( Policy_Params::from_configMap(configMap) ),
    params( SinkParams::from_configMap(configMap, foreach_cell) ),
    n_passive_scalars( configMap.getValue<int>("passive_scalars", "n_passive_scalars",
        configMap.getValue<std::vector<std::string>>("passive_scalars", "passive_scalars_names", {}).size()) )
  {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {
    timers.get("ParticleUpdate_sink_accretion").start();

    int n_loc = 0;
    if( U.has_ParticleArray( params.family ) )
      n_loc = U.getParticleArray( params.family ).getNumParticles();
    const MpiComm comm = foreach_cell.get_amr_mesh().getMpiComm();
    int n_glob = 0;
    comm.MPI_Allreduce( &n_loc, &n_glob, 1, MpiComm::MPI_Op_t::SUM );
    if( n_glob == 0 )
    {
      timers.get("ParticleUpdate_sink_accretion").stop();
      return;
    }

    const Policy policy( this->policy_params );

    const real_t aexp = scalar_data.hasValue<real_t>("aexp") ? scalar_data.get<real_t>("aexp") : 1;
    const real_t rho_sink = Units::physical_to_supercomoving<Units::Density>( params.rho_sink_physical, aexp );
    const real_t c_acc = params.c_acc;
    const real_t dt = scalar_data.get<real_t>("dt");
    const int R = params.ir_cloud;
    const int ndim = foreach_cell.getDim();
    const int level_max = params.level_max;
    const bool flux_scheme = ( params.accretion_scheme == SinkAccretionScheme::Flux );

    // Allocate the scratch fields before taking any accessor : new_fields() reallocates the
    // field array when it needs more slots, which invalidates every accessor taken before it.
    enum VarIndex_acc{ IREQ, IFAC };
    U.new_fields( {"sink_acc_request", "sink_acc_factor"} );   // zero-initialized incl. ghosts

    UserData::FieldAccessor Uin = policy.getUin(U);
    auto Uacc = U.getAccessor( { {"sink_acc_request", IREQ}, {"sink_acc_factor", IFAC} } );

    // Post-hydro ghosts are stale, and requests reach ir_cloud cells into neighbor octants
    GhostCommunicator ghost_comm( foreach_cell.get_amr_mesh(), U.getShape(), R );
    ghost_comm.exchange_ghosts( Uin );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    // Flux scheme : mass each sink asks for per unit of accretion-zone volume,
    // Mdot*dt/V_acc.
    Kokkos::View<real_t*> flux_share( "sink_flux_share", flux_scheme ? n_loc : 0 );

    // ---------------------------------------------------------------
    // P1 : every sink deposits its per-slot request (ghost cells included)
    if( n_loc > 0 )
    {
      enum VarIndex_particle{ IVX, IVY, IVZ };
      auto Ppos = U.getParticleArray( params.family );
      auto Pvel = U.getParticleAccessor( params.family,
        { {"vx",IVX}, {"vy",IVY}, {"vz",IVZ} } );

      foreach_particle.foreach_particle( "sink_accretion_request", Ppos,
        KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
      {
        ForeachCell::SearchMode_neighbor search_neighbor(
          cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

        const pos_t part_pos = { Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ) };
        const ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );
        const pos_t cell_size = cells.getCellSize( iCell );
        const real_t Vc = cell_size[IX]*cell_size[IY]*cell_size[IZ];

        if( !flux_scheme )
        { // RAMSES 'threshold' : take a fixed fraction of the excess above rho_sink
          foreach_sphere_slot( iCell, R, ndim, search_neighbor, /*include_ghosts*/true,
            [&]( const ForeachCell::CellIndex& iT, real_t w, const Kokkos::Array<int,3>& )
          {
            const real_t excess = Uin.at( iT, ConsState::VarIndex::Irho ) - rho_sink;
            if( excess <= 0 ) return;
            Kokkos::atomic_add( &Uacc.at( iT, IREQ ), c_acc * excess * w * Vc );
          });
          return;
        }

        // RAMSES 'flux' (Bleuler & Teyssier 2014) : the sink swallows the net mass
        // inflow through its accretion sphere,
        //   Mdot = -Int_V div( rho (v - v_sink) ) dV  =  -Oint_S rho (v - v_sink).n dA
        // Walk 1 accumulates the accretion-zone volume and mean density, and the
        // surface flux. Only slot faces whose neighbouring slot is outside the
        // sphere contribute : interior faces cancel pairwise, which is the discrete
        // divergence theorem. RAMSES instead evaluates div() at each cloud particle
        // and sums it over the sphere - the same integral, but the surface form
        // needs no neighbour lookup outside the sphere, which matters here because
        // ir_cloud may be as large as the AMR block and a second getNeighbor() hop
        // could leave the directly-contiguous octants.
        const real_t face_area = cell_size[IX]*cell_size[IY];
        const real_t vsx = Pvel.at(iPart, IVX);
        const real_t vsy = Pvel.at(iPart, IVY);
        const real_t vsz = Pvel.at(iPart, IVZ);

        real_t V_acc = 0, mass_acc = 0, flux_out = 0;
        foreach_sphere_slot( iCell, R, ndim, search_neighbor, /*include_ghosts*/true,
          [&]( const ForeachCell::CellIndex& iT, real_t w, const Kokkos::Array<int,3>& off )
        {
          const real_t rho = Uin.at( iT, ConsState::VarIndex::Irho );
          V_acc    += w * Vc;
          mass_acc += rho * w * Vc;

          // rho*(v - v_sink) of this slot, per direction
          const real_t F[3] = {
            Uin.at( iT, ConsState::VarIndex::Irho_vx ) - rho*vsx,
            Uin.at( iT, ConsState::VarIndex::Irho_vy ) - rho*vsy,
            Uin.at( iT, ConsState::VarIndex::Irho_vz ) - rho*vsz };

          for( int d = 0 ; d < 3 ; d++ )
          for( int s = -1 ; s <= 1 ; s += 2 )
          {
            Kokkos::Array<int,3> nb = off;
            nb[d] += s;
            if( sink_slot_in_sphere( nb[IX], nb[IY], nb[IZ], R ) ) continue;
            flux_out += w * s * F[d] * face_area;
          }
        });

        if( V_acc <= 0 ) return;
        const real_t rho_mean = mass_acc / V_acc;
        if( rho_mean <= 0 ) return;

        // Weak regulator keeping the accretion zone near the threshold density
        const real_t fa = 0.1 * ( log10(rho_mean) - log10(rho_sink) ) + 1.0;
        const real_t Mdot = -flux_out * fa;
        if( Mdot <= 0 ) return;   // net outflow : nothing to accrete this step

        // Mdot*dt shared over the accretion zone by volume (RAMSES weight/volume)
        const real_t share = Mdot * dt / V_acc;
        flux_share(iPart) = share;

        foreach_sphere_slot( iCell, R, ndim, search_neighbor, /*include_ghosts*/true,
          [&]( const ForeachCell::CellIndex& iT, real_t w, const Kokkos::Array<int,3>& )
        {
          Kokkos::atomic_add( &Uacc.at( iT, IREQ ), share * w * Vc );
        });
      });
    }

    // ---------------------------------------------------------------
    // P2 : requests come home, owner cells compute the settle factor
    ghost_comm.reduce_ghosts( Uacc );
    foreach_cell.foreach_cell( "sink_accretion_limit", U.getShape(),
      CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      const real_t req = Uacc.at( iCell, IREQ );
      if( req <= 0 ) return;
      const pos_t cell_size = cells.getCellSize( iCell );
      const real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];
      const real_t excess = Uin.at( iCell, ConsState::VarIndex::Irho ) - rho_sink;
      const real_t m_avail = c_acc * FMAX( excess, 0. ) * Vcell;
      Uacc.at( iCell, IFAC ) = FMIN( 1., m_avail / req );
    });

    // P3 : publish the settled factors (and totals) to the ghost copies
    ghost_comm.exchange_ghosts( Uacc );

    // ---------------------------------------------------------------
    // P4 : sinks re-walk their sphere and gain their settled share.
    // The conservative fields still hold the pre-removal state here.
    real_t dM_tot = 0;
    int n_coarse = 0;
    if( n_loc > 0 )
    {
      enum VarIndex_particle{ IMASS, IVX, IVY, IVZ, IDMACC };
      auto Ppos = U.getParticleArray( params.family );
      auto Pdata = U.getParticleAccessor( params.family,
        { {"mass",IMASS}, {"vx",IVX}, {"vy",IVY}, {"vz",IVZ}, {"dm_acc",IDMACC} } );

      const real_t Lx = params.xmax - params.xmin;
      const real_t Ly = params.ymax - params.ymin;
      const real_t Lz = params.zmax - params.zmin;
      const auto periodic = params.periodic;
      const real_t xmin = params.xmin, ymin = params.ymin, zmin = params.zmin;

      foreach_particle.reduce_particle( "sink_accretion_gain", Ppos,
        KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart, real_t& dM_sum, int& coarse_count )
      {
        ForeachCell::SearchMode_neighbor search_neighbor(
          cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

        const pos_t part_pos = { Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ) };
        const ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );
        const pos_t cell_size = cells.getCellSize( iCell );
        const real_t Vc = cell_size[IX]*cell_size[IY]*cell_size[IZ];

        // The accretion radius follows the sink's current cell size : warn when
        // the mesh has coarsened under a sink (keep sinks refined, e.g. jeans refinement)
        coarse_count += ( cells.getCellLevel(iCell) < level_max ) ? 1 : 0;

        // Same per-slot share as P1, times the factor the owner cell settled on
        const real_t share = flux_scheme ? flux_share(iPart) : 0;

        real_t dM = 0, dpx = 0, dpy = 0, dpz = 0, dxw = 0, dyw = 0, dzw = 0;
        foreach_sphere_slot( iCell, R, ndim, search_neighbor, /*include_ghosts*/true,
          [&]( const ForeachCell::CellIndex& iT, real_t w, const Kokkos::Array<int,3>& )
        {
          const real_t rho_t = Uin.at( iT, ConsState::VarIndex::Irho );
          real_t m;
          if( flux_scheme )
          {
            m = share * w * Vc * Uacc.at( iT, IFAC );
          }
          else
          {
            const real_t excess = rho_t - rho_sink;
            if( excess <= 0 ) return;
            m = c_acc * excess * w * Vc * Uacc.at( iT, IFAC );
          }
          if( m <= 0 ) return;

          const real_t ux = Uin.at( iT, ConsState::VarIndex::Irho_vx ) / rho_t;
          const real_t uy = Uin.at( iT, ConsState::VarIndex::Irho_vy ) / rho_t;
          const real_t uz = Uin.at( iT, ConsState::VarIndex::Irho_vz ) / rho_t;
          const auto tpos = cells.getCellCenter( iT );

          dM  += m;
          dpx += m * ux;  dpy += m * uy;  dpz += m * uz;
          dxw += m * sink_wrapped_delta( tpos[IX], part_pos[IX], Lx, periodic[IX] );
          dyw += m * sink_wrapped_delta( tpos[IY], part_pos[IY], Ly, periodic[IY] );
          dzw += m * sink_wrapped_delta( tpos[IZ], part_pos[IZ], Lz, periodic[IZ] );
        });

        Pdata.at(iPart, IDMACC) = dM;
        if( dM > 0 )
        {
          const real_t M = Pdata.at(iPart, IMASS);
          const real_t M_new = M + dM;
          // momentum conservation, and the RAMSES center-of-mass shift
          Pdata.at(iPart, IVX) = ( M * Pdata.at(iPart, IVX) + dpx ) / M_new;
          Pdata.at(iPart, IVY) = ( M * Pdata.at(iPart, IVY) + dpy ) / M_new;
          Pdata.at(iPart, IVZ) = ( M * Pdata.at(iPart, IVZ) + dpz ) / M_new;
          Ppos.pos(iPart, IX) = fmod( ( part_pos[IX] + dxw/M_new - xmin ) + Lx, Lx ) + xmin;
          Ppos.pos(iPart, IY) = fmod( ( part_pos[IY] + dyw/M_new - ymin ) + Ly, Ly ) + ymin;
          Ppos.pos(iPart, IZ) = fmod( ( part_pos[IZ] + dzw/M_new - zmin ) + Lz, Lz ) + zmin;
          Pdata.at(iPart, IMASS) = M_new;
          dM_sum += dM;
        }
      }, dM_tot, n_coarse);
    }
    if( n_coarse > 0 )
      std::cout << "WARNING : " << n_coarse << " sink(s) sit in cells coarser than level_max,"
                   " their accretion radius has grown accordingly" << std::endl;

    // ---------------------------------------------------------------
    // P5 : owner cells remove exactly what the sinks gained. Removal at the local
    // gas state = density-proportional scaling of every per-volume quantity.
    std::vector<UserData::FieldAccessor::FieldInfo> passive_fields;
    for (int i = 0; i < n_passive_scalars; ++i)
      passive_fields.push_back( {"rho_scalar_" + std::to_string(i), i} );
    UserData::FieldAccessor Uin_passive;
    if (!passive_fields.empty())
      Uin_passive = U.getAccessor( passive_fields );

    foreach_cell.foreach_cell( "sink_accretion_remove", U.getShape(),
      CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      const real_t m_rem = Uacc.at( iCell, IFAC ) * Uacc.at( iCell, IREQ );
      if( m_rem <= 0 ) return;
      const pos_t cell_size = cells.getCellSize( iCell );
      const real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];

      const real_t rho_old = Uin.at( iCell, ConsState::VarIndex::Irho );
      const real_t rho_new = rho_old - m_rem / Vcell;
      const real_t factor = rho_new / rho_old;

      Uin.at( iCell, ConsState::VarIndex::Irho )    = rho_new;
      Uin.at( iCell, ConsState::VarIndex::Irho_vx ) *= factor;
      Uin.at( iCell, ConsState::VarIndex::Irho_vy ) *= factor;
      Uin.at( iCell, ConsState::VarIndex::Irho_vz ) *= factor;
      Uin.at( iCell, ConsState::VarIndex::Ie_tot )  *= factor;
      Uin.at( iCell, ConsState::VarIndex::Ie_int )  *= factor;
      for (int ivar = 0; ivar < Uin_passive.nbFields(); ++ivar)
        Uin_passive.at_ivar(iCell, ivar) *= factor;
    });

    U.delete_field( "sink_acc_request" );
    U.delete_field( "sink_acc_factor" );

    // The center-of-mass shift can move a sink across a rank boundary, and the
    // source_term hook is not followed by an automatic redistribution.
    if( U.has_ParticleArray( params.family ) )
      U.distributeParticles( params.family );

    timers.get("ParticleUpdate_sink_accretion").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;
  Policy_Params policy_params;
  SinkParams params;
  int n_passive_scalars;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_sink_accretion,
                  "ParticleUpdate_sink_accretion")

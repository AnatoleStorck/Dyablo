#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "SinkUtils.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "mpi/GhostCommunicator.h"
#include "utils/units/Units.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

#include <algorithm>
#include <numeric>

namespace dyablo {

namespace {

constexpr real_t PI = 3.14159265358979323846;

using CellIndex = ForeachCell::CellIndex;
using Policy = HyperbolicPolicy_State_Hydro;
using ConsState = HyperbolicPolicy_ConsHydroState;

enum VarIndex_g{ IGX, IGY, IGZ };

/// Everything the per-cell candidate predicate needs, device-copyable.
struct SinkFormationCtx {
  UserData::FieldAccessor Uin;        //!< hydro conservatives (ghosts exchanged ir_cloud deep)
  UserData::FieldAccessor Ug;         //!< gx,gy,gz - only valid when check_energies
  ForeachCell::CellMetaData cells;
  Kokkos::View<real_t**> sink_table;  //!< GlobalSinkTable::device
  int n_sinks;
  Policy policy;
  int ndim;
  int level_max;
  int ir_cloud;
  real_t rho_sink;
  real_t excl_radius2;                //!< (2*r_acc)^2 : min distance^2 to an existing sink
  real_t G;                           //!< only used by the Jeans proxy (check_energies=false)
  real_t gamma0;
  bool check_energies;
  real_t Lx, Ly, Lz;
  Kokkos::Array<bool,3> periodic;
};

struct FaceVel { real_t u, v, w; };

/// Velocity of the face neighbor reached by @p offset (finer subcells averaged,
/// central fallback at the domain boundary - zero-gradient).
KOKKOS_INLINE_FUNCTION
FaceVel read_face_velocity( const SinkFormationCtx& ctx, const CellIndex& iCell,
                            const CellIndex::offset_t& offset,
                            const ForeachCell::SearchMode_neighbor& search_neighbor,
                            const FaceVel& central )
{
  const CellIndex iN = iCell.getNeighbor( offset, search_neighbor );
  if( !iN.is_valid() )
    return central;
  if( iN.level_diff() >= 0 )
  {
    auto q = ctx.policy.consToPrim( ctx.policy.getConsState( ctx.Uin, iN ) );
    return { q.u, q.v, q.w };
  }
  real_t mu = 0, mv = 0, mw = 0;
  const int nbCells = foreach_smaller_neighbor( ctx.ndim, iN, offset, search_neighbor,
    [&]( const CellIndex& iSub )
  {
    auto q = ctx.policy.consToPrim( ctx.policy.getConsState( ctx.Uin, iSub ) );
    mu += q.u; mv += q.v; mw += q.w;
  });
  return { mu/nbCells, mv/nbCells, mw/nbCells };
}

/**
 * @brief Sink-formation candidate predicate (simplified RAMSES clump-finder gate).
 *
 * A cell is a candidate when :
 *   1. it is at the finest refinement level,
 *   2. its density exceeds rho_sink,
 *   3. it is a local density maximum over its 26 neighbors (x1.0001 plateau
 *      tiebreak like RAMSES peakcheck) and the flow is locally converging,
 *   4. no existing sink lies within 2*r_acc,
 *   5. the accretion sphere passes the energy gate : gravity dominates both the
 *      kinetic and the thermal support (RAMSES check_energies,
 *      flag_formation_sites.f90), or - without a gravity solver - the thermal
 *      Jeans length is unresolved at the finest level.
 *
 * Candidates closer than 2*r_acc to each other are thinned deterministically on
 * the host afterwards (densest wins).
 */
KOKKOS_INLINE_FUNCTION
bool sink_candidate_check( const SinkFormationCtx& ctx, const CellIndex& iCell )
{
  // 1. finest level only : guarantees the accretion radius is ir_cloud*dx_min
  if( ctx.cells.getCellLevel(iCell) != ctx.level_max )
    return false;

  auto qc = ctx.policy.consToPrim( ctx.policy.getConsState( ctx.Uin, iCell ) );

  // 2. density threshold
  if( qc.rho <= ctx.rho_sink )
    return false;

  ForeachCell::SearchMode_neighbor search_neighbor(
    ctx.cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

  // 3a. local density maximum over the 26 neighbors
  const real_t rho_max_test = qc.rho * 1.0001;
  for( int dk = -1 ; dk <= 1 ; dk++ )
  for( int dj = -1 ; dj <= 1 ; dj++ )
  for( int di = -1 ; di <= 1 ; di++ )
  {
    if( di == 0 && dj == 0 && dk == 0 ) continue;
    const CellIndex::offset_t offset = { (int16_t)di, (int16_t)dj, (int16_t)dk };
    const CellIndex iN = iCell.getNeighbor( offset, search_neighbor );
    if( !iN.is_valid() ) continue;
    if( iN.level_diff() >= 0 )
    {
      if( ctx.Uin.at( iN, ConsState::VarIndex::Irho ) > rho_max_test ) return false;
    }
    else
    { // test each finer subcell individually - averaging could mask a denser one
      bool denser = false;
      foreach_smaller_neighbor( ctx.ndim, iN, offset, search_neighbor,
        [&]( const CellIndex& iSub )
      {
        if( ctx.Uin.at( iSub, ConsState::VarIndex::Irho ) > rho_max_test ) denser = true;
      });
      if( denser ) return false;
    }
  }

  // 3b. converging flow : div(v) < 0 from face-neighbor central differences
  const auto cell_size = ctx.cells.getCellSize( iCell );
  const real_t dx = cell_size[IX];
  const FaceVel central{ qc.u, qc.v, qc.w };
  real_t div_v = 0;
  for( int d = 0 ; d < 3 ; d++ )
  {
    CellIndex::offset_t oL{}, oR{};
    oL[d] = -1; oR[d] = +1;
    const FaceVel L = read_face_velocity( ctx, iCell, oL, search_neighbor, central );
    const FaceVel R = read_face_velocity( ctx, iCell, oR, search_neighbor, central );
    const real_t vL = (d == IX) ? L.u : (d == IY) ? L.v : L.w;
    const real_t vR = (d == IX) ? R.u : (d == IY) ? R.v : R.w;
    div_v += ( vR - vL ) / ( 2.0 * dx );
  }
  if( div_v >= 0 )
    return false;

  // 4. no existing sink within 2*r_acc
  const auto cpos = ctx.cells.getCellCenter( iCell );
  for( int i = 0 ; i < ctx.n_sinks ; i++ )
  {
    const real_t rx = sink_wrapped_delta( ctx.sink_table(i, GlobalSinkTable::X), cpos[IX], ctx.Lx, ctx.periodic[IX] );
    const real_t ry = sink_wrapped_delta( ctx.sink_table(i, GlobalSinkTable::Y), cpos[IY], ctx.Ly, ctx.periodic[IY] );
    const real_t rz = sink_wrapped_delta( ctx.sink_table(i, GlobalSinkTable::Z), cpos[IZ], ctx.Lz, ctx.periodic[IZ] );
    if( rx*rx + ry*ry + rz*rz < ctx.excl_radius2 )
      return false;
  }

  // 5. energy gate over the accretion sphere
  const real_t Vc = cell_size[IX]*cell_size[IY]*cell_size[IZ];
  real_t M = 0, px = 0, py = 0, pz = 0, K2 = 0, therm3 = 0, W = 0, Vsum = 0;
  foreach_sphere_slot( iCell, ctx.ir_cloud, ctx.ndim, search_neighbor, /*include_ghosts*/true,
    [&]( const CellIndex& iT, real_t w )
  {
    const real_t V_slot = w * Vc;
    auto qt = ctx.policy.consToPrim( ctx.policy.getConsState( ctx.Uin, iT ) );
    M      += qt.rho * V_slot;
    px     += qt.rho * qt.u * V_slot;
    py     += qt.rho * qt.v * V_slot;
    pz     += qt.rho * qt.w * V_slot;
    K2     += qt.rho * ( qt.u*qt.u + qt.v*qt.v + qt.w*qt.w ) * V_slot;
    therm3 += 3.0 * qt.p * V_slot;
    Vsum   += V_slot;
    if( ctx.check_energies )
    {
      const auto tpos = ctx.cells.getCellCenter( iT );
      const real_t rx = sink_wrapped_delta( tpos[IX], cpos[IX], ctx.Lx, ctx.periodic[IX] );
      const real_t ry = sink_wrapped_delta( tpos[IY], cpos[IY], ctx.Ly, ctx.periodic[IY] );
      const real_t rz = sink_wrapped_delta( tpos[IZ], cpos[IZ], ctx.Lz, ctx.periodic[IZ] );
      W += qt.rho * (   ctx.Ug.at(iT, IGX) * rx
                      + ctx.Ug.at(iT, IGY) * ry
                      + ctx.Ug.at(iT, IGZ) * rz ) * V_slot;
    }
  });
  if( M <= 0 || Vsum <= 0 )
    return false;

  if( ctx.check_energies )
  {
    // 2*E_kin < |W| and 3*Int(P dV) < |W|, with the bulk motion removed from E_kin
    const real_t Ekin = 0.5 * ( K2 - ( px*px + py*py + pz*pz ) / M );
    if( !( W < 0 && 2.0*Ekin < -W && therm3 < -W ) )
      return false;
  }
  else
  {
    // thermal-Jeans proxy : lambda_J^2 = pi * c_s^2 / (G rho), c_s^2 = gamma*<P>/rho
    // evaluated with the sphere-averaged pressure and the peak density
    const real_t P_avg = therm3 / ( 3.0 * Vsum );
    const real_t cs2 = ctx.gamma0 * P_avg / qc.rho;
    const real_t lambda_J2 = PI * cs2 / ( ctx.G * qc.rho );
    if( dx*dx <= lambda_J2 )
      return false;
  }

  return true;
}

} // anonymous namespace

/**
 * @brief Sink particle formation (RAMSES sink_particle.f90 creation path, with a
 * simplified FLASH-style formation-site detector in place of the PHEW clump finder).
 *
 * Runs in the [particles] spawn hook. Candidate cells (see sink_candidate_check)
 * are gathered on every rank, thinned deterministically (densest first, no two
 * accepted sites or existing sinks within 2*r_acc), then spawned as particles of
 * the [sink] sink_family with a seed mass taken from the peak cell.
 */
class ParticleUpdate_sink_formation : public ParticleUpdate {
public:
  using Policy = HyperbolicPolicy_State_Hydro;
  using Policy_Params = HyperbolicPolicy_Hydro_Params;

  ParticleUpdate_sink_formation(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    policy_params( Policy_Params::from_configMap(configMap) ),
    params( SinkParams::from_configMap(configMap, foreach_cell) ),
    cosmology( configMap.getValue<bool>("cosmology", "active", false) ),
    n_passive_scalars( configMap.getValue<int>("passive_scalars", "n_passive_scalars",
        configMap.getValue<std::vector<std::string>>("passive_scalars", "passive_scalars_names", {}).size()) ),
    four_pi_G( [&]() -> real_t {
      // Only the Jeans-proxy gate needs G; the virial gate reads gx,gy,gz instead.
      if( params.check_energies )
        return 0.0;
      using G_units = decltype(Units::NEWTON_G());
      return configMap.getValue_in_code_unit<G_units>( "gravity", "4_Pi_G" );
    }() )
  {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {
    timers.get("ParticleUpdate_sink_formation").start();

    const Policy policy( this->policy_params );
    UserData::FieldAccessor Uin = policy.getUin(U);

    const real_t aexp = scalar_data.hasValue<real_t>("aexp") ? scalar_data.get<real_t>("aexp") : 1;
    const real_t time = cosmology ?
      scalar_data.get<real_t>("time_physical")
      : scalar_data.get<real_t>("time");

    const real_t rho_sink = Units::physical_to_supercomoving<Units::Density>( params.rho_sink_physical, aexp );
    const real_t mass_seed_cfg = Units::physical_to_supercomoving<Units::Mass>( params.mass_sink_seed_physical, aexp );

    // Ensure the sink family exists (also lets accretion/merging run without formation)
    if( !U.has_ParticleArray( params.family ) )
    {
      U.new_ParticleArray( params.family, 0 );
      for( const char* attr : { "mass", "vx", "vy", "vz", "id", "birth_time", "dm_acc" } )
        U.new_ParticleAttribute( params.family, attr );
    }

    // The post-hydro conservative fields have stale ghosts; the candidate stencil
    // reads up to ir_cloud cells into neighboring octants.
    GhostCommunicator ghost_comm( foreach_cell.get_amr_mesh(), U.getShape(), params.ir_cloud );
    ghost_comm.exchange_ghosts( Uin );

    UserData::FieldAccessor Ug;
    if( params.check_energies )
    {
      DYABLO_ASSERT_HOST_RELEASE( U.has_field("gx"),
        "sink/check_energies=true needs the gravity acceleration fields gx,gy,gz (enable [gravity], or set sink/check_energies=false)" );
      Ug = U.getAccessor( { {"gx",IGX}, {"gy",IGY}, {"gz",IGZ} } );
      // The time loop's own exchange may be shallower than ir_cloud
      ghost_comm.exchange_ghosts( Ug );
    }

    const MpiComm comm = foreach_cell.get_amr_mesh().getMpiComm();

    GlobalSinkTable table = build_global_sink_table( U, foreach_particle, comm, params.family );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    const SinkFormationCtx ctx{
      Uin, Ug, cells,
      table.device, table.n_tot,
      policy,
      foreach_cell.getDim(),
      (int)params.level_max,
      params.ir_cloud,
      rho_sink,
      ( 2*params.r_acc ) * ( 2*params.r_acc ),
      four_pi_G / ( 4.0 * PI ),
      policy_params.gamma0,
      params.check_energies,
      params.xmax - params.xmin, params.ymax - params.ymin, params.zmax - params.zmin,
      params.periodic
    };

    // -------------------------------------------------------------------
    // Pass 1 : count local candidate cells
    uint32_t n_cand_loc32 = 0;
    foreach_cell.reduce_cell( "count_sink_candidates", U.getShape(),
      CELL_LAMBDA(const ForeachCell::CellIndex& iCell, uint32_t& count)
    {
      count += sink_candidate_check( ctx, iCell ) ? 1 : 0;
    }, n_cand_loc32);
    const int n_cand_loc = n_cand_loc32;

    // -------------------------------------------------------------------
    // Pass 2 : store candidate cells + payload for the global resolution
    enum CandCol : int { CAND_X=0, CAND_Y=1, CAND_Z=2, CAND_RHO=3, CAND_NCOL=4 };
    Kokkos::View< ForeachCell::CellIndex* > cand_cells( "sink_candidate_cells", n_cand_loc );
    Kokkos::View< real_t** > cand_data( "sink_candidate_data", n_cand_loc, (int)CAND_NCOL );
    Kokkos::View<uint32_t> cand_counter( "sink_candidate_counter" );
    foreach_cell.foreach_cell( "store_sink_candidates", U.getShape(),
      CELL_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
      if( sink_candidate_check( ctx, iCell ) )
      {
        uint32_t id = Kokkos::atomic_fetch_add( &cand_counter(), 1 );
        cand_cells(id) = iCell;
        const auto pos = ctx.cells.getCellCenter( iCell );
        cand_data(id, CAND_X)   = pos[IX];
        cand_data(id, CAND_Y)   = pos[IY];
        cand_data(id, CAND_Z)   = pos[IZ];
        cand_data(id, CAND_RHO) = ctx.Uin.at( iCell, ConsState::VarIndex::Irho );
      }
    });

    // -------------------------------------------------------------------
    // Gather all candidates on every rank (same recipe as the sink table)
    int n_cand_incl = 0;
    comm.MPI_Scan( &n_cand_loc, &n_cand_incl, 1, MpiComm::MPI_Op_t::SUM );
    int n_cand_tot = 0;
    comm.MPI_Allreduce( &n_cand_loc, &n_cand_tot, 1, MpiComm::MPI_Op_t::SUM );
    const int cand_offset = n_cand_incl - n_cand_loc;

    std::vector<real_t> cand_glob( (size_t)n_cand_tot * CAND_NCOL, 0.0 );
    if( n_cand_loc > 0 )
    {
      auto cand_data_host = Kokkos::create_mirror_view( cand_data );
      Kokkos::deep_copy( cand_data_host, cand_data );
      for( int i=0; i<n_cand_loc; i++ )
        for( int c=0; c<(int)CAND_NCOL; c++ )
          cand_glob[ (size_t)(cand_offset + i)*CAND_NCOL + c ] = cand_data_host(i,c);
    }
    if( n_cand_tot > 0 )
      comm.MPI_Allgatherv_inplace( cand_glob.data(), n_cand_loc * (int)CAND_NCOL );

    // -------------------------------------------------------------------
    // Deterministic host resolution, identical on every rank :
    // sweep candidates by (rho desc, x,y,z asc) and greedy-accept those at least
    // 2*r_acc away from every existing sink and every previously accepted site.
    std::vector<int>    accepted( n_cand_tot, 0 );
    std::vector<real_t> assigned_id( n_cand_tot, 0 );
    int n_accept_glob = 0;
    if( n_cand_tot > 0 )
    {
      auto cand = [&]( int gi, CandCol c ) -> real_t { return cand_glob[ (size_t)gi*CAND_NCOL + c ]; };

      std::vector<int> order( n_cand_tot );
      std::iota( order.begin(), order.end(), 0 );
      std::sort( order.begin(), order.end(), [&]( int a, int b )
      {
        if( cand(a,CAND_RHO) != cand(b,CAND_RHO) ) return cand(a,CAND_RHO) > cand(b,CAND_RHO);
        if( cand(a,CAND_X)   != cand(b,CAND_X) )   return cand(a,CAND_X)   < cand(b,CAND_X);
        if( cand(a,CAND_Y)   != cand(b,CAND_Y) )   return cand(a,CAND_Y)   < cand(b,CAND_Y);
        return cand(a,CAND_Z) < cand(b,CAND_Z);
      });

      // ids are recovered from the existing sinks : restart-safe, no persistent counter
      real_t id_base = 1;
      for( int i=0; i<table.n_tot; i++ )
        id_base = std::max( id_base, table.at(i, GlobalSinkTable::ID) + 1 );

      const real_t Lx = params.xmax - params.xmin;
      const real_t Ly = params.ymax - params.ymin;
      const real_t Lz = params.zmax - params.zmin;
      const real_t excl2 = ( 2*params.r_acc ) * ( 2*params.r_acc );
      auto dist2 = [&]( real_t x1, real_t y1, real_t z1, real_t x2, real_t y2, real_t z2 )
      {
        const real_t rx = sink_wrapped_delta( x1, x2, Lx, params.periodic[IX] );
        const real_t ry = sink_wrapped_delta( y1, y2, Ly, params.periodic[IY] );
        const real_t rz = sink_wrapped_delta( z1, z2, Lz, params.periodic[IZ] );
        return rx*rx + ry*ry + rz*rz;
      };

      std::vector<int> accepted_list;
      for( int gi : order )
      {
        const real_t x = cand(gi,CAND_X), y = cand(gi,CAND_Y), z = cand(gi,CAND_Z);
        bool ok = true;
        for( int i=0; i<table.n_tot && ok; i++ )
          if( dist2( x,y,z, table.at(i,GlobalSinkTable::X), table.at(i,GlobalSinkTable::Y), table.at(i,GlobalSinkTable::Z) ) < excl2 )
            ok = false;
        for( int ga : accepted_list )
        {
          if( !ok ) break;
          if( dist2( x,y,z, cand(ga,CAND_X), cand(ga,CAND_Y), cand(ga,CAND_Z) ) < excl2 )
            ok = false;
        }
        if( ok )
        {
          accepted[gi] = 1;
          assigned_id[gi] = id_base + n_accept_glob;
          n_accept_glob++;
          accepted_list.push_back( gi );
        }
      }
    }

    // -------------------------------------------------------------------
    // Pass 3 : spawn accepted local candidates and deplete their host cell
    U.new_ParticleArray( "spawned_sinks", n_cand_loc );
    for( const char* attr : { "mass", "vx", "vy", "vz", "id", "birth_time", "dm_acc" } )
      U.new_ParticleAttribute( "spawned_sinks", attr );

    { // scope guard important to avoid keeping references to "spawned_sinks" array
      Kokkos::View<int*>    accept_d( "sink_accept", n_cand_loc );
      Kokkos::View<real_t*> id_d( "sink_new_id", n_cand_loc );
      {
        auto accept_h = Kokkos::create_mirror_view( accept_d );
        auto id_h = Kokkos::create_mirror_view( id_d );
        for( int i=0; i<n_cand_loc; i++ )
        {
          accept_h(i) = accepted[ cand_offset + i ];
          id_h(i) = assigned_id[ cand_offset + i ];
        }
        Kokkos::deep_copy( accept_d, accept_h );
        Kokkos::deep_copy( id_d, id_h );
      }

      enum VarIndex_particle{ IMASS, IVX, IVY, IVZ, IID, IBIRTH_TIME, IDMACC };
      UserData::ParticleAccessor Pnew_data = U.getParticleAccessor( "spawned_sinks", {
        {"mass", IMASS},
        {"vx", IVX},
        {"vy", IVY},
        {"vz", IVZ},
        {"id", IID},
        {"birth_time", IBIRTH_TIME},
        {"dm_acc", IDMACC}
      } );
      auto Pnew = U.getParticleArray( "spawned_sinks" );

      std::vector<UserData::FieldAccessor::FieldInfo> passive_fields;
      for (int i = 0; i < n_passive_scalars; ++i)
        passive_fields.push_back( {"rho_scalar_" + std::to_string(i), i} );
      UserData::FieldAccessor Uin_passive;
      if (!passive_fields.empty())
        Uin_passive = U.getAccessor( passive_fields );

      const real_t c_acc = params.c_acc;

      foreach_particle.foreach_particle( "fill_spawned_sinks", Pnew,
        KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
      {
        if( accept_d(iPart) == 0 )
        {
          Pnew_data.at(iPart, IMASS) = 0; // mask : not merged into the family
          return;
        }

        const auto iCell = cand_cells(iPart);
        const auto cell_size = cells.getCellSize( iCell );
        const auto cell_pos = cells.getCellCenter( iCell );
        const real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];

        auto q = policy.consToPrim( policy.getConsState( Uin, iCell ) );

        // Seed from the peak cell, never more than the accretable excess
        const real_t m_seed_max = c_acc * ( q.rho - rho_sink ) * Vcell;
        const real_t m_seed = ( mass_seed_cfg > 0 ) ? FMIN( mass_seed_cfg, m_seed_max ) : m_seed_max;

        Pnew.pos(iPart, IX) = cell_pos[IX];
        Pnew.pos(iPart, IY) = cell_pos[IY];
        Pnew.pos(iPart, IZ) = cell_pos[IZ];
        Pnew_data.at(iPart, IMASS) = m_seed;
        Pnew_data.at(iPart, IVX) = q.u;
        Pnew_data.at(iPart, IVY) = q.v;
        Pnew_data.at(iPart, IVZ) = q.w;
        Pnew_data.at(iPart, IID) = id_d(iPart);
        Pnew_data.at(iPart, IBIRTH_TIME) = time;
        Pnew_data.at(iPart, IDMACC) = 0;

        // Deplete the host cell at its local gas state (1:1 cell<->candidate, so no
        // atomics needed) : density-proportional scaling of every per-volume
        // quantity removes mass at the local velocity and specific energy.
        const real_t rho_new = q.rho - m_seed / Vcell; 
        const real_t factor = rho_new / q.rho;
        Uin.at( iCell, ConsState::VarIndex::Irho )    = rho_new;
        Uin.at( iCell, ConsState::VarIndex::Irho_vx ) *= factor;
        Uin.at( iCell, ConsState::VarIndex::Irho_vy ) *= factor;
        Uin.at( iCell, ConsState::VarIndex::Irho_vz ) *= factor;
        Uin.at( iCell, ConsState::VarIndex::Ie_tot )  *= factor;
        Uin.at( iCell, ConsState::VarIndex::Ie_int )  *= factor;
        for (int ivar = 0; ivar < Uin_passive.nbFields(); ++ivar)
          Uin_passive.at_ivar(iCell, ivar) *= factor;
      });
    }

    U.merge_particles_if( params.family, "spawned_sinks", "mass" );

    if( n_accept_glob > 0 && comm.MPI_Comm_rank() == 0 )
      std::cout << "Formed " << n_accept_glob << " sink particles" << std::endl;

    timers.get("ParticleUpdate_sink_formation").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;
  Policy_Params policy_params;
  SinkParams params;
  bool cosmology;
  int n_passive_scalars;
  real_t four_pi_G;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_sink_formation,
                  "ParticleUpdate_sink_formation")

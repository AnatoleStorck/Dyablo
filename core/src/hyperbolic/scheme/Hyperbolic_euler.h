#pragma once

#include <type_traits>

#include "HyperbolicUpdate_base.h"
#include "mpi/GhostCommunicator_partial_blocks.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {
namespace{
using CellIndex     = ForeachCell::CellIndex;
using FieldAccessor = UserData::FieldAccessor;
using offset_t      = typename CellIndex::offset_t;
using PatchArray = ForeachCell::CellArray_patch;

}// namespace
}// namespace dyablo

namespace dyablo {

namespace impl{
namespace {
template <typename T, typename = int, typename=int>
struct HasDensityAndPressure : std::false_type { };

template <typename T>
struct HasDensityAndPressure <T, decltype((void) T::rho, 0), decltype((void) T::p, 0)> : std::true_type { };
}
}

/**
 * @brief Euler update algorithm
 * 
 * @tparam State the type of state to treat
 */
template<typename Policy>
class Hyperbolic_euler : public HyperbolicUpdate {
  static_assert( is_HyperbolicPolicy_v<Policy>,
  "Policy must be wrapped in HyperbolicPolicy_base");

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

public:
  Hyperbolic_euler(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    timers(timers),
    policy_params(Policy::getParams(configMap)),
    ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallp( configMap.getValue<real_t>("hydro","smallp", 1e-10) ),
    slope_enabled( configMap.getValue<bool>("hydro","slope_enabled", true) ),
    dual_energy( configMap.getValue<bool>("hydro","dual_energy", false) ),
    dual_energy_eta( configMap.getValue<real_t>("hydro","dual_energy_eta", 0.5) )
  { }

  /**
   * @brief Solves hydro for one step using the euler method
   * 
   * @param U the input/output global array
   * @param scalar_data input scalar data
   */
  void update( UserData& U, ScalarSimulationData& scalar_data)
  {
    real_t dt = scalar_data.get<real_t>("dt");
    int ndim = this->ndim;
    bool slope_enabled = this->slope_enabled;
    bool dual_energy = this->dual_energy;
    real_t dual_energy_eta = this->dual_energy_eta;
    real_t gamma0 = this->gamma0;
    real_t smallp = this->smallp;

    const Policy policy( this->policy_params, scalar_data ); 
    Timers& timers = this->timers; 
    ForeachCell& foreach_cell = this->foreach_cell;

    FieldAccessor Uin = policy.getUin(U);
    FieldAccessor Uout = policy.getUout(U);
    
    timers.get("HyperbolicUpdate_euler").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    ForeachCell::SearchMode_neighbor search_neighbor( this->foreach_cell.get_amr_mesh().getLightOctree(), ForeachCell::SearchMode_neighbor::ORIGIN );
    ForeachCell::SearchMode_local search_local( ForeachCell::SearchMode_local::ASSERT );

    // Initializing output array 
    // TODO : remove this and copy Uin->Uout in timeloop or field creation logic
    foreach_cell.foreach_cell( "Hyperbolic_euler::init",
      Uout.getShape(),
      KOKKOS_LAMBDA(const CellIndex &iCell) 
    {
      ConsState uC = policy.getConsState(Uin, iCell);
      policy.setConsState(Uout, iCell, uC);
    });

    // Setting the ghosts to 0 to accumulate fluxes
    foreach_cell.foreach_ghost_cell( "Hyperbolic_euler::resetting_ghosts",
      Uout.getShape(),
      KOKKOS_LAMBDA(const CellIndex &iCell) 
    {
      ConsState empty_state{};
      policy.setConsState(Uout, iCell, empty_state);
    });

    int nb_ghosts = slope_enabled ? 2 : 1;
    PatchArray::Ref Qpatch_ = foreach_cell.reserve_patch_tmp("Qpatch", nb_ghosts, nb_ghosts, (ndim == 3)?nb_ghosts:0, State_traits<PrimState>::nvars);

    foreach_cell.foreach_patch( "Hyperbolic_euler::update", 
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Qpatch = patch.allocate_tmp(Qpatch_);

      patch.foreach_cell( Qpatch, 
        CELL_LAMBDA( const CellIndex& iCell_Qpatch )
      {
        ForeachCell::SearchMode_neighbor search_neighbor_origin( cellmetadata.getLightOctree(), ForeachCell::SearchMode_neighbor::ORIGIN );
        CellIndex iCell_Uin = Uin.getShape().convert_index(iCell_Qpatch, search_neighbor_origin);
        int level_diff = iCell_Uin.level_diff();
        ConsState u{};
        if (iCell_Uin.is_boundary())
          u = policy.getBoundaryValue(Uin, iCell_Uin, cellmetadata);
        else if (level_diff < 0) {
          int subcell_count = 
          foreach_sibling(ndim, iCell_Uin, search_neighbor_origin,
            [&](const CellIndex& iCell_neigh) {
              ConsState uloc = policy.getConsState(Uin, iCell_neigh);
              u += uloc;
            });
          u /= subcell_count;
        }
        else
          u = policy.getConsState(Uin, iCell_Uin);
        
        const PrimState q = policy.consToPrim( u );
        policy.setPrimState( Qpatch, iCell_Qpatch, q );
      });

      patch.foreach_cell( Uout.getShape(), 
        KOKKOS_LAMBDA( const CellIndex& iCell_Uout )
      {
        ForeachCell::SearchMode_neighbor search_neighbor( cellmetadata.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

        // Return Slope at position iCell
        auto get_slope = [&](const CellIndex &iCell_Uin, const CellIndex &iCell_Qpatch, ComponentIndex3D dir) 
        {
          if(!slope_enabled)
            return PrimState{};

          const PrimState qC = policy.getPrimState(Qpatch, iCell_Qpatch );
          offset_t off_m{}; off_m[dir] = -1;
          const PrimState qL = policy.getPrimState(Qpatch, iCell_Qpatch + off_m); 
          offset_t off_p{}; off_p[dir] =  1;
          const PrimState qR = policy.getPrimState(Qpatch, iCell_Qpatch + off_p); 
        
          //!\ Neighbor cells in Qpatch are averaged cells -> size iCell_L != size iCell_Qpatch_L
          CellIndex iCell_L = iCell_Uin.getNeighbor(off_m, search_neighbor);
          CellIndex iCell_R = iCell_Uin.getNeighbor(off_p, search_neighbor);   

          // Getting the length right and left
          // Smaller -> use averaged same-size cell -> 1*dx
          // Bigger -> same-size cell in Qpatch has vame value as actual bigger cell -> dx/2 + dx             
          constexpr real_t sizes[] = {1.0, 1.0, 1.5}; 
          const real_t dL = sizes[iCell_L.level_diff()+1];
          const real_t dR = sizes[iCell_R.level_diff()+1];  

          // Computing minmod slope for the direction
          PrimState slope = policy.compute_slope( qL, qC, qR, dL, dR);
          return slope;
        }; // get_slope

        auto normal_component = [](const auto &q, ComponentIndex3D dir) -> real_t {
          return dir==IX ? q.u : (dir==IY ? q.v : q.w);
        };
        // Also return contact velocity for dual-energy
        auto solve_riemann = [&](const PrimState &a, const PrimState &b, ComponentIndex3D dir, real_t &ustar) {
          if constexpr ( Policy::has_dual_energy() )
            return policy.riemann_solver(a, b, dir, ustar);
          else
            return policy.riemann_solver(a, b, dir);
        };

        auto process_dir = [&](const CellIndex &iCell_Uin, const CellIndex &iCell_Qpatch, ComponentIndex3D dir) {
          // Getting centered value and slope
          PrimState qC0 = policy.getPrimState( Qpatch, iCell_Qpatch );
          PrimState slope_C = get_slope(iCell_Uin, iCell_Qpatch, dir);
          real_t size_C = cellmetadata.getCellSize(iCell_Uin)[dir];

          real_t dim_fac = (ndim == 2 ? 0.5 : 0.25);
          real_t ustarL = 0, ustarR = 0;

          // Compute left side flux
          ConsState fluxL {};
          {
            PrimState qC = qC0 - 0.5 * slope_C;
            if constexpr ( Policy::has_dual_energy() )
              ustarL = normal_component(qC, dir); // fallback (boundary / finer neighbor faces)

            offset_t off_m{};
            off_m[dir] = -1;
            const CellIndex iCell_Uin_m = iCell_Uin.getNeighbor(off_m, search_neighbor);
            if( iCell_Uin_m.is_boundary() )
            {
              fluxL = policy.getBoundaryFlux(Uin, iCell_Uin_m, qC, cellmetadata);
            }
            else
            {
              int Ldiff = iCell_Uin_m.level_diff();
              if (Ldiff >= 0)
              {
                CellIndex iCell_Qpatch_m = iCell_Qpatch + off_m;
                PrimState qL0 = policy.getPrimState( Qpatch, iCell_Qpatch_m );
                PrimState slope_L = get_slope(iCell_Uin_m, iCell_Qpatch_m, dir);
                real_t size_L = cellmetadata.getCellSize(iCell_Uin_m)[dir];

                // Reconstructing
                PrimState qL = qL0 + 0.5 * slope_L;

                // Solving
                fluxL = solve_riemann(qL, qC, dir, ustarL);

                // Adding flux to the neighbor if it is bigger
                if (Ldiff == 1)
                {
                  ConsState du_n = fluxL * - dim_fac * dt / size_L;
                  policy.atomic_addConsState(Uout, iCell_Uin_m, du_n);
                }
              } // If smaller we skip
            }
          }

          // Compute right side flux
          ConsState fluxR {};
          {
            PrimState qC = qC0 + 0.5 * slope_C;
            if constexpr ( Policy::has_dual_energy() )
              ustarR = normal_component(qC, dir); // fallback (boundary / finer neighbor faces)

            offset_t off_p{};
            off_p[dir] = 1;
            const CellIndex iCell_Uin_p = iCell_Uin.getNeighbor(off_p, search_neighbor);
            if( iCell_Uin_p.is_boundary() )
            {
              fluxR = policy.getBoundaryFlux(Uin, iCell_Uin_p, qC, cellmetadata);
            }
            else
            {
              int Rdiff = iCell_Uin_p.level_diff();
              if (Rdiff >= 0)
              {
                CellIndex iCell_Qpatch_p = iCell_Qpatch + off_p;
                PrimState qR0 = policy.getPrimState( Qpatch, iCell_Qpatch_p );
                PrimState slope_R = get_slope(iCell_Uin_p, iCell_Qpatch_p, dir);
                real_t size_R = cellmetadata.getCellSize(iCell_Uin_p)[dir];

                // Reconstructing
                PrimState qR = qR0 - 0.5 * slope_R;

                // Solving
                fluxR = solve_riemann(qC, qR, dir, ustarR);

                // Adding flux to the neighbor if it is bigger
                if (Rdiff == 1)
                {
                  ConsState du_n = fluxR * dim_fac * dt / size_R;
                  policy.atomic_addConsState(Uout, iCell_Uin_p, du_n);
                }
              }
            }
          }

          ConsState du = (fluxL-fluxR) * dt / size_C;
          // Add the non-conservative source term -P div(u) * dt.
          if constexpr ( Policy::has_dual_energy() )
          {
            if( dual_energy )
              du.e_int -= qC0.p * (ustarR - ustarL) * dt / size_C;
          }
          return du;
        };


        CellIndex iCell_Qpatch = Qpatch.getShape().convert_index(iCell_Uout, search_local);

        ConsState du{};
        du += process_dir(iCell_Uout, iCell_Qpatch, IX);
        du += process_dir(iCell_Uout, iCell_Qpatch, IY);
        if (ndim == 3)
           du += process_dir(iCell_Uout, iCell_Qpatch, IZ);
        policy.atomic_addConsState(Uout, iCell_Uout, du);
      });
    });
      

    // Reducing the ghosts to accumulate the flux in the data arrays 
    int ghost_count = 1;
    GhostCommunicator_partial_blocks ghost_comm ( 
      foreach_cell.get_amr_mesh(),
      Uout.getShape(),
      ghost_count );
    ghost_comm.reduce_ghosts( Uout );
    
    if constexpr ( Policy::has_dual_energy() )
    {
      // Dual-energy finalization (Teyssier 2015, sec. 6.6) :
      //  - e_cons  = E_tot - E_kin  (conservative internal energy, large truncation error
      //              in cold supersonic flows but reproduces shock/reconnection heating)
      //  - e_prim  = the non-conservatively-evolved internal energy carried in e_int
      //  - e_trunc = 0.5 rho (du)^2 estimate of the local truncation error (eq. 158)
      // Pick e_cons when it exceeds eta * e_trunc, otherwise keep e_prim and reset E_tot
      foreach_cell.foreach_cell( "HyperbolicUpdate_euler::dual_energy_finalize", Uout.getShape(),
        KOKKOS_LAMBDA(  const ForeachCell::CellIndex& iCell)
      {
        ConsState u = policy.getConsState(Uout, iCell);
        if constexpr ( Policy::has_postProcess() )
          u = policy.postProcess( u );

        const real_t inv_rho = (u.rho != 0 ? 1.0/u.rho : 0.0);
        const real_t ekin = 0.5 * (u.rho_u*u.rho_u + u.rho_v*u.rho_v + u.rho_w*u.rho_w) * inv_rho;
        const real_t e_cons = u.e_tot - ekin;
        const real_t e_int_min = smallp / (gamma0 - 1.0);

        real_t e_final;
        if( dual_energy ) { // Do the truncation error estimate

          ForeachCell::SearchMode_neighbor search_neighbor( cellmetadata.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );
          auto vel_comp = [&]( const ConsState& s, ComponentIndex3D dir ) -> real_t {
            const real_t ir = (s.rho != 0 ? 1.0/s.rho : 0.0);
            return (dir==IX ? s.rho_u : (dir==IY ? s.rho_v : s.rho_w)) * ir;
          };

          const ConsState u_old = policy.getConsState(Uin, iCell);
          const real_t rho_old = u_old.rho;

          real_t du2 = 0;
          auto accumulate_dir = [&]( ComponentIndex3D dir ) {
            const real_t vC = vel_comp(u_old, dir);
            offset_t off_m{}; off_m[dir] = -1;
            offset_t off_p{}; off_p[dir] =  1;
            const CellIndex iCell_m = iCell.getNeighbor(off_m, search_neighbor);
            const CellIndex iCell_p = iCell.getNeighbor(off_p, search_neighbor);
            real_t dm = 0, dp = 0;
            if( !iCell_m.is_boundary() )
              dm = fabs( vC - vel_comp(policy.getConsState(Uin, iCell_m), dir) );
            if( !iCell_p.is_boundary() )
              dp = fabs( vel_comp(policy.getConsState(Uin, iCell_p), dir) - vC );
            const real_t du = fmax(dm, dp);
            du2 += du*du;
          };
          accumulate_dir(IX);
          accumulate_dir(IY);
          if( ndim == 3 )
            accumulate_dir(IZ);

          const real_t e_trunc = 0.5 * rho_old * du2;

          if( e_cons > dual_energy_eta * e_trunc )
          {
            // Conservative branch : trust E_tot (shocks, reconnection layers, smooth flow).
            e_final = fmax(e_cons, e_int_min);
          }
          else
          {
            // Non-conservative branch : keep e_prim and reset E_tot = E_kin + e_prim.
            e_final = fmax(u.e_int, e_int_min);
            u.e_tot = ekin + e_final;
          }
        }
        else {
          // Dual energy disabled
          e_final = fmax(e_cons, e_int_min);
        }

        u.e_int = e_final;
        policy.setConsState( Uout, iCell, u );
      });
    }
    else if constexpr ( Policy::has_postProcess() )
    {
      foreach_cell.foreach_cell( "HyperbolicUpdate::post-process", Uout.getShape(),
        KOKKOS_LAMBDA(  const ForeachCell::CellIndex& iCell)
      {
        ConsState u = policy.getConsState(Uout, iCell);
        ConsState u_pp = policy.postProcess( u );
        policy.setConsState( Uout, iCell, u_pp );
      });
    }

    policy.printWarnings();

    Kokkos::fence();
    timers.get("HyperbolicUpdate_euler").stop();
  }

protected:
  ForeachCell& foreach_cell;
  
  Timers& timers;  
  typename Policy::Params policy_params;

  int ndim;
  real_t gamma0;
  real_t smallr, smallp;
  bool slope_enabled;
  bool dual_energy;
  real_t dual_energy_eta;
};

} // namespace dyablo


#pragma once

#include "ForeachParticle.h"
#include "ParticleFamilies.h"
#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"
#include "utils/mpi/MpiComm.h"
#include "utils/units/Units.h"

#include <vector>

namespace dyablo {

/***
 * Shared infrastructure for the sink-particle modules
 * (ParticleUpdate_sink_formation / _sink_accretion / _sink_merging).
 *
 * Sink particles are ordinary domain-decomposed Dyablo particles (family
 * [sink]sink_family). Global knowledge (creation proximity checks, merging)
 * goes through a transient replicated snapshot table built with
 * MPI_Allgatherv (see GlobalSinkTable below).
 ***/

/// Runtime parameters of the sink-particle model, parsed from the [sink] section.
struct SinkParams
{
  std::string family;             //!< particle family name for sinks
  real_t rho_sink_physical;       //!< creation threshold and accretion floor (physical code units)
  int    ir_cloud;                //!< accretion radius in units of dx at level_max (RAMSES ir_cloud)
  real_t c_acc;                   //!< fraction of the density excess accreted per step (RAMSES c_acc)
  real_t mass_sink_seed_physical; //!< sink seed mass; 0 => seed from the peak-cell excess
  bool   check_energies;          //!< virial gate over the accretion sphere (needs [gravity])
  real_t merge_distance_cells;    //!< merging radius in units of dx at level_max

  uint32_t level_max;             //!< [amr] level_max
  real_t dx_min;                  //!< cell size at level_max
  real_t r_acc;                   //!< accretion radius = ir_cloud * dx_min

  real_t xmin, xmax, ymin, ymax, zmin, zmax;
  Kokkos::Array<bool,3> periodic;

  static SinkParams from_configMap( ConfigMap& configMap, ForeachCell& foreach_cell )
  {
    SinkParams p;
    p.family              = configMap.getValue<std::string>("sink", "sink_family", "sinks");
    p.rho_sink_physical   = configMap.getValue_in_code_unit<Units::Density>("sink", "rho_sink", "1e10 proton_mass/cm**3");
    p.ir_cloud            = configMap.getValue<int>("sink", "ir_cloud", 4);
    p.c_acc               = configMap.getValue<real_t>("sink", "c_acc", 0.75);
    p.mass_sink_seed_physical = configMap.getValue_in_code_unit<Units::Mass>("sink", "mass_sink_seed", "0 solar_mass");
    p.check_energies      = configMap.getValue<bool>("sink", "check_energies", true);
    p.merge_distance_cells = configMap.getValue<real_t>("sink", "merge_distance_cells", 2.0);

    p.level_max = configMap.getValue<uint32_t>("amr", "level_max");

    p.xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0);
    p.xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0);
    p.ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
    p.ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
    p.zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);
    p.zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);

    AMRmesh& pmesh = foreach_cell.get_amr_mesh();
    p.periodic = { pmesh.getPeriodic(IX), pmesh.getPeriodic(IY), pmesh.getPeriodic(IZ) };

    const Kokkos::Array<uint32_t, 3> bs = foreach_cell.blockSize();
    const real_t min_dx = (p.xmax-p.xmin) / ((1 << p.level_max) * bs[IX]);
    const real_t min_dy = (p.ymax-p.ymin) / ((1 << p.level_max) * bs[IY]);
    const real_t min_dz = (p.zmax-p.zmin) / ((1 << p.level_max) * bs[IZ]);
    DYABLO_ASSERT_HOST_RELEASE( min_dx == min_dy && min_dx == min_dz,
      "Sink particles only support cubic cells, but dx/dy/dz at level_max are "
      << min_dx << "/" << min_dy << "/" << min_dz );
    p.dx_min = min_dx;
    p.r_acc = p.ir_cloud * p.dx_min;

    DYABLO_ASSERT_HOST_RELEASE( foreach_cell.getDim() == 3,
      "Sink particles are only implemented in 3D" );

    // The accretion/energy-gate stencil can't be larger than the AMR block size
    // (CellIndex::getNeighbor beyond directly contiguous octants is undefined behavior)
    const uint32_t block_size_min = std::min( { bs[IX], bs[IY], bs[IZ] } );
    DYABLO_ASSERT_HOST_RELEASE( p.ir_cloud >= 1 && (uint32_t)p.ir_cloud <= block_size_min,
      "sink/ir_cloud (" << p.ir_cloud << ") must be between 1 and the AMR block size ("
      << block_size_min << ")" );
    DYABLO_ASSERT_HOST_RELEASE( p.c_acc > 0 && p.c_acc <= 1,
      "sink/c_acc (" << p.c_acc << ") must be in (0,1]" );

    return p;
  }
};

/// Per-axis minimum-image displacement a-b (periodic-aware)
KOKKOS_INLINE_FUNCTION
real_t sink_wrapped_delta( real_t a, real_t b, real_t L, bool periodic )
{
  real_t d = a - b;
  if( periodic )
  {
    if( d >  0.5*L ) d -= L;
    if( d < -0.5*L ) d += L;
  }
  return d;
}

/***
 * @brief Transient replicated snapshot of every sink in the simulation.
 *
 * Rows are ordered by (MPI rank, local particle index) : the local particles of
 * this rank are exactly rows [offset, offset+n_loc) in iPart order, so a local
 * particle iPart maps to table row offset+iPart as long as the local array is
 * not modified between build and use.
 ***/
struct GlobalSinkTable
{
  enum Col : int { ID=0, MASS=1, X=2, Y=3, Z=4, VX=5, VY=6, VZ=7, NCOL=8 };

  int n_tot = 0;      //!< global number of sinks
  int n_loc = 0;      //!< sinks owned by this rank
  int offset = 0;     //!< global row index of this rank's first sink
  std::vector<real_t> host;        //!< [n_tot*NCOL] row-major, identical on every rank
  Kokkos::View<real_t**> device;   //!< [n_tot][NCOL] copy for kernels

  real_t at( int i, Col c ) const { return host[ (size_t)i*NCOL + c ]; }
};

/***
 * @brief Build the replicated snapshot of every sink in the simulation.
 *
 * Collective : must be called by every rank.
 *
 * NOTE : defined in SinkUtils.cpp, NOT here. It launches a Kokkos kernel, and an
 * extended __device__ lambda may not live in a header function compiled into
 * several translation units : nvcc derives the lambda's type name from the
 * enclosing function's address plus a counter, so every TU emits the *same*
 * weak symbol for it while the per-TU function-pointer statics it uses stay
 * TU-local. The linker keeps one copy, and calls coming from the other TUs jump
 * through a null pointer.
 ***/
GlobalSinkTable build_global_sink_table(
    const UserData& U, const ForeachParticle& foreach_particle,
    const MpiComm& comm, const std::string& family );

/***
 * @brief Walk the sphere of radius R cells around iCell_center, in "slot" quadrature.
 *
 * Each offset (di,dj,dk) with |offset|^2 <= R^2 is one slot of nominal volume
 * V_center = dx_c^3 (the *center cell's* volume). apply(iCell_target, w) is
 * called once per (slot x target cell) with the fraction w of a slot the target
 * covers : a same-or-coarser neighbor gets w=1 (a coarse cell covered by k slots
 * is visited k times), a refined neighbor splits its slot between its 2^ndim
 * subcells (w=1/2^ndim each). Integrals over the sphere are therefore computed
 * with V_slot = w * V_center per call, without per-cell deduplication - the same
 * role RAMSES cloud particles play, without materializing them.
 *
 * Slots outside the domain are dropped. Slots in MPI ghost octants are kept only
 * when include_ghosts=true : reading ghosts is valid after a ghost exchange, and
 * writing is only valid into scratch fields that are reduce_ghosts'ed after.
 ***/
template< typename Apply >
KOKKOS_INLINE_FUNCTION
void foreach_sphere_slot( const ForeachCell::CellIndex& iCell_center,
                          int R, int ndim,
                          const ForeachCell::SearchMode_neighbor& search_neighbor,
                          bool include_ghosts,
                          const Apply& apply )
{
  const int R_k = (ndim == 3) ? R : 0;
  for( int dk = -R_k ; dk <= R_k ; dk++ )
  for( int dj = -R   ; dj <= R   ; dj++ )
  for( int di = -R   ; di <= R   ; di++ )
  {
    if( di*di + dj*dj + dk*dk > R*R ) continue;

    ForeachCell::CellIndex iCell_n = iCell_center.getNeighbor(
      { (int16_t)di, (int16_t)dj, (int16_t)dk }, search_neighbor );

    if( !iCell_n.is_valid() ) continue;
    if( !include_ghosts && iCell_n.iOct.isGhost ) continue;

    if( iCell_n.level_diff() >= 0 )
    {
      apply( iCell_n, (real_t)1 );
    }
    else
    { // Neighbor is refined : the slot is covered by its 2^ndim subcells
      const int sub_k_count = (ndim == 3) ? 2 : 1;
      const real_t w = 1.0 / ( 2 * 2 * sub_k_count );
      for( int16_t sk = 0 ; sk < sub_k_count ; sk++ )
      for( int16_t sj = 0 ; sj < 2           ; sj++ )
      for( int16_t si = 0 ; si < 2           ; si++ )
      {
        ForeachCell::CellIndex iCell_s = iCell_n.getNeighbor( {si,sj,sk}, search_neighbor );
        if( !iCell_s.is_valid() ) continue;
        if( !include_ghosts && iCell_s.iOct.isGhost ) continue;
        apply( iCell_s, w );
      }
    }
  }
}

} // namespace dyablo

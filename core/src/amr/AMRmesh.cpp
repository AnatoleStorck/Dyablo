#include "AMRmesh.h"

#include "LightOctree.h"
#include "mpi/ViewCommunicator.h"
#include "utils/config/ConfigMap.h"
#include "UserData.h"
#include <Kokkos_StdAlgorithms.hpp>

namespace dyablo{

using Storage_t = LightOctree_storage< Kokkos::DefaultHostExecutionSpace::memory_space >;
using logical_coord_t = Storage_t::logical_coord_t;
using level_t = Storage_t::level_t;
using oct_index_t = uint32_t;
using global_oct_index_t = uint64_t;
using morton_t = uint64_t;
using markers_t = Kokkos::View<int*, Storage_t::MemorySpace>;

struct AMRmesh::PData
{
  int ndim;
  uint8_t level_min, level_max;
  Kokkos::Array<bool,3> periodic;
  
  Storage_t storage; 

  MpiComm mpi_comm;
  global_oct_index_t total_num_octs, first_local_oct;

  GhostMap_t ghostmap;
  markers_t markers;

  std::unique_ptr<LightOctree> lmesh;
  int lmesh_epoch = 0;
  int pmesh_epoch = 1;
};

namespace{


KOKKOS_INLINE_FUNCTION
morton_t shift_level( const morton_t& m, int level_diff )
{
  if( level_diff >= 0 )
    return m << 3*level_diff;
  else
    return m >> -3*level_diff;
}

template <typename MemorySpace_t>
KOKKOS_INLINE_FUNCTION
morton_t compute_morton( const LightOctree_storage<MemorySpace_t>& octs, const oct_index_t& iOct_local, level_t target_level)
{
  auto pos = octs.get_logical_coords({iOct_local, false});
  level_t level = octs.getLevel({iOct_local, false});
  morton_t m = compute_morton_key( pos[IX], pos[IY], pos[IZ] );
  return shift_level( m, target_level-level );
}

// get first iOct_local with morton(iOct_local) >= morton
oct_index_t lower_bound_morton( const Storage_t& octs, morton_t morton, level_t target_level )
{
  oct_index_t nbOcts = octs.getNumOctants();

  oct_index_t begin = 0;
  oct_index_t end = nbOcts-1;
  while( begin < end )
  {
    oct_index_t pivot = begin + (end-begin)/2;
    morton_t morton_pivot = compute_morton(octs, pivot, target_level);
    if( morton_pivot == morton )
      return pivot;
    else if( morton_pivot < morton )
      begin = pivot+1;
    else //if( morton_pivot > morton )
      end = pivot;
  }
  return begin;
}

struct NeighborPair
{
  oct_index_t iOct_local;
  int rank_neighbor;
};

AMRmesh::GhostMap_t discover_ghosts(
  const LightOctree_storage<>& storage_device,
  const std::vector<morton_t>& morton_intervals_,
  level_t level_min, level_t level_max, 
  const Kokkos::Array<bool,3>& periodic,
  const MpiComm& mpi_comm)
{
  using GhostMap_t = AMRmesh::GhostMap_t;
  using CellMask = AMRmesh::GhostMap_t::CellMask;
  using Face = AMRmesh::GhostMap_t::Face;

  int mpi_rank = mpi_comm.MPI_Comm_rank();
  int mpi_size = mpi_comm.MPI_Comm_size();
  int ndim = storage_device.getNdim();

  // Copy morton_intervals to device
  DYABLO_ASSERT_HOST_DEBUG( morton_intervals_.size() == mpi_size+1, "morton_intervals_ should be of size mpi_size+1" );
  Kokkos::View<morton_t*> morton_intervals_device("discover_ghosts::morton_intervals", mpi_size+1);
  {
    auto morton_intervals_host = Kokkos::create_mirror_view( morton_intervals_device  );
    std::copy( morton_intervals_.begin(), morton_intervals_.end(), morton_intervals_host.data() );
    Kokkos::deep_copy( morton_intervals_device, morton_intervals_host );
  }

  uint32_t nbLeaves = storage_device.getNumOctants();
  uint32_t nbLocals = storage_device.getNumOctants() + storage_device.getNumIntermediates();
  
  size_t neighbor_count_guess = nbLocals;
  Kokkos::UnorderedMap< NeighborPair, CellMask > neighborMap( neighbor_count_guess );

  // Storage must be allocated before launching the kernel, 
  // but we don't know the number of ghosts to send :
  // When insert fails, we reallocate and restart from the first failed insert
  oct_index_t first_fail = 0;
  while( first_fail < nbLocals )
  {
    oct_index_t iter_start = first_fail;
    first_fail = nbLocals;
    Kokkos::parallel_reduce( "discover_ghosts", 
      Kokkos::RangePolicy<>( iter_start, nbLocals),
      KOKKOS_LAMBDA( oct_index_t iOct_i, oct_index_t& first_fail_local )
    {
      auto compute_morton = [level_max]( const Kokkos::Array<logical_coord_t, 3>& pos, level_t level )
      { 
        morton_t morton = compute_morton_key( pos[IX], pos[IY], pos[IZ] );
        morton = shift_level( morton, level_max-level );
        return morton;
      };

      auto find_rank = [mpi_size, &morton_intervals_device]( morton_t morton )
      { 
        // upper_bound : first verifying value > morton
        int rank;
        {
          int begin = 0;
          int end = mpi_size;
          while( begin < end )
          {
            int pivot = begin + (end-begin)/2;
            morton_t morton_pivot = morton_intervals_device(pivot);
            if( morton_pivot <= morton )
              begin = pivot+1;
            else //if( morton_pivot > morton )
              end = pivot;
          }
          // Just before the upper bound
          rank=begin-1;
        }
        DYABLO_ASSERT_KOKKOS_DEBUG( rank<mpi_size, "Rank not found for morton " );
        DYABLO_ASSERT_KOKKOS_DEBUG( morton_intervals_device(rank) <= morton && morton < morton_intervals_device(rank+1), 
                            "morton not within local interval" );
        return rank;
      };

      // Register current octant as ghost for neighbor_rank, aggregate masks for faces
      auto register_neighbor = [mpi_rank, iOct_i, &neighborMap, &first_fail_local]( int neighbor_rank, Face face )
      {
        if(neighbor_rank != mpi_rank)
        {
          CellMask faceMask = 1 << face;
          auto insert_result = neighborMap.insert( NeighborPair{iOct_i, neighbor_rank }, faceMask );
          if( insert_result.existing() )
          {
            int it = neighborMap.find(NeighborPair{iOct_i, neighbor_rank });
            Kokkos::atomic_or( &neighborMap.value_at( it ), faceMask );
          }
          else if( insert_result.failed() )
            first_fail_local = first_fail_local < iOct_i ? first_fail_local : iOct_i;
        }
      };

      auto getiOct = [nbLeaves]( oct_index_t iOct_i )
      {
        bool intermediate = false;
        if( iOct_i >= nbLeaves )
        {
          iOct_i -= nbLeaves;
          intermediate = true;
        }
        return LightOctree::OctantIndex{
          .iOct = iOct_i,
          .isGhost = false,
          .isIntermediate = intermediate,
        };
      };

      auto iOct = getiOct(iOct_i);

      Kokkos::Array<logical_coord_t, 3> pos = storage_device.get_logical_coords( iOct );
      level_t level = storage_device.getLevel( iOct );

      DYABLO_ASSERT_KOKKOS_DEBUG(find_rank(compute_morton(pos, level)) == mpi_rank, "Expected local rank but found remote instead");

      logical_coord_t max_ix = storage_device.cell_count(IX, level);
      logical_coord_t max_iy = storage_device.cell_count(IY, level);
      logical_coord_t max_iz = storage_device.cell_count(IZ, level);
      int dz_max = (ndim == 2)? 0:1;
      for( int dz=-dz_max; dz<=dz_max; dz++ )
      for( int dy=-1; dy<=1; dy++ )
      for( int dx=-1; dx<=1; dx++ )
      // If neighbor inside domain
      if(    (periodic[IX] || ( /* 0<=pos[IX]+dx && */ pos[IX]+dx<max_ix ))
          && (periodic[IY] || ( /* 0<=pos[IY]+dy && */ pos[IY]+dy<max_iy ))
          && (periodic[IZ] || ( /* 0<=pos[IZ]+dz && */ pos[IZ]+dz<max_iz )) )
      {
        if( dx==0 && dy==0 && dz==0 ) 
        {
          // Discover as child
          // Local Octant is a ghost for the rank owning its parent
          if( level > level_min ) // No parent at level_min
          {
            Kokkos::Array<logical_coord_t, 3> pos_parent{
              (pos[IX] >> 1), // remove one level
              (pos[IY] >> 1),
              (pos[IZ] >> 1),
            };
            morton_t m_parent = compute_morton( pos_parent, level-1 );
            int parent_rank = find_rank(m_parent);
            register_neighbor(parent_rank, Face::FULL_BLOCK);
          }

          // Discover as parent
          // Local Octant is a ghost for all ranks owning it's children
          if( iOct.isIntermediate ) // Leaves can't be parent
          {
            DYABLO_ASSERT_KOKKOS_DEBUG( level < level_max, "Internal error looking for children after level_max" );

            // TODO : skip when all suboctants are local, very common case
            // Note : at least one of the suboctants is local or this intermediate wouldn't exist
            Kokkos::Array<logical_coord_t, 3> pos_child_origin{
              (pos[IX] << 1), // add one level
              (pos[IY] << 1),
              (pos[IZ] << 1),
            };

            for( int16_t sz=0; sz<=1; sz++ )
            for( int16_t sy=0; sy<=1; sy++ )
            for( int16_t sx=0; sx<=1; sx++ )
            {
              Kokkos::Array<logical_coord_t, 3> pos_child{
                pos_child_origin[IX] + sx,
                pos_child_origin[IY] + sy,
                pos_child_origin[IZ] + sz,
              };
              morton_t m_suboctant = compute_morton( pos_child, level+1 );
              int child_rank = find_rank(m_suboctant);
              register_neighbor(child_rank, Face::FULL_BLOCK);
            }
          }
        }
        else 
        { 
          // Discover as neighbor
          // Local Octant is a ghost for all ranks owning it's neighbors
          Kokkos::Array<logical_coord_t, 3> pos_n{
            (pos[IX]+dx+max_ix)%max_ix, 
            (pos[IY]+dy+max_iy)%max_iy, 
            (pos[IZ]+dz+max_iz)%max_iz
          };
          morton_t morton_n = compute_morton( pos_n, level );
          int neighbor_rank = find_rank( morton_n );

          Face neighbor_face = Face::FACE_COUNT; // To avoid warnings, should be set in one of the if below
          if      ( dx==-1 ) neighbor_face = Face::XL; // Whole face is included for corners 
          else if ( dx== 1 ) neighbor_face = Face::XR; 
          else if ( dy==-1 ) neighbor_face = Face::YL;
          else if ( dy== 1 ) neighbor_face = Face::YR;
          else if ( dz==-1 ) neighbor_face = Face::ZL;
          else if ( dz== 1 ) neighbor_face = Face::ZR;
          else 
          {
            DYABLO_ASSERT_KOKKOS_DEBUG( false, "discover_ghosts : cannot determine face");
          }

          // Verify that the whole same-size virtual neighbor is owned by neighbor_rank
          // i.e : last suboctant of same-size neighbor is owned by the same MPI
          // TODO : get morton of last neighbor to filter even more
          morton_t morton_next = shift_level(morton_n, level-level_max) + 1;
                   morton_next = shift_level(morton_next, level_max-level) - 1;
            
          register_neighbor(neighbor_rank, neighbor_face);
          if(  !iOct.isIntermediate // if intermediate it has a matching intermediate neighbor with same size
            && level != level_max  // if it's already at level_max, neighbor can't be smaller
            && morton_next >= morton_intervals_device(neighbor_rank+1) // All suboctants are owned by the same process
          )
          {
            // Neighbors may be scattered between multiple MPIs  
            // Iterate over NEIGHBORS (not all suboctants) and send them to corresponding ranks
            
            // Apply offset to get smaller origin neighbor
            Kokkos::Array<logical_coord_t, 3> pos_n_smaller_origin{
              (pos_n[IX] << 1) + (dx == -1), // add one level + right half of suboctant if left of original cell
              (pos_n[IY] << 1) + (dy == -1),
              (pos_n[IZ] << 1) + (dz == -1),
            };

            // Iterate over neighbor suboctants
            int sx_max = (dx==0); // constrained to the same plane as origin if offset in this direction
            int sy_max = (dy==0);
            int sz_max = (ndim==2) ? 0 : (dz==0);
            for( int16_t sz=0; sz<=sz_max; sz++ )
            for( int16_t sy=0; sy<=sy_max; sy++ )
            for( int16_t sx=0; sx<=sx_max; sx++ )
            {
              Kokkos::Array<logical_coord_t, 3> pos_n_smaller{
                pos_n_smaller_origin[IX] + sx,
                pos_n_smaller_origin[IY] + sy,
                pos_n_smaller_origin[IZ] + sz,
              };
              morton_t m_suboctant = compute_morton( pos_n_smaller, level+1 );
              int neighbor_rank = find_rank(m_suboctant);
              register_neighbor(neighbor_rank, neighbor_face);
            }
          }
        }
      }
    }, Kokkos::Min<oct_index_t>(first_fail) ); 
  
    // first_fail should be std::numeric_limit<oct_index_t>::max() if every insert succeeded
    if( first_fail < nbLocals )
    {
      std::cout << "Ghost storage too small : rehash" << std::endl;
      neighbor_count_guess *= 2;
      neighborMap.rehash( neighbor_count_guess );
      std::cout << "Restart from iOct " << first_fail << std::endl;
    }
  }

  // Copy content of neighborMap to result variable
  AMRmesh::GhostMap_t to_send{};  
  {
    uint32_t nbLocalLeaves = storage_device.getNumOctants();

    auto& leaves = to_send.to_send_leaves;
    auto& intermediates = to_send.to_send_intermediates;

    leaves.send_sizes = Kokkos::View<uint32_t*>("discover_ghosts::leaves::sizes", mpi_size);
    intermediates.send_sizes = Kokkos::View<uint32_t*>("discover_ghosts::intermediates::sizes", mpi_size);
    Kokkos::parallel_for( "discover_ghosts::count_ghosts", neighborMap.capacity(),
      KOKKOS_LAMBDA( uint32_t i )
    {
      if( neighborMap.valid_at(i) )
      {
        const NeighborPair& p = neighborMap.key_at(i);
        if( p.iOct_local < nbLocalLeaves )
          Kokkos::atomic_inc(&leaves.send_sizes(p.rank_neighbor));
        else
          Kokkos::atomic_inc(&intermediates.send_sizes(p.rank_neighbor));
      }
    });

    uint32_t total_leaf_count = 0;
    uint32_t total_intermediate_count = 0;
    Kokkos::parallel_reduce( "discover_ghosts::count_total_ghosts", mpi_size,
      KOKKOS_LAMBDA(uint32_t i, uint32_t& leaf_count, uint32_t& intermediate_count)
    {
      leaf_count += leaves.send_sizes(i);
      intermediate_count += intermediates.send_sizes(i);
    }, total_leaf_count, total_intermediate_count );

    leaves.send_iOcts = Kokkos::View<uint32_t*>( "discover_ghosts::leaves::iOcts", total_leaf_count );
    leaves.send_cell_masks = Kokkos::View<CellMask*>( "discover_ghosts::leaves::cell_masks", total_leaf_count );
    intermediates.send_iOcts = Kokkos::View<uint32_t*>( "discover_ghosts::intermediates::iOcts", total_intermediate_count );
    intermediates.send_cell_masks = Kokkos::View<CellMask*>( "discover_ghosts::intermediates::cell_masks", total_intermediate_count );

    // Clean CellMasks values 
    // remove other faces when Face::FULL_BLOCK is present
    // ensures that resulting mask is < GhostMap_t::MASK_COUNT
    auto clean_mask = KOKKOS_LAMBDA(CellMask mask)
    {
      if( mask & (1 << Face::FULL_BLOCK) )
        mask = GhostMap_t::MASK_FULL_BLOCK;
      DYABLO_ASSERT_KOKKOS_DEBUG( mask < GhostMap_t::MASK_COUNT, "discover_ghosts : Mask out ouf bounds" );
      return mask;
    };

    uint32_t offset_first_leaf = 0;
    uint32_t offset_first_intermediate = 0;
    for( int rank=0; rank<mpi_size; rank++ )
    {
      uint32_t nbLeaves_rank = 0;
      Kokkos::parallel_scan( "discover_ghosts::fill_ghostmap_leaves", neighborMap.capacity(),
        KOKKOS_LAMBDA( uint32_t i, oct_index_t& offset_local, bool final)
      {
        if( neighborMap.valid_at(i) )
        {
          const NeighborPair& p = neighborMap.key_at(i);
          if( p.rank_neighbor == rank )
          {
            if( p.iOct_local < nbLocalLeaves )
            {
              if(final)
              {
                uint32_t offset = offset_first_leaf + offset_local;
                leaves.send_iOcts( offset ) = p.iOct_local;
                CellMask m = neighborMap.value_at(i);
                leaves.send_cell_masks( offset ) = clean_mask(m);
              }
              offset_local++;
            }
          }
        }
      }, nbLeaves_rank);
      uint32_t nbIntermediates_rank = 0;
      Kokkos::parallel_scan( "discover_ghosts::fill_ghostmap_intermediates", neighborMap.capacity(),
        KOKKOS_LAMBDA( uint32_t i, oct_index_t& offset_local, bool final)
      {

        if( neighborMap.valid_at(i) )
        {
          const NeighborPair& p = neighborMap.key_at(i);
          if( p.rank_neighbor == rank )
          {
            if( p.iOct_local >= nbLocalLeaves )
            {
              if(final)
              {
                uint32_t offset = offset_first_intermediate + offset_local;
                intermediates.send_iOcts( offset ) = p.iOct_local-nbLocalLeaves;
                CellMask m = neighborMap.value_at(i);
                intermediates.send_cell_masks( offset ) = clean_mask(m);
              }
              offset_local++;
            }
          }
        }
      }, nbIntermediates_rank);
      offset_first_leaf += nbLeaves_rank;
      offset_first_intermediate += nbIntermediates_rank;
    }
  }

  return to_send;  
}

/***
 * Create intermediates from leaves and add them to storage_device
 * 
 * Local intermediates are all octants between level_min and level_max-1 
 * for which their "upper left" (=smallest Morton) corner leaf octant is local.
 *  => The first suboctant of a local octant is local (intermediate or leaf)
 *  => The parent octant is local <=> "upper left" sibling is also local
 * 
 * @param storage input only local leaves, leaves + intermediates
 */
void init_intermediates(LightOctree_storage<>& storage_device)
{
  using OctantIndex = LightOctree::OctantIndex;

  int dim = storage_device.getNdim();
  int level_min = storage_device.level_min;
  uint32_t nbLocalLeaves = storage_device.getNumOctants();

  int first_intermediate_level = level_min;

  // Count intermediates
  uint32_t nbIntermediates_local = 0;
  Kokkos::parallel_reduce( "AMRmesh::init::count_intermediates", nbLocalLeaves,
    KOKKOS_LAMBDA( const uint32_t ioct_local , uint32_t& nbIntermediates_local)
  {
      const OctantIndex iOct_local{
        .iOct = ioct_local, 
        .isGhost = false, 
        .isIntermediate = false
      };
      uint32_t level = storage_device.getLevel(iOct_local);
      auto logical_coords = storage_device.get_logical_coords(iOct_local);
      // Create parent cell if current cell is the "origin subcell" below first_intermediate_level
      // This ensures intermediate is created once only by the process owning the origin subcell
      while (logical_coords[IX] % 2u == 0u && logical_coords[IY] % 2u == 0u 
          && logical_coords[IZ] % 2u == 0u && level > first_intermediate_level) {
          level--;
          logical_coords[IX] /= 2u;
          logical_coords[IY] /= 2u;
          logical_coords[IZ] /= 2u;
          nbIntermediates_local++;
      } 
  }, nbIntermediates_local);

  // Expand Storage to hold intermediates
  {
    LightOctree_storage<> storage_device_new(dim, nbLocalLeaves, 0, nbIntermediates_local, 0, level_min, storage_device.coarse_grid_size);
    Kokkos::deep_copy( storage_device_new.getLocalSubview(), storage_device.getLocalSubview() );
    storage_device = storage_device_new;
  }
  
  // Fill intermediates
  Kokkos::parallel_scan( "AMRmesh::init::fill_intermediates", nbLocalLeaves,
    KOKKOS_LAMBDA( const uint32_t ioct_local , uint32_t& i_intermediate, bool final)
  {
      const OctantIndex iOct_local{
        .iOct = ioct_local, 
        .isGhost = false, 
        .isIntermediate = false
      };
      uint32_t level = storage_device.getLevel(iOct_local);
      auto logical_coords = storage_device.get_logical_coords(iOct_local);
      while ( logical_coords[IX] % 2u == 0u && logical_coords[IY] % 2u == 0u 
            && logical_coords[IZ] % 2u == 0u && level > first_intermediate_level) 
      {
          level--;
          logical_coords[IX] /= 2u;
          logical_coords[IY] /= 2u;
          logical_coords[IZ] /= 2u;
          if( final )
          {
            const OctantIndex iOct_intermediate{
              .iOct = i_intermediate, 
              .isGhost = false, 
              .isIntermediate = true
            };
            storage_device.set(iOct_intermediate, 
              logical_coords[IX],
              logical_coords[IY],
              logical_coords[IZ],
              level
            );
          }
          i_intermediate++;
      } 
  });
}

AMRmesh::GhostMap_t init_ghosts(
  LightOctree_storage<>& storage_device,
  const std::vector<morton_t>& morton_intervals_,
  level_t level_min, level_t level_max, 
  const Kokkos::Array<bool,3>& periodic,
  const MpiComm& mpi_comm )
{
  AMRmesh::GhostMap_t ghostmap = discover_ghosts( 
                                      storage_device, 
                                      morton_intervals_,
                                      level_min, 
                                      level_max,
                                      periodic,
                                      mpi_comm );
  
  ViewCommunicator ghost_comm_leaves( ghostmap.to_send_leaves.send_sizes, ghostmap.to_send_leaves.send_iOcts );
  ViewCommunicator ghost_comm_intermediates( ghostmap.to_send_intermediates.send_sizes, ghostmap.to_send_intermediates.send_iOcts );
  { // Reallocate storage_device to hold ghosts
    int nbGhostLeaves = ghost_comm_leaves.getNumGhosts();
    int nbGhostIntermediates = ghost_comm_intermediates.getNumGhosts();
    LightOctree_storage<> storage_device_new( storage_device.getNdim(),
                                              storage_device.getNumOctants(),
                                              nbGhostLeaves,
                                              storage_device.getNumIntermediates(),
                                              nbGhostIntermediates,
                                              storage_device.level_min,
                                              storage_device.coarse_grid_size  );
    Kokkos::deep_copy( storage_device_new.getAllLocalsSubview(), storage_device.getAllLocalsSubview() );
    storage_device = storage_device_new;
  }

  ghost_comm_leaves.exchange_ghosts<0>( storage_device.getLocalSubview(), storage_device.getGhostsSubview() );
  ghost_comm_intermediates.exchange_ghosts<0>( storage_device.getLocalIntermediatesSubview(), storage_device.getGhostIntermediatesSubview() );

  return ghostmap;
}

void private_init(AMRmesh::PData& pdata, const Kokkos::Array<logical_coord_t,3>& coarse_grid_size)
{
  const int dim = pdata.ndim;
  const Storage_t& storage = pdata.storage;
  const MpiComm& mpi_comm = pdata.mpi_comm;
  DYABLO_ASSERT_HOST_RELEASE(dim == 3 || coarse_grid_size[IZ] == 1, "coarse_grid_size[IZ] must be 1 in 2D but is : " << coarse_grid_size[IZ]);

  level_t level_min = pdata.level_min;

  int mpi_rank = mpi_comm.MPI_Comm_rank();
  int mpi_size = mpi_comm.MPI_Comm_size();

  uint32_t full_cube_width = (1U << level_min);
  
  // Compute local octant count and first global octant index
  uint64_t first_local_oct = ((morton_t)mpi_rank * pdata.total_num_octs ) / (morton_t)mpi_size;
  pdata.first_local_oct = first_local_oct;
  uint64_t end_local_oct = ((morton_t)(mpi_rank+1) * pdata.total_num_octs ) / (morton_t)mpi_size;
  uint32_t nbOcts_local = end_local_oct - first_local_oct ;

  // Compute morton intervals
  std::vector<morton_t> morton_intervals( mpi_size+1 ); // 2D or 3D morton depending on `dim`
  if( coarse_grid_size[IX] == full_cube_width
  &&  coarse_grid_size[IY] == full_cube_width
  &&  ( coarse_grid_size[IZ] == full_cube_width || dim == 2 ) )
  { // If full cube We have an analytical formula for morton_intervals
    morton_t morton_max = pdata.total_num_octs;

    for(int i=0; i<=mpi_size; i++)
      morton_intervals[i] = ((morton_t)i * morton_max ) / (morton_t)mpi_size;    
  }
  else
  { // If not full cube we have to perform expensive scan to find load balancing
    morton_t morton_max = (dim == 3)? 
      compute_morton_key(coarse_grid_size[IX]-1, coarse_grid_size[IY]-1, coarse_grid_size[IZ]-1)+1:
      compute_morton_key(coarse_grid_size[IX]-1, coarse_grid_size[IY]-1)+1;      
    Kokkos::View<morton_t> first_morton_index_device("first_morton_index");
    Kokkos::parallel_scan( "AMRmesh::init::morton_intervals", morton_max,
      KOKKOS_LAMBDA( uint64_t morton, uint64_t& iOct_global, bool final )
    {
      logical_coord_t ix = morton_extract_coord<IX>(dim, morton);
      logical_coord_t iy = morton_extract_coord<IY>(dim, morton);
      logical_coord_t iz;
      if (dim==3)
        iz = morton_extract_coord<IZ>(dim,morton);
      else 
        iz = 0;

      if( ix < coarse_grid_size[IX] && iy < coarse_grid_size[IY] && iz < coarse_grid_size[IZ] )
      {
        if(final && iOct_global == first_local_oct)
          first_morton_index_device() = morton;

        iOct_global ++;
      }
    });
    auto first_morton_index_host = Kokkos::create_mirror_view( first_morton_index_device );
    Kokkos::deep_copy(first_morton_index_host, first_morton_index_device);

    mpi_comm.MPI_Allgather( &first_morton_index_host(), morton_intervals.data(), 1 );
    morton_intervals[mpi_size] = morton_max;
  }

  uint64_t morton_begin = morton_intervals[mpi_rank];
  uint64_t morton_end = morton_intervals[mpi_rank+1];

  // Fill local octs
  uint32_t nbOcts_added = 0;
  LightOctree_storage<> storage_device(dim, nbOcts_local, 0, 0, 0, level_min, coarse_grid_size);
  Kokkos::parallel_scan( "AMRmesh::init", 
    Kokkos::RangePolicy<>(morton_begin, morton_end),
    KOKKOS_LAMBDA( uint64_t morton, uint32_t& iOct, bool final )
  {
    logical_coord_t ix = morton_extract_coord<IX>(dim, morton);
    logical_coord_t iy = morton_extract_coord<IY>(dim, morton);
    logical_coord_t iz;
    if (dim==3)
      iz = morton_extract_coord<IZ>(dim,morton);
    else 
      iz = 0;

    if( ix < coarse_grid_size[IX] && iy < coarse_grid_size[IY] && iz < coarse_grid_size[IZ] )
    {
      if(final)
        storage_device.set( {iOct, false}, ix, iy, iz, level_min );

      iOct ++;
    }
  }, nbOcts_added);
  DYABLO_ASSERT_HOST_RELEASE( nbOcts_local == nbOcts_added, "Too few octs were added during Kokkos::scan : added " << nbOcts_added << ", expected " << nbOcts_local );

  // Add intermediates to storage_device
  init_intermediates(storage_device);

  std::vector<morton_t> morton_intervals_3D( mpi_size+1 );
  if( dim == 3 )
    morton_intervals_3D = morton_intervals;
  else
  {
    for(int i=0; i<=mpi_size; i++)
    {
      logical_coord_t ix = morton_extract_coord<IX>(2, morton_intervals[i]);
      logical_coord_t iy = morton_extract_coord<IY>(2, morton_intervals[i]);
      morton_intervals_3D[i] = compute_morton_key( ix, iy, 0 );
    }
  }

  pdata.ghostmap = init_ghosts( 
      storage_device, 
      morton_intervals_3D, 
      pdata.level_min,pdata.level_min, // This is not an error : mortons are computed a level_min
      pdata.periodic, 
      pdata.mpi_comm );

  pdata.storage = storage_device.deep_copy<Storage_t::MemorySpace>();

  pdata.markers = markers_t( "markers", storage.getNumOctants() );

  std::cout << "AMRmesh initialized : " << storage.getNumOctants() << " ( "  << storage.getNumGhosts() << " ) " << std::endl;
}

} // namespace

AMRmesh::AMRmesh( int dim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max, const MpiComm& mpi_comm)
: AMRmesh( dim, periodic, level_min, level_max, {(1U << level_min), (1U << level_min), (dim==3)?(1U << level_min):1 }, mpi_comm )
{}

AMRmesh::AMRmesh( int dim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max, const std::array<uint32_t,3>& coarse_grid_size, const MpiComm& mpi_comm)
: pdata( std::make_unique<PData>
  (PData{
    .ndim = dim,
    .level_min =      level_min,
    .level_max =      level_max,
    .periodic =       Kokkos::Array<bool,3>{periodic[IX],periodic[IY],periodic[IZ]},
    .storage =        Storage_t(),
    .mpi_comm =       mpi_comm,
    .total_num_octs = coarse_grid_size[IX]*coarse_grid_size[IY]*coarse_grid_size[IZ],
    .first_local_oct = 0,
    .ghostmap{},
    /*
    .markers = 
    .lmesh = 
    .lmesh_epoch = 0,
    .pmesh_epoch = 1
    */
  }))
{
  DYABLO_ASSERT_HOST_RELEASE( level_min <= level_max, (int)level_min << " <= " << (int)level_max );
  DYABLO_ASSERT_HOST_RELEASE( coarse_grid_size[IX] <= (1U << level_min), "Coarse grid size too big for level_min : " << coarse_grid_size[IX] << " > " << (int)(1U << level_min) );
  DYABLO_ASSERT_HOST_RELEASE( coarse_grid_size[IY] <= (1U << level_min), "Coarse grid size too big for level_min : " << coarse_grid_size[IY] << " > " << (int)(1U << level_min) );
  DYABLO_ASSERT_HOST_RELEASE( coarse_grid_size[IZ] <= (dim==3)?(1U << level_min):1, "Coarse grid size too big for level_min : " << coarse_grid_size[IZ] << " > " << (int)(1U << level_min) );

  if(  level_min > 0
    && coarse_grid_size[IX] <= (1U << (level_min-1)) 
    && coarse_grid_size[IY] <= (1U << (level_min-1)) 
    && coarse_grid_size[IZ] <= ((dim==3)?(1U << (level_min-1)):0) )
  {
    std::cout << "WARNING : AMRmesh coarse grid size = {" << coarse_grid_size[IX] << ", " << coarse_grid_size[IY] << ", " << coarse_grid_size[IZ] << "} is too small for level_min = " << (int)level_min << "." << std::endl;
    std::cout << "WARNING : level_min could be decreased to " << std::ceil(std::log2( std::max({coarse_grid_size[IX], coarse_grid_size[IY], coarse_grid_size[IZ]}) )) << std::endl;
    DYABLO_ASSERT_HOST_RELEASE(false, "level_min too big for coarse grid size");
  }
  private_init( *pdata, {coarse_grid_size[IX], coarse_grid_size[IY], coarse_grid_size[IZ]} );
}

AMRmesh::~AMRmesh()
{}

AMRmesh::Parameters AMRmesh::parse_parameters(ConfigMap& configMap)
{
  Parameters res;
  int ndim = configMap.getValue<int>("mesh", "ndim", 3);
  res.dim = ndim;
  BoundaryConditionType bxmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING);
  BoundaryConditionType bxmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING);
  BoundaryConditionType bymin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING);
  BoundaryConditionType bymax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING);
  BoundaryConditionType bzmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING);
  BoundaryConditionType bzmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING);
  res.periodic = {
    bxmin == BC_PERIODIC || bxmax == BC_PERIODIC,
    bymin == BC_PERIODIC || bymax == BC_PERIODIC,
    bzmin == BC_PERIODIC || bzmax == BC_PERIODIC
  };

  if( configMap.hasValue( "amr","level_min" ) )
  {
    res.level_min = configMap.getValue<int>("amr","level_min");
    res.level_max = configMap.getValue<int>("amr","level_max", res.level_min + 3);
    uint32_t max_width = (1U << res.level_min);
    res.coarse_grid_size = { 
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_x", max_width ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_y", max_width ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_z", (ndim==3)?max_width:1 ) 
    };

    DYABLO_ASSERT_HOST_RELEASE( res.coarse_grid_size[IX] <= max_width, 
      "amr/coarse_oct_resolution_x (" << res.coarse_grid_size[IX] << ") is too big for level_min (" << res.level_min << "), max allowed is " << max_width );
    DYABLO_ASSERT_HOST_RELEASE( res.coarse_grid_size[IY] <= max_width, 
      "amr/coarse_oct_resolution_y (" << res.coarse_grid_size[IY] << ") is too big for level_min (" << res.level_min << "), max allowed is " << max_width );
    DYABLO_ASSERT_HOST_RELEASE( res.coarse_grid_size[IZ] <= ((ndim==3)?max_width:1), 
      "amr/coarse_oct_resolution_z (" << res.coarse_grid_size[IZ] << ") is too big for level_min (" << res.level_min << "), max allowed is " << ((ndim==3)?max_width:1) );
  
  }
  else if( configMap.hasValue("amr","coarse_oct_resolution_x") 
        && configMap.hasValue("amr","coarse_oct_resolution_y") 
        && ( ( ndim == 2 ) || configMap.hasValue("amr","coarse_oct_resolution_z") ) )
  {
    res.coarse_grid_size = { 
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_x" ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_y" ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_z", 1 ) 
    };

    uint32_t width = std::max({res.coarse_grid_size[IX], res.coarse_grid_size[IY], res.coarse_grid_size[IZ]} );
    res.level_min = configMap.getValue<int>( "amr", "level_min", std::ceil(std::log2( width ))  );
    res.level_max = configMap.getValue<int>( "amr", "level_max", res.level_min + 3);
  }

  return res;
}

uint8_t AMRmesh::getDim() const
{
  return pdata->storage.getNdim();
}

bool AMRmesh::getPeriodic(uint8_t i) const
{
  return pdata->periodic[i];
}

Kokkos::Array<uint32_t,3> AMRmesh::get_coarse_grid_size()
{
  return pdata->storage.coarse_grid_size;
}

int AMRmesh::get_max_supported_level()
{
  return 20; // Maybe more? (But never tested)
}

int AMRmesh::get_level_min() const
{
  return pdata->level_min;
}

int AMRmesh::get_level_max() const
{
  return pdata->level_max;
}

const Storage_t AMRmesh::getStorage() const
{
  return pdata->storage;
}

const LightOctree& AMRmesh::getLightOctree()
{
  // Update LightOctree if needed
  updateLightOctree();
  const LightOctree& lmesh = *(pdata->lmesh);
  DYABLO_ASSERT_HOST_RELEASE( lmesh.getNumOctants() == this->getNumOctants(), "LightOctree::getLightOctree() is outdated pmesh " << this->getNumOctants() << "octs vs lmesh " << lmesh.getNumOctants() << "octs" );
  DYABLO_ASSERT_HOST_RELEASE( lmesh.getNumGhosts() == this->getNumGhosts(), "LightOctree::getLightOctree() is outdated pmesh " << this->getNumGhosts() << "ghosts vs lmesh " << lmesh.getNumGhosts() << "ghosts" );
  return lmesh;
}

void AMRmesh::updateLightOctree()
{
  // Update LightOctree if needed
  if( pdata->pmesh_epoch != pdata->lmesh_epoch )
  {
    pdata->lmesh = nullptr;
    pdata->lmesh = std::make_unique<LightOctree>( this, pdata->level_min, pdata->level_max );
    pdata->lmesh_epoch = pdata->pmesh_epoch;
  }
}

MpiComm AMRmesh::getMpiComm() const
{
  return pdata->mpi_comm;
}

uint32_t AMRmesh::getNumOctants() const
{
  return pdata->storage.getNumOctants();
}

uint32_t AMRmesh::getNumGhosts() const
{
  return pdata->storage.getNumGhosts();
}

uint64_t AMRmesh::getGlobalNumOctants() const
{
  return pdata->total_num_octs;
}

uint64_t AMRmesh::getGlobalIdx( uint32_t idx ) const
{
  return idx + pdata->first_local_oct;
}

uint32_t AMRmesh::getNumIntermediates() const
{
  return pdata->storage.getNumIntermediates();
}

uint32_t AMRmesh::getNumIntermediateGhosts() const
{
  return pdata->storage.getNumIntermediateGhosts();
}


AMRmesh::GhostMap_t AMRmesh::loadBalance(uint8_t level)
{
  Storage_t& storage = pdata->storage;
  
  const MpiComm& mpi_comm = this->getMpiComm();
  int mpi_rank = mpi_comm.MPI_Comm_rank();
  int mpi_size = mpi_comm.MPI_Comm_size();
  level_t level_max = pdata->level_max;

  std::vector<morton_t> new_morton_intervals(mpi_size+1);    
  // Get evenly distributed initial intervals and gather mortons
  {
    int nb_mortons = 0;
    global_oct_index_t iOct_begin = this->getGlobalIdx( 0 );
    global_oct_index_t iOct_end = this->getGlobalIdx( this->getNumOctants() );
    for(int i=0; i<mpi_size; i++)
    {
        global_oct_index_t idx = (this->getGlobalNumOctants()*i)/mpi_size ;
        // For each ixd inside old domain, compute morton
        // and fill morton_intervals for this rank
        if( iOct_begin <= idx && idx < iOct_end )
        {
            new_morton_intervals[i] = compute_morton( storage, idx-iOct_begin, level_max );
            nb_mortons++;
        }
    }   

    // allgather morton_intervals
    mpi_comm.MPI_Allgatherv_inplace( new_morton_intervals.data(), nb_mortons );
    new_morton_intervals[0] = 0;
    new_morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
    {
      for(int rank=1; rank<mpi_size; rank++)
      {
          morton_t new_morton_begin_rank;
          level_t adjusted_level = level+1;
          do // Adapt `level` to avoid getting empty processes
          {
            adjusted_level --;
            // Truncate suboctants to keep `adjusted_level` levels of suboctants compact
            new_morton_begin_rank = (new_morton_intervals[rank] >> (3*adjusted_level)) << (3*adjusted_level);
            // Ensure that no process is empty by adjusting `levels` so new_morton_intervals is strictly increasing
          } while( adjusted_level>0 && new_morton_begin_rank <= new_morton_intervals[rank-1] );
          if( adjusted_level != level )
            std::cout << "WARNING : Could not ensure " << level << " levels coherency for rank " << rank << " (would be empty) used " << adjusted_level << " levels instead" << std::endl;
          new_morton_intervals[rank] = new_morton_begin_rank;
      }
    }
    DYABLO_ASSERT_HOST_RELEASE(new_morton_intervals[mpi_rank] < new_morton_intervals[mpi_rank+1], "Process would be empty");
  }

  //std::cout << "Rank " << mpi_rank << ": new morton interval [" << new_morton_intervals[mpi_rank] << ", " << new_morton_intervals[mpi_rank+1] << "[" << std::endl;

  // Compute `new_oct_intervals` corresponding to `morton_intervals`
  std::vector<global_oct_index_t> new_oct_intervals(mpi_size+1); // First global index for rank i
  {
      morton_t old_morton_interval_begin, old_morton_interval_end;
      {
        if( storage.getNumOctants() != 0 )
          old_morton_interval_begin = compute_morton( storage, 0, level_max );
        else
          old_morton_interval_begin = 0;
        std::vector<morton_t> old_morton_intervals(mpi_size+1);
        // TODO maybe just send to mpi_rank-1 ?
        mpi_comm.MPI_Allgather( &old_morton_interval_begin, old_morton_intervals.data(), 1);
        old_morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
        for(int rank=mpi_size-1; rank>0; rank--)
          if( old_morton_intervals[rank] == 0 )
            old_morton_intervals[rank] = old_morton_intervals[rank+1];
        old_morton_interval_begin = old_morton_intervals[mpi_rank];
        old_morton_interval_end = old_morton_intervals[mpi_rank+1];
      }            

      //std::cout << "Rank " << mpi_rank << ": old morton interval [" << old_morton_interval_begin << ", " << old_morton_interval_end << "[" << std::endl;

      int nb_local_pivots=0;
      for(int rank=0; rank<mpi_size; rank++)
      {
          if( old_morton_interval_begin <= new_morton_intervals[rank] && new_morton_intervals[rank] < old_morton_interval_end ) // Determine if pivot is inside of local process
          {
              // find first local octant with morton >= morton_interval[rank]
              oct_index_t pivot = lower_bound_morton( storage, new_morton_intervals[rank], level_max );
              nb_local_pivots++;
              new_oct_intervals[rank] = this->getGlobalIdx(pivot);
          }
      }

      mpi_comm.MPI_Allgatherv_inplace( new_oct_intervals.data(), nb_local_pivots );
      new_oct_intervals[mpi_size] = this->getGlobalNumOctants();

  }
  //std::cout << "Rank " << mpi_rank << ": iOct interval [" << new_oct_intervals[mpi_rank] << ", " << new_oct_intervals[mpi_rank+1] << "[" << std::endl;
  DYABLO_ASSERT_HOST_RELEASE( new_oct_intervals[mpi_rank] <= new_oct_intervals[mpi_rank+1], "iOct_interval upper bound smaller than lower bound" );

  // List octants to exchange
  GhostMap_t::SendList res{};
  {
    res.send_sizes = Kokkos::View<uint32_t*>( "Loadbalance::send_sizes", mpi_size );
    res.send_iOcts = Kokkos::View<uint32_t*>( "Loadbalance::send_iOct", this->getNumOctants() );

    // Fill send_iOcts with octant index (all ocatnts are sent)
    Kokkos::parallel_for( "Loadbalance::create_send_iOct", this->getNumOctants(),
      KOKKOS_LAMBDA( uint32_t iOct )
    {
      res.send_iOcts(iOct) = iOct;
    });

    // Fill send_sizes with sizes determined according to new morton intervals
    auto send_sizes_host = Kokkos::create_mirror_view(res.send_sizes);
    for( int rank=0; rank<mpi_size; rank++ )
    {
      // intersection between local and remote ranks
      global_oct_index_t global_local_begin = this->getGlobalIdx(0);
      global_oct_index_t global_local_end = this->getGlobalIdx(this->getNumOctants()) ;
      global_oct_index_t global_intersect_begin = std::max( new_oct_intervals[rank],   global_local_begin );
      global_oct_index_t global_intersect_end   = std::min( new_oct_intervals[rank+1], global_local_end  );
      
      if( global_intersect_end > global_intersect_begin )
        send_sizes_host( rank ) = global_intersect_end - global_intersect_begin;      
    }
    Kokkos::deep_copy(res.send_sizes, send_sizes_host);
  }

  // Exchange octs that changed domain 
  oct_index_t new_nbOcts = new_oct_intervals[mpi_rank+1]-new_oct_intervals[mpi_rank];
  LightOctree_storage<> new_storage_device(this->getDim(), new_nbOcts, 0, 0, 0, storage.level_min, storage.coarse_grid_size );
  {
    // Use storage on device to perform remaining operations
    LightOctree_storage<> old_storage_device = storage.deep_copy<LightOctree_storage<>::MemorySpace>();

    ViewCommunicator loadbalance_communicator( res.send_sizes, res.send_iOcts );
    loadbalance_communicator.exchange_ghosts<0>( old_storage_device.oct_data, new_storage_device.oct_data );

    DYABLO_ASSERT_HOST_RELEASE( new_storage_device.oct_data.extent(0) == new_nbOcts, "Mismatch between nbOcts and allocation size : " << new_nbOcts << " != " << new_storage_device.oct_data.extent(0) );
  }

  // update misc metadata
  pdata->first_local_oct = new_oct_intervals[mpi_rank];
  pdata->pmesh_epoch++;

  // Add intermediates to storage_device
  init_intermediates(new_storage_device);
  // Add ghosts to storage_device
  pdata->ghostmap = init_ghosts(new_storage_device, new_morton_intervals, pdata->level_min, level_max, pdata->periodic, mpi_comm);

  storage = new_storage_device.deep_copy<Storage_t::MemorySpace>();

  pdata->markers = markers_t("markers", this->getNumOctants());

  // Print new domain decomposition
  if( new_nbOcts != 0 )
  {
    //std::cout << "Rank " << mpi_rank << ": actual morton interval [" << compute_morton( storage, 0, level_max) << ", " << compute_morton( storage,  getNumOctants()-1, level_max) << "]" << std::endl;
    DYABLO_ASSERT_HOST_RELEASE( compute_morton( storage, 0, level_max) >= new_morton_intervals[mpi_rank], "First octant outside of morton interval : " << compute_morton( storage, 0, level_max) );
    DYABLO_ASSERT_HOST_RELEASE( compute_morton( storage, getNumOctants()-1, level_max) < new_morton_intervals[mpi_rank+1], "Last octant outside of morton interval : " << compute_morton( storage, getNumOctants()-1, level_max));
  }
  else 
  {
    std::cout << "Rank " << mpi_rank << ": actual morton interval [EMPTY]" << std::endl;
    //std::cout << "WARNING : Rank has 0 octant, this is probably not okay" << std::endl;
  } 

  DYABLO_ASSERT_HOST_RELEASE(this->getNumOctants() > 0, "Process cannot be empty" );

  return GhostMap_t{
    .to_send_leaves = res
  };
}

void AMRmesh::loadBalance_userdata( int compact_levels, UserData& userData )
{
  auto ghostmap = this->loadBalance(compact_levels);
  ViewCommunicator lb_comm(ghostmap.to_send_leaves.send_sizes, ghostmap.to_send_leaves.send_iOcts);  
  userData.exchange_loadbalance( lb_comm );
}

void AMRmesh::setMarker(uint32_t iOct, int marker)
{
  pdata->markers(iOct) = marker;
}

void AMRmesh::setMarkers( const Kokkos::View<int*>& oct_markers )
{
  Kokkos::deep_copy( pdata->markers, oct_markers );
}

Kokkos::View<int*> AMRmesh::getMarkers() const
{
  Kokkos::View<int*> oct_markers( "AMRmesh::markers", pdata->markers.size() );
  Kokkos::deep_copy( oct_markers, pdata->markers );
  return oct_markers;
}

void AMRmesh::adapt()
{
  const MpiComm& mpi_comm = this->getMpiComm();
  Storage_t& storage = pdata->storage;
  int mpi_size = mpi_comm.MPI_Comm_size();
  level_t level_max = pdata->level_max;

  using OctantIndex = LightOctree_hashmap::OctantIndex;

  LightOctree_hashmap lmesh(this, pdata->level_min, pdata->level_max);
  LightOctree_storage<> old_storage_device = this->getStorage().deep_copy<LightOctree_storage<>::MemorySpace>();

  int ndim = lmesh.getNdim();
  oct_index_t nbOcts = getNumOctants();

  // Double buffering for markers
  Kokkos::View<int*, Kokkos::LayoutLeft> markers_device_in("adapt::markers", nbOcts);
  Kokkos::View<int*, Kokkos::LayoutLeft> markers_device_out("adapt::markers", nbOcts);
  Kokkos::View<int*, Kokkos::LayoutLeft> markers_ghosts_device("adapt::markers_ghost", this->getNumGhosts());
  // Helper functions for markers
  auto getMarker = KOKKOS_LAMBDA( const OctantIndex& iOct ) -> int
  { 
    if( iOct.isGhost )
      return markers_ghosts_device( iOct.iOct );
    else
      return markers_device_in( iOct.iOct );
  };
  auto setMarker = KOKKOS_LAMBDA( const OctantIndex& iOct, int marker )
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( !iOct.isGhost, "Ghost markers cannot be modified locally");
    markers_device_out( iOct.iOct ) = marker;
  };

  // Compute correct markers : 2:1 balance and remove partially coarsened octants
  {
    // Copy CPU markers to markers_device
    Kokkos::deep_copy( markers_device_in, pdata->markers );
    ViewCommunicator ghost_comm( pdata->ghostmap.to_send_leaves.send_sizes, pdata->ghostmap.to_send_leaves.send_iOcts );
    
    // Check for 2:1 balance and partially coarsened octants and modify marker to enforce these rules
    // return true if marker was modified, false otherwise
    auto check_21_cell = KOKKOS_LAMBDA( oct_index_t iOct )
    {
      level_t level_current = lmesh.getLevel({iOct, false});
      int marker_current = getMarker( {iOct, false} );
      int marker_old = marker_current;

      DYABLO_ASSERT_KOKKOS_DEBUG( marker_current >= -1, "Marker must be -1, 0, 1" );
      DYABLO_ASSERT_KOKKOS_DEBUG( marker_current <= 1, "Marker must be -1, 0, 1" );

      // Check siblings for partial coarsening
      if( marker_current == -1 )
      {
        auto pos = old_storage_device.get_logical_coords( {iOct, false} );
        // Shift to get on which size siblings are
        int sx = (pos[IX]%2==0)?1:-1;
        int sy = (pos[IY]%2==0)?1:-1;
        int sz = (pos[IZ]%2==0)?1:-1;
        // Iterate over siblings
        for(int z=0; z<(ndim-1); z++)
        for(int y=0; y<2; y++)
        for(int x=0; x<2; x++)
        if( x!=0 || y!=0 || z!=0 )
        {
          auto ns = lmesh.findNeighbors({iOct, false},{(int8_t)(sx*x),(int8_t)(sy*y),(int8_t)(sz*z)});
          DYABLO_ASSERT_KOKKOS_DEBUG(ns.size()>0, "Siblings can't be outside domain");
          level_t level_siblings = lmesh.getLevel( ns[0] );
          DYABLO_ASSERT_KOKKOS_DEBUG( level_siblings >= level_current, "Siblings cannot be coarser");

          int marker_siblings = getMarker( ns[0] );
          // Cancel coarsening if siblings cannot be coarsened enough
          if( level_current+marker_current < level_siblings+marker_siblings )
            marker_current = 0;
        }
      }          

      // Check neighborhood for 2:1 violations
      int nz_max = ndim == 2? 0:1;
      for( int8_t nz=-nz_max; nz<=nz_max; nz++ )
      for( int8_t ny=-1; ny<=1; ny++ )
      for( int8_t nx=-1; nx<=1; nx++ )
      if( nx!=0 || ny!=0 || nz!=0 )
      {
        auto ns = lmesh.findNeighbors({iOct,false}, {nx,ny,nz});
        for(int n=0; n<ns.size(); n++)
        {
          level_t level_neighbor = lmesh.getLevel(ns[n]);
          int maker_neighbor = getMarker( ns[n] );
          // If current marker violates 2:1 (too coarse compared to neighbors)
          if( level_current+marker_current < level_neighbor+maker_neighbor-1 )
          {
            // Set to smallest compatible marker
            marker_current = (level_neighbor-level_current)+maker_neighbor-1;
          }
        }
      }

      DYABLO_ASSERT_KOKKOS_DEBUG( marker_current >= -1, "Marker must be -1, 0, 1" );
      DYABLO_ASSERT_KOKKOS_DEBUG( marker_current <= 1, "Marker must be -1, 0, 1" );

      setMarker( {iOct, false}, marker_current );

      return marker_current != marker_old;
    };

    // Re-check 2:1 with updated ghosts while at last one process updated markers
    // TODO : only re-check when ghosts have been modified
    bool updated_markers_global = true;
    while( updated_markers_global )
    {
      updated_markers_global = false;
      
      ghost_comm.exchange_ghosts<0>(markers_device_in, markers_ghosts_device);

      oct_index_t updated_markers_local = 1;
      while( updated_markers_local != 0 )
      {
        // TODO only re-verify octants close to recentely modified markers
        updated_markers_local = 0;
        Kokkos::parallel_reduce( "adapt::check_all_2:1", nbOcts, 
          KOKKOS_LAMBDA( oct_index_t iOct, oct_index_t& modified_count )
        {
          bool modified = check_21_cell(iOct);
          if( modified ) modified_count++;
        }, updated_markers_local);

        // Swap marker buffers
        Kokkos::deep_copy( markers_device_in, markers_device_out );

        if( updated_markers_local != 0 ) 
        {
          updated_markers_global = true;
        }
      }
    
      mpi_comm.MPI_Allreduce( &updated_markers_global, &updated_markers_global, 1, MpiComm::MPI_Op_t::LOR );
    }
  }

  // Apply corrected markers
  {
    int nSiblings = 4*(ndim-1);
    oct_index_t old_nbOcts = old_storage_device.getNumOctants();

    // Use parallel_scan to compute new_nbOcts and oct_offsets
    oct_index_t new_nbOcts = 0; // Number of octants after applying markers
    // Offset in new oct data where to write new octant(s) related to old octant
    Kokkos::View<oct_index_t*> oct_offsets("adapt::oct_offsets", old_nbOcts);
    Kokkos::parallel_scan("adapt::count_new_nbOcts", old_nbOcts,
      KOKKOS_LAMBDA( oct_index_t iOct_old, oct_index_t& iOct_new, bool final )
    {
      int marker = getMarker( {iOct_old, false} );
      auto pos = old_storage_device.get_logical_coords( {iOct_old, false} );

      // Compute number of octants written for iOct_old
      int nwrite = 0;
      {
        if     ( marker == 1 )  nwrite = nSiblings;
        else if( marker == 0 )  nwrite = 1;
        // Write coarsened oct when iOct_old is the first sibling
        // NOTE : ! first sibling may not be in same domain !
        else if( marker == -1)  nwrite = ( pos[IX]%2==0 && pos[IY]%2==0 && pos[IZ]%2==0 );
        else DYABLO_ASSERT_KOKKOS_DEBUG( false, "Marker must be -1, 0, 1" );
      }

      if(final) // Write offset in final pass
        oct_offsets( iOct_old ) = iOct_new;

      iOct_new += nwrite; // iOct_new is exclusive prefix sum for nwrite
    }, new_nbOcts);

    // Allocate new storage on device
    LightOctree_storage<> new_storage_device( ndim, new_nbOcts, 0, 0, 0, storage.level_min, storage.coarse_grid_size  );

    // Write new octant data
    Kokkos::parallel_for( "adapt::apply", old_nbOcts,
      KOKKOS_LAMBDA( oct_index_t iOct_old )
    {
      level_t level = old_storage_device.getLevel( {iOct_old, false} );
      auto pos = old_storage_device.get_logical_coords( {iOct_old, false} );
      int marker = getMarker( {iOct_old, false} );
      oct_index_t iOct_new = oct_offsets( iOct_old );

      DYABLO_ASSERT_KOKKOS_DEBUG( marker == -1 || marker == 0 || marker == 1, "Marker must be -1, 0, 1" );

      // int nwritten = 0; // Unused, but could be used for debug with parallel_reduce
      if( marker == 1 )
      { // Refine : write nSiblings suboctants
        for( int j=0; j<nSiblings; j++ )
        {
            int dz = j/(2*2);
            int dy = (j-dz*2*2)/2;
            int dx = j-dz*2*2-dy*2; // This is Z-curve order

            new_storage_device.set( {iOct_new + j, false}, 
              pos[IX]*2 + dx, 
              pos[IY]*2 + dy, 
              pos[IZ]*2 + dz, 
              level + 1 );
        }
        //nwritten = nSiblings;
      }
      else if (marker == 0)
      { // Not modified : copy old octant at nex position
        new_storage_device.set( {iOct_new, false}, 
              pos[IX], 
              pos[IY], 
              pos[IZ], 
              level);
        //nwritten = 1;
      }
      else if (marker == -1)
      {
        // Write coarsened oct only when iOct_old is the first sibling
        if( pos[IX]%2==0 && pos[IY]%2==0 && pos[IZ]%2==0 )
        {
          new_storage_device.set( {iOct_new, false}, 
                pos[IX]/2, 
                pos[IY]/2, 
                pos[IZ]/2, 
                level - 1);
          //nwritten = 1;
        }
        //else nwritten = 0;
      }
    });

    // Compute morton interval
    std::vector<morton_t> morton_intervals(mpi_size+1);
    {
      morton_t morton_interval_begin;
      if( storage.getNumOctants() != 0 )
      {
        Kokkos::View<morton_t> morton_interval_begin_device("morton_interval_begin");
        Kokkos::parallel_for( "adapt:compute_morton", 1, KOKKOS_LAMBDA(int)
        {
          morton_interval_begin_device() = compute_morton( new_storage_device, 0, level_max );
        });
        auto morton_interval_begin_host = Kokkos::create_mirror_view(morton_interval_begin_device);
        Kokkos::deep_copy( morton_interval_begin_host, morton_interval_begin_device );
        morton_interval_begin = morton_interval_begin_host();
      }
      else
        morton_interval_begin = 0;
      
      // TODO maybe just send to mpi_rank-1 ?
      mpi_comm.MPI_Allgather( &morton_interval_begin, morton_intervals.data(), 1);
      morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
      for(int rank=mpi_size-1; rank>0; rank--)
        if( morton_intervals[rank] == 0 )
         morton_intervals[rank] = morton_intervals[rank+1];
    }

    // Add intermediates to new_storage_device
    init_intermediates(new_storage_device);
    // Add ghosts to new_storage_device
    pdata->ghostmap = init_ghosts( new_storage_device, morton_intervals, pdata->level_min, pdata->level_max, pdata->periodic, pdata->mpi_comm );
    pdata->storage = new_storage_device.deep_copy<Storage_t::MemorySpace>();
  
    // compute total count and global index of first local octant
    {
      global_oct_index_t new_nbOcts_global = new_nbOcts;
      global_oct_index_t new_nbOcts_inclusive_prefix_sum;
      mpi_comm.MPI_Scan( &new_nbOcts_global, &new_nbOcts_inclusive_prefix_sum, 1, MpiComm::MPI_Op_t::SUM );
      pdata->first_local_oct = new_nbOcts_inclusive_prefix_sum - new_nbOcts_global;
      mpi_comm.MPI_Allreduce( &new_nbOcts_global, &pdata->total_num_octs, 1, MpiComm::MPI_Op_t::SUM );
      pdata->markers = markers_t("markers", new_nbOcts);
    }     
  }

  pdata->pmesh_epoch++;
}

void AMRmesh::adaptGlobalRefine()
{
  Kokkos::Experimental::fill( Kokkos::OpenMP(), pdata->markers, 1 );
  this->adapt();
}

const AMRmesh::GhostMap_t& AMRmesh::getGhostMap() const
{
  return pdata->ghostmap;
}

} // namespace dyablo

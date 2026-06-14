#pragma once

#include <memory>
#include <array>

#include "amr/LightOctree_forward.h"
#include "amr/LightOctree_storage.h"
#include "GlobalMpiSession.h"
#include "kokkos_shared.h"

class ConfigMap;

namespace dyablo{

class UserData;

class AMRmesh{
public:
  // Pimpl idiom
  struct PData;
private:   
  std::unique_ptr<PData> pdata;

public:
  /**
   * Construct a new empty AMR mesh initialized with a fixed grid at level_min
   * @param dim number of dimensions 2D/3D
   * @param periodic set perodicity for each dimension (last is ignored in 2D)
   * @param level_min minimum refinement level 
   * @param level_max maximum refinement level
   * (TODO : clarify level_min/level_max) 
   **/
  AMRmesh( int dim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world());
  AMRmesh( int dim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max, const std::array<uint32_t,3>& coarse_grid_size, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world());
  ~AMRmesh();

  struct Parameters
  {
    int dim;
    std::array<bool,3> periodic;
    uint8_t level_min, level_max;
    std::array<uint32_t,3> coarse_grid_size;
  };
  static Parameters parse_parameters(ConfigMap& configmap);

  //----- Mesh parameters -----
  /// Get number of dimensions
  uint8_t getDim() const;

  /// Get periodicity of direction i
  bool getPeriodic(uint8_t i) const;

  Kokkos::Array<uint32_t,3> get_coarse_grid_size();

  int get_max_supported_level();
  int get_level_min() const;
  int get_level_max() const;

  const LightOctree_storage<Kokkos::DefaultHostExecutionSpace::memory_space> getStorage() const;
  /**
   * Get the LightOctree associated to the current AMR mesh
   * May reallocate LightOctree if mesh has been modified
   **/
  const LightOctree& getLightOctree();
  /// Update LightOctree to make sure next call to getLightOctree() will not reallocate
  void updateLightOctree(); 

  //----- MPI info -----
  MpiComm getMpiComm() const;

  //----- Octant count -----
  /// Get number of local octants
  uint32_t getNumOctants() const;
  /// Get number of ghost octants
  uint32_t getNumGhosts() const;
  /// Get total number of octants across all MPI process
  uint64_t getGlobalNumOctants() const;

  /// Get number of local octants
  uint32_t getNumIntermediates() const;
  /// Get number of ghost octants
  uint32_t getNumIntermediateGhosts() const;

  /// Get the global id associated to local octant idx
  uint64_t getGlobalIdx( uint32_t idx ) const;

  struct GhostMap_t
  {
    enum Face{
      XL, XR,
      YL, YR,
      ZL, ZR,
      FACE_COUNT,
      FULL_BLOCK,
    };
    /// CellMasks represent a set of faces or special values defined below
    using CellMask = int;
    static constexpr CellMask MASK_FULL_BLOCK = 1 << Face::FACE_COUNT; /// Special CellMask value for full blocks
    static constexpr CellMask MASK_COUNT = MASK_FULL_BLOCK+1; /// Last CellMask value, all valid CellMask values must be <= cellMask_MAX

    struct SendList
    {
      Kokkos::View< uint32_t* > send_sizes; /// Number of octants to send to each process (of size nb_proc)
      Kokkos::View< uint32_t* > send_iOcts; /// Octants to send (of size sum(neighbor_leaves_send_sizes(i)) )
      Kokkos::View<CellMask*> send_cell_masks; /// Mask for each octant to send representing which faces of the octant are needed
    };

    SendList to_send_leaves; /// Leaf octants to send for stencil operations
    SendList to_send_intermediates; /// Intermediate octants to send for stencil operations
  };

  //----- Mesh modification -----
  /**
   * Change octants distribution to evenly redistribute the load
   * @param compact_levels are the number of levels to keep compact at the bottom of the tree
   *        all suboctants of octants at level (level_max - compact_level) are kept in the same process
   **/
  GhostMap_t loadBalance(uint8_t compact_levels=0); 

  /**
   * @copydoc AMRmesh_impl::loadBalance(uint8)
   * @param userData Kokkos::View of octant-related data to be redistributed
   *        userData must have layout Kokkos::LayoutLeft and rightmost index must be octants
   *        for each octant iOct that needs to be moved, values from userData(..., iOct) 
   *        are transfered to the new owning mpi rank
   **/
  void loadBalance_userdata( int compact_levels, UserData& userData );

  /**
   * Set marker for refinement 
   * @param marker +1 to mark iOct for refinement, -1 for coarsening
   **/
  void setMarker(uint32_t iOct, int marker);
  void setMarkers( const Kokkos::View<int*>& oct_markers );

  /**
   * Get a copy of the current refinement markers (one value per local octant).
   * Returned in the default memory space so it can be combined with markers
   * produced by refine conditions (e.g. to merge several conditions).
   **/
  Kokkos::View<int*> getMarkers() const;

  /**
   * Coarsen and refine octants according to markers set with setMarker()
   * adapt() includes 2:1 balancing in the directions set with `balance_codim` in the constructor
   * NOTE : refining/coarsening octants by more than one level is not supported (yet?)
   **/
  void adapt();
  /// Refine all octants : same as adapt with all octants marked +1
  void adaptGlobalRefine();

  const GhostMap_t& getGhostMap() const;
};

} // namespace dyablo
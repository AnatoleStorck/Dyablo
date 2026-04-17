#pragma once

#include "UserData.h"
#include "ScalarSimulationData.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo{

/**
 * Interface for HyperbolicPolicy used in Finite Volume solvers
 * to verify that HyperbolicPolicy implements all methods correctly
 * We recommand that you wrap your HyperbolicPolicy implementation 
 * in this template interface to check every interface methods
 **/
template< typename HyperbolicPolicy_impl >
class HyperbolicPolicy_base
{
protected:
  HyperbolicPolicy_impl impl;
public:
  KOKKOS_INLINE_FUNCTION
  const HyperbolicPolicy_impl& get_impl() const
  {
    return impl;
  }

  /// Structure containing primitive variables
  using PrimState = typename HyperbolicPolicy_impl::PrimState;
  /// Structure containing conservative variables
  using ConsState = typename HyperbolicPolicy_impl::ConsState;
  using CellIndex = ForeachCell::CellIndex;
  using CellMetaData = ForeachCell::CellMetaData;
  using FieldAccessor = UserData::FieldAccessor;

  using Params = typename HyperbolicPolicy_impl::Params;
  static Params getParams( ConfigMap& configMap )
  {
    return HyperbolicPolicy_impl::getParams(configMap);
  }

    /// HyperbolicPolicy needs to be constructible from a ConfigMap
  HyperbolicPolicy_base( const Params& params, const ScalarSimulationData& scalar_data )
  : impl(params, scalar_data)
  {}

  /**
   * Extract a FieldAccessor from U for input fields 
   * This must allow reading current timestep fields with getConsState()
   **/
  FieldAccessor getUin( UserData& U ) const
  {
    return impl.getUin(U);
  }

  /**
   * Extract a FieldAccessor from U for updated fields
   * This must allow reading/settings fields with get/setConsState()
   **/
  FieldAccessor getUout( UserData& U ) const
  {
    return impl.getUout(U);
  }

  /**
   * @brief Returns the conservative state at a given cell index in an array
   * 
   * @tparam Array_t the type of array where we are looking up
   * 
   * @param U the array in which we are getting the state
   * @param iCell the index of the cell 
   * @return the conservative state at position iCell in U : U(iCell)
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    return impl.getConsState(U, iCell);
  }

  /**
   * @brief Write the conservative state to an array
   * 
   * U(iCell) := u
   * 
   * @tparam Array_t the type of array in which the primitive value is stored
   * 
   * @param U the array where we are storing the state
   * @param iCell the index of cell
   * @param u the State value to store in the array
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    impl.setConsState(U, iCell, u);
  }

  /**
   * @brief Atomically accumulate a conservative state to an array
   * 
   * U(iCell) += u
   * 
   * @tparam Array_t the type of array in which the primitive value is stored
   * 
   * @param U the array where we are storing the state
   * @param iCell the index of cell
   * @param u the State value to accumulate with the existing value in the array
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    impl.atomic_addConsState(U, iCell, u);
  }

  /**
   * @brief Returns the primitive state at a given cell index in an array
   * 
   * @tparam Array_t the type of array where we are looking up
   * 
   * @param Q the array in which we are getting the state
   * @param iCell the index of the cell 
   * @return the conservative state at position iCell in Q : Q(iCell)
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    return impl.getPrimState(Q, iCell);
  }

  /**
   * @brief Write the primitive state to an array
   * 
   * Q(iCell) := q
   * 
   * @tparam Array_t the type of array in which the primitive value is stored
   * 
   * @param Q the array where we are storing the state
   * @param iCell the index of cell
   * @param q the State value to store in the array
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    impl.setPrimState(Q, iCell, q);
  }

  /***
   * @brief Converts from conservative state to primitive state
   * 
   * @param u the initial conservative state
   ***/
  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& u ) const
  {
    return impl.consToPrim(u);
  }

  /***
   * @brief Converts from primitive state to conservative state
   * 
   * @param u the initial primitive state
   ***/
  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& q ) const
  {
    return impl.primToCons(q);
  }

  /** 
   * @brief Riemann solver
   * 
   * Compute flux at interface using states at both sides of the interface
   *
   * @param qleft left state (primitive variables)
   * @param qright right state (primitive variables)
   * @param dir x, y or z direction
   * @param policy_scalar_data PolicyScalarData object for the passage of scalar information  
   * @return output flux
   */
  KOKKOS_INLINE_FUNCTION
  ConsState riemann_solver( PrimState qL, PrimState qR, ComponentIndex3D dir) const
  {
    return impl.riemann_solver(qL, qR, dir);
  }

  
  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    return impl.compute_slope(qL, qC, qR, dL, dR);
  }

  /**
   * @brief Get field values for a cell outside of the simulation domain
   * 
   * This method is usually called by kernels when trying to access neighbors outside
   * of the simulation domain.
   * Returns the state of a virtual cell outside of the domain.
   * The value returned may be determined by the values inside the domain 
   * (e.g. reflecting/absorbing boundary conditions)
   * 
   * @param U array containing conservative variables
   * @param iCell_boundary cell index outside of the domain (iCell_boundary.is_boundary() must be true)
   * @param metadata CellMetaData object allowing for mesh queries on size/position
   * @param policy_scalar_data PolicyScalarData object for the passage of scalar information  
   * @return The conservative state defined at the given boundary position 
   */
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const 
  {
    return impl.getBoundaryValue(*this, U, iCell_boundary, metadata);
  }
  
  /**
   * @brief Get flux value for a neighboring cell outside of the simulation domain
   * 
   * This method is called when trying to compute the flux with neighbors outside
   * of the simulation domain.
   * Returns the flux at the interface on domain boundary.
   * The value returned may be determined by the values inside the domain 
   * (e.g. reflecting/absorbing boundary conditions)
   * 
   * @param U array containing conservative variables
   * @param iCell_boundary cell index outside of the domain (iCell_boundary.is_boundary() must be true), 
   *          the cell must have an interface with a cell inside the domain
   * @param q_in_reconstructed reconstructed state inside domain at boundary interface
   * @param metadata CellMetaData object allowing for mesh queries on size/position
   * @param policy_scalar_data PolicyScalarData object for the passage of scalar information  
   * @return The flux at the boundary interface 
   */
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Array_t &U, const CellIndex &iCell_boundary, const PrimState &q_in_reconstructed, const CellMetaData &metadata) const
  {
    return impl.getBoundaryFlux(*this, U, iCell_boundary, q_in_reconstructed, metadata);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr static bool has_postProcess()
  {return HyperbolicPolicy_impl::has_postProcess();}

  /**
   * @brief Add a post-processing step after the solver
   * 
   * This is usually a light sub-grid processing, for example to clean negative values for pressure or density
   **/
  KOKKOS_INLINE_FUNCTION
  ConsState postProcess( const ConsState &u ) const
  {
    if constexpr (has_postProcess())
      return impl.postProcess(u);
    else 
      return u;
  }

  void printWarnings() const
  {
    impl.printWarnings();
  }
};

template< class T >
struct is_HyperbolicPolicy 
  : std::false_type
{};

template< class U >
struct is_HyperbolicPolicy<HyperbolicPolicy_base<U>> 
  : std::true_type
{};

template< class T >
inline constexpr bool is_HyperbolicPolicy_v 
  = is_HyperbolicPolicy<T>::value;

} // namespace dyablo
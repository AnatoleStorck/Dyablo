#include "HyperbolicPolicy_base.h"

#pragma once

namespace dyablo {

template< typename State_t, int N_passive_scalars >
struct HyperbolicPolicy_State_passive_scalar : public State_t
{
  constexpr static int max_scalars = N_passive_scalars;

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    auto fields_info = State_t::getFieldsInfo();
    int nvars = fields_info.size();
    for (int i=0; i < max_scalars; ++i) {
      std::ostringstream oss;
      oss << "rho_scalar_" << i;
      fields_info.push_back({oss.str(), nvars + i});
    }
    return fields_info;
  } 

  real_t rho_scalar[max_scalars];
};

template< typename T, typename... Ts>
KOKKOS_INLINE_FUNCTION
auto State_tuple_prepend( T& v, const State_tuple<Ts...>& tuple )
{
  return State_tuple<T&, Ts...>(v,tuple);
}

template< typename State_t, int N_passive_scalars >
struct State_traits< HyperbolicPolicy_State_passive_scalar<State_t, N_passive_scalars> >
{
  static constexpr bool is_state = true;
  static constexpr int nvars = State_traits<State_t>::nvars + N_passive_scalars;
  KOKKOS_INLINE_FUNCTION
  static auto as_tuple( HyperbolicPolicy_State_passive_scalar<State_t, N_passive_scalars>& s )
  {
    return State_tuple_prepend( s.rho_scalar, State_traits<State_t>::as_tuple(static_cast<State_t&>(s)));
  }
};

template< typename State_t, int N_passive_scalars >
struct State_traits< const HyperbolicPolicy_State_passive_scalar<State_t, N_passive_scalars> >
{
  static constexpr bool is_state = true;
  static constexpr int nvars = State_traits<State_t>::nvars + N_passive_scalars;
  KOKKOS_INLINE_FUNCTION
  static const auto as_tuple( const HyperbolicPolicy_State_passive_scalar<State_t, N_passive_scalars>& s )
  {
    return State_tuple_prepend( s.rho_scalar, State_traits<const State_t>::as_tuple(static_cast<const State_t&>(s)));
  }
};

template< typename BasePolicy, int N_passive_scalars >
class HyperbolicPolicy_passive_scalars_impl 
  : protected BasePolicy
{
private:
  using FieldAccessor = BasePolicy::FieldAccessor;
  using CellIndex = BasePolicy::CellIndex;

protected:
  int nscalars;
  using BasePrimState = BasePolicy::PrimState;
  using BaseConsState = BasePolicy::ConsState;
  using CellMetaData = ForeachCell::CellMetaData;
public:
  using PrimState = HyperbolicPolicy_State_passive_scalar<BasePrimState, N_passive_scalars>;
  using ConsState = HyperbolicPolicy_State_passive_scalar<BaseConsState, N_passive_scalars>;

  struct Params 
  {
    BasePolicy::Params hydro_params;
    int nscalars;
  };

  static Params getParams( ConfigMap& configMap )
  {
    return Params{
      .hydro_params = BasePolicy::getParams(configMap),
      .nscalars = configMap.getValue<int>("run",  "n_passive_scalars"),
    };
  }

  HyperbolicPolicy_passive_scalars_impl( const Params& params, const ScalarSimulationData& scalar_data )
  : BasePolicy(params.hydro_params, scalar_data),
    nscalars( params.nscalars )
  {
    DYABLO_ASSERT_HOST_RELEASE(
      params.nscalars <= N_passive_scalars, 
      "Number of passive scalars (" << params.nscalars << ") is larger than maximum allowed (" << N_passive_scalars << ")"
    ); 
  }

  FieldAccessor getUin( UserData& U ) const
  {
    auto Uin_fieldinfo = ConsState::getFieldsInfo();
    return U.getAccessor( Uin_fieldinfo );
  }

  FieldAccessor getUout( UserData& U ) const
  {
    auto Uout_fieldinfo = ConsState::getFieldsInfo();
    for (auto &v: Uout_fieldinfo)
      v.name += "_next";
    return U.getAccessor( Uout_fieldinfo );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    ConsState u = {BasePolicy::getConsState( U, iCell )};
    constexpr int scalar0_ivar = State_traits<BaseConsState>::nvars;
    for (int i=0; i < nscalars; ++i)
      u.rho_scalar[i] = U.at(iCell, scalar0_ivar + i);
    return u;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    BasePolicy::setConsState( U, iCell, u );
    constexpr int scalar0_ivar = State_traits<BaseConsState>::nvars;
    for (int i=0; i < nscalars; ++i)
      U.at(iCell, scalar0_ivar + i) = u.rho_scalar[i];
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    BasePolicy::atomic_addConsState( U, iCell, u );
    constexpr int scalar0_ivar = State_traits<BaseConsState>::nvars;
    for (int i=0; i < nscalars; ++i)
      Kokkos::atomic_add(&U.at(iCell, scalar0_ivar + i), u.rho_scalar[i]);
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    PrimState q = { BasePolicy::getPrimState(Q, iCell) };
    constexpr int scalar0_ivar = State_traits<BasePrimState>::nvars;
    for (int i=0; i < nscalars; ++i)
      q.rho_scalar[i] = Q.at(iCell, scalar0_ivar + i);
    return q;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    BasePolicy::setPrimState( Q, iCell, q );
    constexpr int scalar0_ivar = State_traits<BasePrimState>::nvars;
    for (int i=0; i < nscalars; ++i)
      Q.at(iCell, scalar0_ivar + i) = q.rho_scalar[i];
  }

  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& U ) const
  {
    PrimState Q = { BasePolicy::consToPrim( U ) };
    const real_t inv_rho = U.rho != 0 ? 1.0 / U.rho : 0.0;
    for (int i=0; i < nscalars; ++i)
      Q.rho_scalar[i] = U.rho_scalar[i] * inv_rho;
    return Q;
  }

  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& Q ) const
  {
    ConsState U = {BasePolicy::primToCons( Q )};
    for (int i=0; i < nscalars; ++i)
      U.rho_scalar[i] = Q.rho_scalar[i] * Q.rho;
    return U;
  }

  KOKKOS_INLINE_FUNCTION
  ConsState riemann_solver( PrimState qL, PrimState qR, ComponentIndex3D dir ) const
  {
    real_t ustar = 0;
    ConsState flux = {BasePolicy::riemann_solver(qL, qR, dir, ustar)};
    auto &qref = (ustar > 0 ? qL : qR);
    for (int i=0; i < nscalars; ++i)
      flux.rho_scalar[i] = flux.rho * qref.rho_scalar[i];
    return flux;
  }

  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    PrimState slope = {BasePolicy::compute_slope(qL,qC,qR,dL,dR)}; // Slopes are set to 0 for passive scalars since they are not needed
    return slope;
  }

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Policy_t      &policy, 
                              const Array_t       &U, 
                              const CellIndex     &iCell_boundary, 
                              const CellMetaData  &metadata) const 
  {
    ConsState boundary_value = {BasePolicy::getBoundaryValue(static_cast<const BasePolicy&>(policy.get_impl()), U, iCell_boundary, metadata)};
    CellIndex iCell_inside;
    typename CellIndex::offset_t offset;   
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);  
    auto sign = [](int x){return (x>0)-(x<0);}; 
    typename CellIndex::offset_t symmetric_offset {
      (int16_t)(-offset[IX] + sign(offset[IX])), 
      (int16_t)(-offset[IY] + sign(offset[IY])), 
      (int16_t)(-offset[IZ] + sign(offset[IZ]))
    };  
    CellIndex iCell_sym = iCell_inside + symmetric_offset;
    ConsState u_sym = policy.getConsState( U, iCell_sym );    
    for (int i=0; i < nscalars; ++i)
      boundary_value.rho_scalar[i] = u_sym.rho_scalar[i];
    return boundary_value;
  }

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux(  const Policy_t      &policy, 
                              const Array_t       &U, 
                              const CellIndex     &iCell_boundary, 
                              const PrimState     &q_in_reconstructed,
                              const CellMetaData  &metadata) const 
  {
    ConsState boundary_flux = {BasePolicy::getBoundaryFlux(static_cast<const BasePolicy&>(policy.get_impl()), U, iCell_boundary, q_in_reconstructed, metadata)};
    for (int i=0; i < nscalars; ++i)
      boundary_flux.rho_scalar[i] = q_in_reconstructed.rho_scalar[i] * boundary_flux.rho;
    return boundary_flux;
  }

  using BasePolicy::has_postProcess;
  KOKKOS_INLINE_FUNCTION
  ConsState postProcess( const ConsState &u ) const
  {
    if constexpr (has_postProcess())
    {
      ConsState res = {BasePolicy::postProcess(u)};
      for (int i=0; i < nscalars; ++i)
        res.rho_scalar[i] = u.rho_scalar[i];
      return res;
    }
    else 
      return u;
  }
  using BasePolicy::printWarnings;
};

} // namespace dyablo
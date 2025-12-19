/**
 * Define arithmetic operations on States. States are structures containing 
 * variables for a cell (or a particle) needed inside kernels. 
 * 
 * A state `State_t` contains N variables of type `real_t`.
 * To enable arithmetic operators +,-,*,/,+=,-=,*=,/= as well as `state_get<I>(state)` and  `state_foreach_var()` :
 * * DECLARE_STATE_TYPE( State_t, N ) must be called to declare State_t as a state
 * * An index for every member variable must be set with DECLARE_STATE_GET( State_t, <index>, <member> )
 *      Every index from 0 to N-1 must be set
 * See State_hydro.h for an example
 **/

#pragma once

#include "real_type.h"
#include "kokkos_shared.h"

namespace dyablo {


namespace {

template<int N>
struct as_tuple_t
{
    template<typename S>
    KOKKOS_INLINE_FUNCTION
    static auto as_tuple(S& s)
    {
        static_assert( !std::is_same_v<S,S>, "as_tuple, not defined for this size" );
    }
};

#define DEFINE_AS_TUPLE(N, ...) \
template<> \
struct as_tuple_t<N> \
{ \
    template<typename S> \
    KOKKOS_INLINE_FUNCTION \
    static auto as_tuple(S& s) \
    { \
        auto& [__VA_ARGS__] = s; \
        return std::forward_as_tuple(__VA_ARGS__); \
    } \
};

DEFINE_AS_TUPLE(1,e0)
DEFINE_AS_TUPLE(2,e0,e1)
DEFINE_AS_TUPLE(3,e0,e1,e2)
DEFINE_AS_TUPLE(4,e0,e1,e2,e3)
DEFINE_AS_TUPLE(5,e0,e1,e2,e3,e4)
DEFINE_AS_TUPLE(6,e0,e1,e2,e3,e4,e5)
DEFINE_AS_TUPLE(7,e0,e1,e2,e3,e4,e5,e6)
DEFINE_AS_TUPLE(8,e0,e1,e2,e3,e4,e5,e6,e7)
DEFINE_AS_TUPLE(9,e0,e1,e2,e3,e4,e5,e6,e7,e8)
DEFINE_AS_TUPLE(10,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9)
DEFINE_AS_TUPLE(11,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10)
DEFINE_AS_TUPLE(12,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11)
DEFINE_AS_TUPLE(13,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12)
DEFINE_AS_TUPLE(14,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13)
DEFINE_AS_TUPLE(15,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14)

template <int I, typename T> 
using state_tuple_elt_t = std::remove_reference_t<std::tuple_element_t<I,T>>;

template< int I=0, typename F, typename... Tuple_t >
KOKKOS_INLINE_FUNCTION
void state_foreach_aux( const F& f, const Tuple_t&... t )
{
    constexpr size_t N = (std::tuple_size<Tuple_t>(),...);
    static_assert( ((N == std::tuple_size<Tuple_t>()) && ...), "state_foreach : tuples not the same size" );

    constexpr bool is_array = ((std::is_bounded_array_v<state_tuple_elt_t<I,Tuple_t>>) && ...);
    static_assert( ((is_array == std::is_bounded_array_v<state_tuple_elt_t<I,Tuple_t>>) && ...), "state_foreach : tuples don't have the same arrays" );
    if constexpr( is_array )
    {
        //constexpr size_t array_len = (std::size(state_tuple_elt_t<I,Tuple_t>{}),...);
        constexpr size_t array_len = std::min({std::size(state_tuple_elt_t<I,Tuple_t>{}) ...});
        static_assert( ((array_len == std::size(state_tuple_elt_t<I,Tuple_t>{})) && ...), "state_foreach : arrays not the same size" );
        
        for(int i=0; i<array_len; i++)
        {
            f( std::get<I>(t)[i]... );
        }
    }
    else
    {
        f( std::get<I>(t)... );
    }
    
    if constexpr( I+1 < N )
        state_foreach_aux<I+1>(f, t...);
}

} // namespace

/// By default T is not a State
template<typename T>
struct State_traits
{
    static constexpr bool is_state = false;
};

#define DECLARE_STATE_TYPE_ARRAY_AUX(constness, State, N_VARS, N_FIELDS ) \
template<> \
struct State_traits<constness State> \
{ \
    static constexpr bool is_state = true; \
    static constexpr int nvars = N_VARS; \
    KOKKOS_INLINE_FUNCTION\
    static constness auto as_tuple( constness State& s ) \
    { \
        return as_tuple_t<N_FIELDS>::as_tuple(s); \
    } \
}; \

/// Type-trait for States without arrays
#define DECLARE_STATE_TYPE( State, N_VARS) \
DECLARE_STATE_TYPE_ARRAY_AUX(, State, N_VARS, N_VARS ) \
DECLARE_STATE_TYPE_ARRAY_AUX(const, State, N_VARS, N_VARS ) \

/// Type-trait for States with arrays
/// N_VARS is the total number of vars (summing array lengths)
/// N_FIELDS are the number of fields in struct : (1 array = 1 field)
#define DECLARE_STATE_TYPE_ARRAY( State, N_VARS, N_FIELDS ) \
DECLARE_STATE_TYPE_ARRAY_AUX(, State, N_VARS, N_FIELDS ) \
DECLARE_STATE_TYPE_ARRAY_AUX(const, State, N_VARS, N_FIELDS ) \

#define DECLARE_STATE_GET( State, I, var ) /*empty*/

/**
 * Iterate over each member variable for a set of states
 * @tparam I start index (mainly here for metaprogramming purpose)
 * @param states... states to read or modify. They can be const or not.
 *                  Mixing state types is not advised
 * @param f function to apply to each field in states of type
 *          f : (real_t(&), real_t(&), ...) -> void
 *          one real_t for each const State& in `states...`
 *          one real& for each State& in `states`
 *          e.g. state_foreach_var( [](real_t&, real_t, real_t){...}, State&, const State&, const State& );
 **/
template< typename F, typename... State_t >
KOKKOS_INLINE_FUNCTION
void state_foreach_var( const F& f, State_t&... states )
{
    state_foreach_aux( f, State_traits<State_t>::as_tuple(states)... );
}

//################
// Arithmetic operators on states
//################

// Operator +
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l+r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l+rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator+(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs+r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator+=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l+=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator+=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l+=rhs;}, lhs );
    return lhs;
}

// Operator -
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l-r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l-rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator-(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs-r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator-=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l-=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator-=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l-=rhs;}, lhs );
    return lhs;
}

// Operator *
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l*r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l*rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator*(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs*r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator*=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l*=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator*=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l*=rhs;}, lhs );
    return lhs;
}

// Operator /
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator/(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l/r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator/(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l/rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator/(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs/r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator/=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l/=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator/=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l/=rhs;}, lhs );
    return lhs;
}


}
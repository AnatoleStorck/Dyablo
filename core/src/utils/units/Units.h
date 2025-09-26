#pragma once

#include "real_type.h"
#include "Kokkos_Core.hpp"
#include "../misc/Dyablo_assert.h"

namespace dyablo {

namespace Units{

namespace {

[[maybe_unused]] 
KOKKOS_INLINE_FUNCTION
constexpr real_t pow_int(real_t x, int exp) 
{
    if( exp < 0 )
        return 1 / pow_int( x, -exp );
    if( exp == 0)
       return 1;
    real_t temp = pow_int(x, exp/2);       
    if (exp%2 == 0)
        return temp*temp;
    else 
        return x*temp*temp;
}
} // namespace

/**
 * Represents a unit with it's dimensionnality as template parameters and 
 * it's value in arbitrary common units. 
 * 
 * @tparams Dims... dimensions for dimensionnal analysis (the names are probably explicit enough)
 * 
 * Notes:
 * - 'Units' represent physical units as well as quantities and may be used in 
 *   simple formulas for compile-time dimensional analysis 
 * - (Using SI as arbitrary common unit is only internal implementation details, 
 *    besides SI really is the only true real unit system)
 */
template<int Time=0, int Length=0, int Mass=0, int Current=0, int Temperature=0, int mol=0, int LuminousIntensity=0>
class Unit
{
public:
    real_t value_SI; // Value expressed in SI unit

public:
    /**
     * Construct a unit from it's value in SI units
     * /!\ please only create new units from multiplying/dividing existing units
     * This constructor is explicit to force correct dimensionnal analysis when using floats
     **/
    KOKKOS_INLINE_FUNCTION
    constexpr explicit Unit(real_t value_SI)
    : value_SI(value_SI)
    {}

    /**
     * Get the value of current 'Unit' (actually a quantity) expressed in unit `unit`
     * @tparam Dims... must be the same as current unit dims, this is only here for the error message
     * Note : this is the only way to convert from `Unit` to 
     *        `real_t` that ensures correct dimensional analysis
     **/
    template<int... Dims>
    KOKKOS_INLINE_FUNCTION
    constexpr real_t convert_to( const Unit<Dims...>& unit ) const
    {
        static_assert( std::is_same_v<Unit, Unit<Dims...>>, "Unit conversion error : dimension mismatch" );
        return value_SI / unit.value_SI;
    }

    /// Quantities of same dimensionnality can be added/substracted to help with dimensionnal analysis
    template<int... Dims>
    KOKKOS_INLINE_FUNCTION
    constexpr Unit operator+( const Unit<Dims...>& u2 ) const
    {
        static_assert( std::is_same_v<Unit, Unit<Dims...>>, "Unit addition error : dimension mismatch" );
        return Unit(value_SI + u2.value_SI);
    }

    template<int... Dims>
    KOKKOS_INLINE_FUNCTION
    constexpr Unit operator-( const Unit<Dims...> u2 ) const
    {
        static_assert( std::is_same_v<Unit, Unit<Dims...>>, "Unit substraction error : dimension mismatch" );
        return Unit(value_SI - u2.value_SI);
    }

    template<int p, int... Dims>
    constexpr static Unit<p*Dims...> pow(const Unit<Dims...>& u)
    {
        return Unit<p*Dims...>(pow_int( u.value_SI, p ));
    }

    template<int p>
    KOKKOS_INLINE_FUNCTION
    constexpr auto pow() const
    {
        return pow<p>(*this);
    }
};

/// Multiply Units to create derived units
template< int... Dims1, int... Dims2 >
KOKKOS_INLINE_FUNCTION
constexpr Unit<Dims1+Dims2...> operator*(const Unit<Dims1...>& u1, const Unit<Dims2...>& u2 )
{
    return Unit<Dims1+Dims2...>( u1.value_SI * u2.value_SI );
}

/// Divide Units to create derived units
template< int... Dims1, int... Dims2 >
KOKKOS_INLINE_FUNCTION
constexpr Unit<Dims1-Dims2...> operator/(const Unit<Dims1...>& u1, const Unit<Dims2...>& u2 )
{
    return Unit<Dims1-Dims2...>( u1.value_SI / u2.value_SI );
}

/**
 * Helper template function to perform controlled conversions from other types to units
 * Version for Units : Units are already units
 **/ 
template< int... Dims >
KOKKOS_INLINE_FUNCTION 
constexpr Unit<Dims...> to_unit( const Unit<Dims...>& u )
{
    return u;
}
/**
 * Helper template function to perform controlled conversions from other types to units
 * Version for floats : Automatically convert floats to dimensionless Units (in operators only)
 **/ 
KOKKOS_INLINE_FUNCTION
constexpr Unit<> to_unit( real_t val )
{
    return Unit<>(val);
}

template<typename T>
struct Unit_traits
{
    constexpr static bool is_unit = false;
};

template< int T, int L, int M, int C, int Temp, int Mol, int LI >
struct Unit_traits<Unit<T, L, M, C, Temp, Mol, LI>>
{
    constexpr static bool is_unit = true;
    constexpr static int Time_exp = T;
    constexpr static int Length_exp = L;
    constexpr static int Mass_exp = M;
    constexpr static int Current_exp = C;
    constexpr static int Temp_exp = Temp;
    constexpr static int Mol_exp = Mol;
    constexpr static int LuminousIntensity_exp = LI;
};

/// Is type a unit or not?
template<typename T>
constexpr bool is_unit = Unit_traits<T>::is_unit;

//----------------------------
// Operators for units with automatic type conversion using to_unit()
//----------------------------

// Don't overload operators when units are not involved
template<typename... Ts>
using enable_if_has_unit = std::enable_if_t<((is_unit<Ts> || ...)), int>;

/// Units can be multiplied with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0 >
KOKKOS_INLINE_FUNCTION
constexpr auto operator*(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ) * to_unit( u2 );
}

/// Units can be divided with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0  >
KOKKOS_INLINE_FUNCTION
constexpr auto operator/(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ) / to_unit( u2 );
}

/// Units can be added with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0  >
KOKKOS_INLINE_FUNCTION
constexpr auto operator+(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ).operator+(to_unit( u2 ));
}

/// Units can be substracted with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0  >
KOKKOS_INLINE_FUNCTION
constexpr auto operator-(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ).operator-(to_unit( u2 ));
}

#define DEFINE_UNIT(name, ...) constexpr auto name() {return __VA_ARGS__;}


// SI units = 1
DEFINE_UNIT( second     , Unit<1>              (1) );
DEFINE_UNIT( meter      , Unit<0,1>            (1) );
DEFINE_UNIT( kilogram   , Unit<0,0,1>          (1) );
DEFINE_UNIT( Ampere     , Unit<0,0,0,1>        (1) );
DEFINE_UNIT( Kelvin     , Unit<0,0,0,0,1>      (1) );
DEFINE_UNIT( mol        , Unit<0,0,0,0,0,1>    (1) );
DEFINE_UNIT( candela    , Unit<0,0,0,0,0,0,1>  (1) );

// Unit multiplicators
DEFINE_UNIT( one        , Unit<>(1) );
DEFINE_UNIT( Giga       , 1e9 * one());
DEFINE_UNIT( Mega       , 1e6 * one());
DEFINE_UNIT( Kilo       , 1e3 * one());
DEFINE_UNIT( centi      , 1e-2 * one());
DEFINE_UNIT( milli      , 1e-3 * one());

// MKS units
DEFINE_UNIT( kg         , kilogram() );
DEFINE_UNIT( m          , meter() );
DEFINE_UNIT( s          , second() );
DEFINE_UNIT( km         , Kilo()*meter() );
DEFINE_UNIT( K          , Kelvin() );
DEFINE_UNIT( Pa         , kg() / m() / s() / s() );
DEFINE_UNIT( cd         , candela() );
DEFINE_UNIT( Joule      , kg() * m()*m() /s()/s() );
DEFINE_UNIT( Newton     , kg() * m() /s()/s() );
DEFINE_UNIT( J          , Joule() );
DEFINE_UNIT( N          , Newton() );

// CGS units
DEFINE_UNIT( gram       , milli() * kilogram() );
DEFINE_UNIT( g          , gram() );
DEFINE_UNIT( centimeter , centi() * meter() );
DEFINE_UNIT( cm         , centimeter() );
DEFINE_UNIT( cm_per_s   , cm() / s() );
DEFINE_UNIT( erg        , 1e-7 * Joule() );

// Common astro units
DEFINE_UNIT( parsec     , 3.085678e16 * meter() );
DEFINE_UNIT( pc         , parsec()  );
DEFINE_UNIT( kpc        , Kilo() * parsec() );
DEFINE_UNIT( Mpc        , Mega() * parsec() );
DEFINE_UNIT( Gpc        , Giga() * parsec() );

DEFINE_UNIT( astronomical_unit, 149597870700.0 * meter() );
DEFINE_UNIT( au         , astronomical_unit() );
DEFINE_UNIT( solar_mass , 1.9884e30 * kilogram() );
DEFINE_UNIT( Msun       , solar_mass() );

DEFINE_UNIT( minute     , 60 * second() );
DEFINE_UNIT( hour       , 60 * minute() );
DEFINE_UNIT( year       , 8760.0 * hour() );
DEFINE_UNIT( yr         , year() );
DEFINE_UNIT( kyr        , Kilo() * year() );
DEFINE_UNIT( Myr        , Mega() * year() );
DEFINE_UNIT( Gyr        , Giga() * year() );

DEFINE_UNIT( Dalton, 1.6605388628e-27 * kilogram() );
DEFINE_UNIT( atomic_mass_unit, Dalton() );
DEFINE_UNIT( amu        , Dalton() );
DEFINE_UNIT( electronvolt, 1.602176634e-19 * Joule() );
DEFINE_UNIT( eV         , electronvolt() );

// Convenient multiples
DEFINE_UNIT( m2         , m() * m() );
DEFINE_UNIT( m3         , m() * m() * m() );
DEFINE_UNIT( cm2        , cm() * cm() );
DEFINE_UNIT( cm3        , cm() * cm() * cm() );
DEFINE_UNIT( s2         , s() * s() );

// Other units
DEFINE_UNIT( atom        , mol() / 6.02214076e23 );

// Constants
DEFINE_UNIT( KBOLTZ     , 1.3806e-23 * Joule() / Kelvin() );
DEFINE_UNIT( PROTON_MASS, 1.67262158e-27 * kilogram() );
DEFINE_UNIT( MHE_OVER_MH, 4.002 * one() );
DEFINE_UNIT( HELIUM_MASS, MHE_OVER_MH() * PROTON_MASS() );
DEFINE_UNIT( NEWTON_G   , 6.67408e-11 * Newton() * (meter()*meter()) / (kilogram()*kilogram()) );
DEFINE_UNIT( SOLAR_MASS , solar_mass() );
DEFINE_UNIT( SPEEDOFLIGHT, 299792458 * meter() / second() );
DEFINE_UNIT( H0         , 70.3 * km() / s() / Mpc() );
DEFINE_UNIT( XH         , 0.76 * one() ); // Hydrogen Mass fraction
DEFINE_UNIT( YHE        , 0.24 * one() ); // Helium Mass fraction
DEFINE_UNIT( yHE        , (YHE()/(1.-YHE())/MHE_OVER_MH()) ); // Helium number fraction

using Time          = Unit<1>;
using Length        = Unit<0,1>;
using Mass          = Unit<0,0,1>;
using Current       = Unit<0,0,0,1>;
using Temperature   = Unit<0,0,0,0,1>;
using Mol           = Unit<0,0,0,0,0,1>;
using LuminousIntensity= Unit<0,0,0,0,0,0,1>;

using Area          = decltype( m2() );
using Volume        = decltype( m3() );
using Velocity      = decltype( m()/s() );
using Acceleration  = decltype( m()/s2() );
using Density       = decltype( kg()/m3() );
using Energy        = decltype( J() );
using EnergyDensity = decltype( J()/m3() );
using Pressure      = decltype( Pa() );


class UnitSystem
{
protected:
    Unit<1> time;
    Unit<0,1> length;
    Unit<0,0,1> mass;
    Unit<0,0,0,1> current;
    Unit<0,0,0,0,1> temp;
    Unit<0,0,0,0,0,1> mol;
    Unit<0,0,0,0,0,0,1> luminousIntensity;

public:
    UnitSystem( Unit<1> time = Units::second(), Unit<0,1> length = Units::meter(), Unit<0,0,1> mass = Units::kg(), 
                Unit<0,0,0,1> current = Units::Ampere(), Unit<0,0,0,0,1> temp = Units::Kelvin(), Unit<0,0,0,0,0,1> mol = Units::mol(), Unit<0,0,0,0,0,0,1> luminousIntensity = Units::candela() )
    : time(time), length(length), mass(mass), 
      current(current), temp(temp), mol(mol), luminousIntensity(luminousIntensity)
    {}

    template<typename Unit_t>
    KOKKOS_INLINE_FUNCTION
    Unit_t getUnit( const Unit_t& u = Unit_t(1) ) const
    {
        using traits = Unit_traits<Unit_t>;

        return time     .pow<traits::Time_exp>() 
             * length   .pow<traits::Length_exp>() 
             * mass     .pow<traits::Mass_exp>()
             * current  .pow<traits::Current_exp>()
             * temp     .pow<traits::Temp_exp>()
             * mol      .pow<traits::Mol_exp>()
             * luminousIntensity.pow<traits::LuminousIntensity_exp>();
    }
};

void code_units_init( const UnitSystem& code_units );

const UnitSystem& code_units();

template< typename Unit_t >
real_t constant_to_code_units( const Unit_t& cst )
{
    return cst.convert_to( code_units().getUnit(cst) );
}

namespace Impl{
namespace{

template<typename Unit_t>
KOKKOS_INLINE_FUNCTION
constexpr int get_aexp_exponent()
{
    using traits = Unit_traits<Unit_t>;
    return - 2 * traits::Time_exp - traits::Length_exp;
}

}
}

template< typename Unit_t >
KOKKOS_INLINE_FUNCTION
real_t physical_to_supercomoving( const real_t val, const real_t aexp )
{
    return val * pow_int(aexp, Impl::get_aexp_exponent<Unit_t>());
}

template< typename Unit_t >
KOKKOS_INLINE_FUNCTION
real_t supercomoving_to_physical( const real_t val, const real_t aexp )
{
    return val / pow_int(aexp, Impl::get_aexp_exponent<Unit_t>());
}



} // namespace Units


} //namespace dyablo
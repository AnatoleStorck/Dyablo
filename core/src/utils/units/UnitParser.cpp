#include <UnitParser.hpp>

#include <units/units.hpp>
#include "utils/misc/Dyablo_assert.h"

namespace dyablo{
namespace Units{

real_t parse_unit( const std::string& str, const UnitDims& expected_dims )
{
    using namespace units;

    precise_measurement u = measurement_from_string( str );

    DYABLO_ASSERT_HOST_RELEASE( !is_error(u.units()), "Could not parse unit `" << str << "' :  '" << to_string(u) <<  "'"  );
    
    bool Time_match     = ( expected_dims.Time      == u.units().base_units().second()  );
    bool Length_match   = ( expected_dims.Length    == u.units().base_units().meter()   );
    bool Mass_match     = ( expected_dims.Mass      == u.units().base_units().kg()      );
    bool Current_match  = ( expected_dims.Current   == u.units().base_units().ampere()  );
    bool Temp_match     = ( expected_dims.Temp      == u.units().base_units().kelvin()  );
    bool mol_match      = ( expected_dims.mol       == u.units().base_units().mole()    );
    bool LuminousIntensity_match = ( expected_dims.LuminousIntensity == u.units().base_units().candela() );

    DYABLO_ASSERT_HOST_RELEASE(
           Time_match
        && Length_match
        && Mass_match
        && Current_match
        && Temp_match
        && mol_match
        && LuminousIntensity_match
         , "Error while parsing unit : bad dimension " << std::endl
        << "  Expected : <" << expected_dims.Time << "," << expected_dims.Length << "," << expected_dims.Mass << "," << expected_dims.Current << "," << expected_dims.Temp << "," << expected_dims.mol << ">" << std::endl
        << "  Provided : '" << str << "' : " << "<" << u.units().base_units().second() << "," << u.units().base_units().meter() << "," << u.units().base_units().kg() << "," << u.units().base_units().ampere() << "," << u.units().base_units().kelvin() << "," << u.units().base_units().mole() << ">" << std::endl;
    );

    return u.convert_to_base().value();

}

} // namespace Units
} // namespace dyablo
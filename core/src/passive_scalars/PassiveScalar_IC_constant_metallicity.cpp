#include "PassiveScalar_IC_base.h"

// Sets, on an arbitrary density field, evenly (mass) distributed number densities for a list of metals given a total metallicity.

namespace dyablo{

struct PassiveScalar_IC_constant_metallicity : public PassiveScalar_IC {
  ForeachCell &foreach_cell;

  //std::vector<std::string> passive_names;

  const real_t metallicity;
  const std::vector<std::string> metals;
  const std::vector<real_t> metal_mass;
  const std::vector<std::string> ions;
  const std::vector<real_t> ion_fracs;

  PassiveScalar_IC_constant_metallicity(  
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers,
        std::string passive_scalar_names) : 
    foreach_cell(foreach_cell),
    //passive_names(passive_scalar_names),
    metallicity(configMap.getValue<real_t>("constant_metallicity", "metallicity", 0.0)),
    metals(configMap.getValue<std::vector<std::string>>("constant_metallicity", "metals", std::vector<std::string>{})),
    metal_mass(configMap.getValue<std::vector<real_t>>("constant_metallicity", "metal_mass", std::vector<real_t>{})),
    ions(configMap.getValue<std::vector<std::string>>("constant_metallicity", "ions", std::vector<std::string>{})),
    ion_fracs(configMap.getValue<std::vector<real_t>>("constant_metallicity", "ion_fracs", std::vector<real_t>{}))
  {
  }

  void init( UserData &U ) {

    DYABLO_ASSERT_HOST_RELEASE( metals.size() == metal_mass.size(), "PassiveScalar_IC_constant_metallicity : Mismatch between number of metals and metal masses" );

    std::vector<UserData::FieldAccessor::FieldInfo> new_metal_fields_info;
    std::set<std::string> new_metal_fields;
    for( const std::string& metal : metals )
    {
        // add the number density field for the metal
        std::string field = "n" + metal;

        new_metal_fields.insert(field);
        VarIndex ivar = new_metal_fields_info.size();
        new_metal_fields_info.push_back({field, ivar});
    }

    std::vector<UserData::FieldAccessor::FieldInfo> new_ion_fields_info;
    std::set<std::string> new_ion_fields;
    for( const std::string& ion : ions )
    {
        // add the ion fraction field for the metal
        std::string field = ion;

        new_ion_fields.insert(field);
        VarIndex ivar = new_ion_fields_info.size();
        new_ion_fields_info.push_back({field, ivar});
    }

    std::vector<dyablo::UserData::FieldAccessor_FieldInfo> fields_info = {
        {"rho", 0 }
    };


    auto Umetal = U.getAccessor( new_metal_fields_info );
    auto Uion   = U.getAccessor( new_ion_fields_info );
    auto Urho   = U.getAccessor( fields_info );

    const real_t X = 0.76;
    const real_t Y = 0.24;
    const real_t Z = metallicity;

    const real_t metal_count_no_hhe = metals.size() > 2 ? (metals.size() - 2) : 1;

    std::vector<real_t> coeff_host(metals.size(), 0.0);
    for( size_t i=0; i<metals.size(); i++ )
    {
        if( metals[i] == "H" )
        {
            coeff_host[i] = X * (1.0 - Z) / metal_mass[i];
        }
        else if( metals[i] == "He" )
        {
            coeff_host[i] = Y * (1.0 - Z) / metal_mass[i];
        }
        else
        {
            coeff_host[i] = (1.0 / metal_count_no_hhe) * Z / metal_mass[i];
        }
    }

    auto amu_per_cc     = Units::amu() / Units::cm3();
    auto code_density   = Units::code_units().getUnit<Units::Density>();
    auto number_density = Units::number_density();


    Kokkos::View<real_t*> coeff_view("InitialConditions_uniform_metallicity::coeff", coeff_host.size());
    {
        Kokkos::View<real_t*> coeff_view_cpu( coeff_host.data(), coeff_host.size() );
        Kokkos::deep_copy(coeff_view, coeff_view_cpu);
    }


    foreach_cell.foreach_cell( "PassiveScalar_IC_constant_metallicity::fill_U", U.getShape(),
                KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
    {
        const real_t rho = Urho.at(iCell_U, 0);

        for( size_t i=0; i<coeff_view.size(); i++ )
            // Store conserved element number density n_i so hydro transports n_i with mass flux.
            Umetal.at(iCell_U, i) = (rho * code_density).convert_to(amu_per_cc) * coeff_view(i);

        for( size_t i=0; i<ions.size(); i++ )
            Uion.at(iCell_U, i) = ion_fracs[i] * rho;
    });
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::PassiveScalar_IC_Factory, 
                 dyablo::PassiveScalar_IC_constant_metallicity,
                 "constant_metallicity");

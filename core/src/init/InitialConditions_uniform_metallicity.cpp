#include "InitialConditions_base.h"

// Sets, on an arbitrary density field, evenly (mass) distributed number densities for a list of metals given a total metallicity.

namespace dyablo{

class InitialConditions_uniform_metallicity : public InitialConditions
{
protected:
    ForeachCell& foreach_cell;
    real_t metallicity;
    std::vector<std::string> metals;
    std::vector<real_t> metal_mass;

    /// Simple Constructor to build deriverd InitialConditions with static values with static fields and values
    InitialConditions_uniform_metallicity(ForeachCell& foreach_cell)
    : foreach_cell(foreach_cell)
    {}

public:
    InitialConditions_uniform_metallicity(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        :  InitialConditions_uniform_metallicity(foreach_cell)
    {
        this->metallicity   = configMap.getValue<real_t>("InitialConditions_uniform_metallicity", "metallicity", 0.0);
        this->metals        = configMap.getValue<std::vector<std::string>>("InitialConditions_uniform_metallicity", "metals", std::vector<std::string>{} );
        this->metal_mass    = configMap.getValue<std::vector<real_t>>("InitialConditions_uniform_metallicity", "metal_mass", std::vector<real_t>{} );
    }


    void init( UserData& U )
    {
        DYABLO_ASSERT_HOST_RELEASE( metals.size() == metal_mass.size(), "InitialConditions_uniform_metallicity : Mismatch between number of metals and metal masses" );

        std::vector<UserData::FieldAccessor::FieldInfo> new_fields_info;
        std::set<std::string> new_fields;
        for( const std::string& metal : metals )
        {
            // add the number density field for the metal
            std::string field = "n" + metal;

            new_fields.insert(field); 
            VarIndex ivar = new_fields_info.size();
            new_fields_info.push_back({field, ivar});
        }
        U.new_fields( new_fields ); 


        std::vector<dyablo::UserData::FieldAccessor_FieldInfo> fields_info = {
            {"rho", 0 }
        };

        auto Uin  = U.getAccessor( fields_info );
        auto Uout = U.getAccessor( new_fields_info );

        const real_t X_H = 0.76;
        const real_t X_He = 0.24;
        const real_t metal_count_no_hhe = metals.size() > 2 ? (metals.size() - 2) : 1;

        std::vector<real_t> coeff_host(metals.size(), 0.0);
        for( size_t i=0; i<metals.size(); i++ )
        {
            if( metals[i] == "H" )
            {
                coeff_host[i] = X_H * (1.0 - metallicity) / 1.008;
            }
            else if( metals[i] == "He" )
            {
                coeff_host[i] = X_He * (1.0 - metallicity) / 4.0026;
            }
            else
            {
                coeff_host[i] = (1.0 / metal_count_no_hhe) * metallicity / metal_mass[i];
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

        foreach_cell.foreach_cell( "InitialConditions_uniform_metallicity::fill", U.getShape(),
            KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
        {
            for( size_t i=0; i<coeff_view.size(); i++ )
            {
                Uout.at(iCell, i) = ((Uin.at(iCell, 0) * code_density).convert_to(amu_per_cc) * coeff_view(i));
            };
        });
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
    dyablo::InitialConditions_uniform_metallicity, 
    "uniform_metallicity");

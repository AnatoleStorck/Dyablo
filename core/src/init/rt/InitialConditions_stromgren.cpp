#include "../InitialConditions_uniform.h"
#include "Cosmo.h"

namespace dyablo {

    namespace{

    void compute_spawn_rate(ConfigMap& configMap, CosmoManager& cosmo_manager, real_t rho){

        // Compute dx
        real_t xmax = configMap.getValue_in_code_unit<Units::Length>("mesh", "xmax", 1.0);
        real_t levelMin =configMap.getValue<real_t>("amr", "level_min", 6);
        real_t bx = configMap.getValue<real_t>("amr", "bx", 4);
        real_t dx = xmax / (pow(2, levelMin) * bx);

        // Time duration of the production of photons in code units. (Barkana & Loeb 2001, section 6.2)
        real_t a_start = configMap.getValue<real_t>( "cosmology", "astart", 1.0 );
        real_t t_end = configMap.getValue<real_t>("run", "tend", 1.0);
        real_t t_s = (cosmo_manager.compute_physical_t(configMap.getValue<real_t>("rad", "a_stop_emission", t_end)) - cosmo_manager.compute_physical_t(a_start));

        // Compute number of baryons per cell
        real_t n_gamma = configMap.getValue<real_t>("stromgren", "n_gamma", 4000);
        real_t n_b = rho / Units::constant_to_code_units(Units::PROTON_MASS()); // number of baryons/volume unit
        real_t n_h = n_b * dx * dx * dx;

        // Compute photon rate
        real_t mol_to_atoms = Units::code_units().getUnit<Units::Mol>().convert_to(Units::atom());
        real_t spawn_rate = n_gamma * n_h / t_s / mol_to_atoms;  // in code units atoms/Myr
        configMap.getValue<real_t>("stromgren", "spawn_rate", spawn_rate);

        // Compute Vmax and Rmax
        real_t v_max = n_gamma * n_h / n_b;
        real_t r_max = pow(3*v_max/(4*M_PI), 1./3.);
        printf("ts=%f rho=%e nb=%e nh=%e spawn_rate=%e vmax=%f rmax=%f\n", t_s, rho, n_b, n_h, spawn_rate, v_max, r_max);
    }
}

class InitialConditions_stromgren : public InitialConditions_uniform
{
public:

    CosmoManager cosmo_manager;

    InitialConditions_stromgren(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        :  InitialConditions_uniform(foreach_cell),
           cosmo_manager( configMap )
    {

        real_t rho = 0;        

        if( !configMap.getValue<bool>("cosmology", "active", false) ){
            rho = configMap.getValue_in_code_unit<Units::Density>("InitialConditions_stromgren", "gas_density");
        }

        else{

            // Comoving critical density
            using Inv_Time = decltype( 1/Units::s() );
            real_t H0 = configMap.getValue_in_code_unit<Inv_Time>("cosmology", "H0", "70.0 km/s/Mpc");
            real_t density_factor = configMap.getValue<real_t>( "stromgren", "density_factor", 1000.0 );
            real_t rhoc = density_factor * 3.0 * H0 * H0 / (8.0 * M_PI * Units::constant_to_code_units(Units::NEWTON_G()));
            rho = configMap.getValue<real_t>( "cosmology", "omegab", 0.049 ) * rhoc;

            // Compute spawn rate according to (Barkana & Loeb 2001, section 6.2)
            compute_spawn_rate(configMap, cosmo_manager, rho);
        }

        real_t xe_start = configMap.getValue<real_t>("InitialConditions_stromgren", "xe_start", 1.2e-3);
        real_t rho_HII = xe_start * rho;

        real_t temperature = configMap.getValue_in_code_unit<Units::Temperature>("rad", "temperature", "1e4 K");
        
        auto code_density       = Units::code_units().getUnit<Units::Density>();
        auto code_temperature   = Units::code_units().getUnit<Units::Temperature>();
        auto code_energy_density= Units::code_units().getUnit<Units::EnergyDensity>();
        
        auto e_tot_u = 1.5 * rho*code_density * Units::KBOLTZ()/Units::PROTON_MASS() * temperature*code_temperature;
        real_t e_tot = e_tot_u.convert_to(code_energy_density);

        this->fields = {"rho","rho_HII","e_tot",
                        "rho_vx","rho_vy","rho_vz",
                        "e_rad",
                        "fx_rad","fy_rad","fz_rad"};
        this->values = {rho,rho_HII,e_tot,
                        0,0,0,0,0,0,0};
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_stromgren, 
                 "stromgren");

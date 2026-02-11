#include "../InitialConditions_uniform.h"
#include "Cosmo.h"

namespace dyablo {
    namespace{

    void compute_spawn_rate(ConfigMap& configMap, CosmoManager& cosmo_manager, Units::Density rho){

        using namespace Units;
        using SpawnRate = decltype( 1/s() );
        auto& cu = Units::code_units();

        // Compute dx
        auto xmax = configMap.getValue_in_code_unit<Units::Length>("mesh", "xmax", 1.0) * cu.getUnit<Length>();
        int levelMin = configMap.getValue<int>("amr", "level_min", 6);
        int bx = configMap.getValue<int>("amr", "bx", 4);
        auto dx = xmax / (pow(2, levelMin) * bx);

        // Time duration of the production of photons in code units. (Barkana & Loeb 2001, section 6.2)
        real_t a_start = configMap.getValue<real_t>( "cosmology", "astart", 1.0 );
        real_t t_end = configMap.getValue<real_t>("run", "tend", 1.0);
        auto t_s = (cosmo_manager.compute_physical_t(configMap.getValue<real_t>("rad", "a_stop_emission", t_end)) - cosmo_manager.compute_physical_t(a_start)) * cu.getUnit<Time>();

        // Compute number of baryons per cell
        real_t n_gamma = configMap.getValue<real_t>("stromgren", "n_gamma", 4000);
        auto n_b = rho / PROTON_MASS(); // number of baryons/volume unit
        auto n_h = n_b * dx * dx * dx;

        // Compute photon rate
        real_t spawn_rate = (n_gamma * n_h / t_s).convert_to(cu.getUnit<SpawnRate>());
        configMap.getValue<real_t>("stromgren", "spawn_rate", spawn_rate); // Actually writes to .ini

        // Compute Vmax and Rmax
        real_t v_max = (n_gamma * n_h / n_b).convert_to(cu.getUnit<Volume>());
        real_t r_max = pow(3*v_max/(4*M_PI), 1./3.);

        real_t rho_cu = rho.convert_to( cu.getUnit<Density>() );
        real_t ts_cu = t_s.convert_to( cu.getUnit<Time>() );
        real_t n_b_cu = n_b.convert_to( cu.getUnit<decltype(1/m3())>() );
        real_t n_h_cu = n_h.convert_to(one());
        printf("ts=%f rho=%e nb=%e nh=%e spawn_rate=%e vmax=%f rmax=%f\n", ts_cu, rho_cu, n_b_cu, n_h_cu, spawn_rate, v_max, r_max);
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
        using namespace Units;

        using Inv_Time = decltype( 1/Units::s() );

        auto code_density = code_units().getUnit<Density>();
        auto code_inv_time = code_units().getUnit<Inv_Time>();
        auto code_temperature   = code_units().getUnit<Temperature>();
        auto code_energy_density= code_units().getUnit<EnergyDensity>();

        Density rho(0);        

        if( !configMap.getValue<bool>("cosmology", "active", false) ){
            rho = configMap.getValue_in_code_unit<Density>("InitialConditions_stromgren", "gas_density") * code_density;
        }
        else{

            // Comoving critical density
            auto H0 = configMap.getValue_in_code_unit<Inv_Time>("cosmology", "H0", "70.0 km/s/Mpc") * code_inv_time;
            real_t density_factor = configMap.getValue<real_t>( "stromgren", "density_factor", 1000.0 );
            auto rhoc = density_factor * 3.0 * H0 * H0 / (8.0 * M_PI * NEWTON_G());
            real_t omega_b = configMap.getValue<real_t>( "cosmology", "omegab", 0.049 );
            rho = omega_b * rhoc;

            // Compute spawn rate according to (Barkana & Loeb 2001, section 6.2)
            compute_spawn_rate(configMap, cosmo_manager, rho);
        }

        real_t xe_start = configMap.getValue<real_t>("InitialConditions_stromgren", "xe_start", 1.2e-3);
        auto rho_HII = xe_start * rho;

        auto temperature = configMap.getValue_in_code_unit<Temperature>("rad", "temperature", "1e4 K") * code_temperature;
        
        auto e_tot = 1.5 * rho * KBOLTZ()/PROTON_MASS() * temperature;
        
        real_t e_tot_cu = e_tot.convert_to(code_energy_density);
        real_t rho_cu = rho.convert_to(code_density);
        real_t rho_HII_cu = rho_HII.convert_to(code_density);

        this->fields = {"rho","rho_HII","e_tot",
                        "rho_vx","rho_vy","rho_vz",
                        "e_rad",
                        "fx_rad","fy_rad","fz_rad"};
        this->values = {rho_cu,rho_HII_cu,e_tot_cu,
                        0,0,0,0,0,0,0};
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_stromgren, 
                 "stromgren");

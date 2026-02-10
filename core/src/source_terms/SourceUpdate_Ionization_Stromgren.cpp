#include "SourceUpdate_base.h"

namespace dyablo{

  namespace{
    enum VarIndex_Chem{ Irho, Ie_tot, Irho_vx, Irho_vy, Irho_vz, Ie_rad, Ifx_rad, Ify_rad, Ifz_rad, Irho_HII };

    auto sigma_HI(){

       // Table E1 & E2 Rosdahl et al, 2013
        auto S0 = 5.475e-14 * Units::cm2();
        real_t P = 2.963;
        real_t ya = 32.88;
        real_t x = 13.6/0.4298; // eV/eV = > dimensionless

        // Formula E1 from Rosdahl et al, 2013
        auto sig = S0*((x-1.0)*(x-1.0))* pow(x,(0.5*P-5.5)) / pow((1.0 + sqrt(x/ya)), P);

        return sig;
    }

    // Recombination rate from in cm^{3}.s^{-1} (Hui & Gnedin 1997). Valid between 3 and 1e9 K
    KOKKOS_INLINE_FUNCTION
    real_t get_alpha_b(const real_t temp)
    {
        const real_t lambda = 315614.0 / temp;

        real_t alpha_bh = 2.753e-14;
        alpha_bh *= pow(lambda, 1.5);
        alpha_bh /= pow(1.0 + pow(lambda / 2.74, 0.407), 2.242);

        // Convert to m^{3}.s^{-1}
        alpha_bh *= 1e-6;

        return alpha_bh;
    }

    // HI collisional ionisation coefficient in cm^{3}.s^{-1} (Hui & Gnedin 1997). Valid between 1e4 and 1e9 K
    KOKKOS_INLINE_FUNCTION
    real_t get_beta(const real_t temp)
    {
        const real_t lambda = 315614.0 / temp;
 
        real_t beta_h = 21.11 * pow(temp, -3.0 / 2.0) * exp(-lambda / 2.0) * pow(lambda, -1.089);
        beta_h /= pow(1.0 + pow(lambda / 0.354, 0.874), 1.01);

        // Convert to m^{3}.s^{-1}
        beta_h *= 1e-6;

        return beta_h;
    }

    KOKKOS_INLINE_FUNCTION
    real_t solve_3rd_order_polynomial( real_t x, real_t m, real_t n, real_t p, real_t q )
    {
        int nmax = 500;
        real_t error = 1e3;
        real_t tol = 1e-8;

        // Clamp initial guess to [0, 1]
        real_t xold = FMAX(tol, FMIN(1.0, x));
        real_t xnew = xold;
        
        int i = 0;
        while((error > tol) && (i < nmax)){
            real_t f = m*xold*xold*xold + n*xold*xold + p*xold + q;
            real_t df = 3*m*xold*xold + 2*n*xold + p;
            
            // Avoid division by zero
            if(abs(df) < 1e-14){
                xnew = 1.0;
                break;
            }
            
            xnew = xold - f/df;
            
            // Handle NaN
            if(isnan(xnew)){
                xnew = 1.0;
                break;
            }
            
            // Clamp to [0, 1] during iteration
            xnew = FMAX(tol, FMIN(1.0, xnew));
            
            error = abs(xnew - xold);
            xold = xnew;
            i++;
        }

        return xnew;
    }

    KOKKOS_INLINE_FUNCTION
    real_t solve_raphson_newton(const real_t x, const real_t alpha, const real_t alphab, const real_t beta, const real_t sigma_n_c, const real_t nHSI, const real_t NSI, const real_t dt){

        real_t nh_square_dt = nHSI*nHSI*dt;
        real_t m = (alphab + beta)*nh_square_dt;
        real_t n = nHSI - (alpha + beta)*nHSI/sigma_n_c - alphab*nh_square_dt - 2.0*beta*nh_square_dt;
        real_t p = -nHSI*(1+x) - NSI - 1./(sigma_n_c*dt) + beta*nHSI/sigma_n_c + beta*nh_square_dt;
        real_t q = NSI + nHSI*x + x/(sigma_n_c*dt);

        return solve_3rd_order_polynomial(x, m, n, p, q);
    }
  }

/**
 * @brief Ionization source term for the Stromgren sphere test
 */
class SourceUpdate_Ionization_Stromgren : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;

  real_t alphab_physical, beta_physical;
  real_t sigma_n_c;
  real_t c_rad;
  real_t small_erad;
  bool use_recombination;

  using Volume_over_time = decltype( Units::m3() / Units::s() );

public:
  SourceUpdate_Ionization_Stromgren(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),
    c_rad(configMap.getValue_in_code_unit<Units::Velocity>("rad", "c_rad", "speedoflight")),
    small_erad(configMap.getValue<real_t>( "rad", "small_erad", 1e30 )),
    use_recombination(configMap.getValue<bool>( "rad", "use_recombination", true ))
  {
    auto sigma_n = sigma_HI(); // cm2
    real_t sigma_n_cu = sigma_n.convert_to(Units::code_units().getUnit<Units::Area>());
    this->sigma_n_c = sigma_n_cu * this->c_rad;

    // Absorption polynomial coefficients computed in code units
    {
      using namespace Units;

      real_t temperature = configMap.getValue_in_code_unit<Temperature>("rad", "temperature", "1e4 K");
      real_t temperature_SI = (temperature * code_units().getUnit<Temperature>()).convert_to(K());
      auto beta_physical_u = get_beta(temperature_SI) * m3() / s();
      auto alphab_physical_u = get_alpha_b(temperature_SI) * m3() / s();

      auto code_volume_over_time = code_units().getUnit<Volume_over_time>();
      this->beta_physical   = beta_physical_u.convert_to( code_volume_over_time );
      this->alphab_physical = alphab_physical_u.convert_to( code_volume_over_time );
    }
  }

  void update( UserData &U, ScalarSimulationData& scalar_data)
  {
    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("SourceUpdate_Ionization_Stromgren").start();

    UserData::FieldAccessor Uout = U.getAccessor( 
      {
        {"rho_next",    Irho    }, 
        {"e_rad_next",  Ie_rad  }, 
        {"fx_rad_next", Ifx_rad },
        {"fy_rad_next", Ify_rad },
        {"fz_rad_next", Ifz_rad }, 
        {"rho_HII",     Irho_HII}
      });

    real_t dt = scalar_data.get<real_t>("dt");
    real_t aexp = scalar_data.get<real_t>("aexp");

    real_t sigma_n_c = Units::physical_to_supercomoving<Volume_over_time>(this->sigma_n_c, aexp);  // Area * velocity = volume/time
    real_t ctilde = Units::physical_to_supercomoving<Units::Velocity>(this->c_rad, aexp);
    real_t small_erad = this->small_erad;
    bool use_recombination = this->use_recombination;

    real_t beta   = Units::physical_to_supercomoving<Volume_over_time>(this->beta_physical, aexp);
    real_t alphab   = Units::physical_to_supercomoving<Volume_over_time>(this->alphab_physical, aexp);

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    using PhotonDensity = decltype(Units::mol()/Units::m3());
    using AtomDensity = PhotonDensity;

    auto code_atom_density = Units::code_units().getUnit<AtomDensity>();
    real_t proton_mass_cu = Units::constant_to_code_units(Units::PROTON_MASS());

    auto code_length = Units::code_units().getUnit<Units::Length>();
    // TODO : fix (?) N and nH are not in code units but in atoms / code_length^3
    auto N_unit = Units::atom() / (code_length*code_length*code_length); 

    if(!use_recombination) alphab = 0.0;
    real_t alpha  = alphab; // On the spot approximation

    foreach_cell.foreach_cell( "SourceUpdate_Ionization_Stromgren", Uout.getShape(), 
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout) 
    {
      [[maybe_unused]] auto size = cells.getCellSize(iCell_Uout);
      DYABLO_ASSERT_KOKKOS_DEBUG( size[IX] == size[IY] && size[IX] == size[IZ], "Only square cells supported" );

     // Local Gaz density
      real_t rho = Uout.at(iCell_Uout, VarIndex_Chem::Irho);
      
      // Local atom number density. Here we do an approximation and consider a pure hydrogen system without helium. This is why there is no 0.76 factor. 
      real_t nH = rho / proton_mass_cu;   

      // Local photon number density
      real_t N = (Uout.at(iCell_Uout, VarIndex_Chem::Ie_rad) * code_atom_density).convert_to(N_unit);

      // Local ionisation fraction
      real_t x_old = Uout.at(iCell_Uout, VarIndex_Chem::Irho_HII)/rho;

      // Compute new ionisation fraction
      real_t xnew = solve_raphson_newton(x_old, alpha, alphab, beta, sigma_n_c, nH, N, dt);

      // Compute new N value (equation 5 from Aubert & Teyssier 2008)
      real_t N_new = N + beta*nH*nH*(1.0-xnew)*xnew*dt - alphab*nH*nH*xnew*xnew*dt - nH*(xnew-x_old);

      // Avoid negative values
      if(N_new<0) N_new = small_erad;

      // Store results
      Uout.at(iCell_Uout, VarIndex_Chem::Ie_rad) = (N_new * N_unit).convert_to(code_atom_density); 
      Uout.at(iCell_Uout, VarIndex_Chem::Irho_HII) = rho * xnew;

      {
        // Local Flux.
        // In principle we need a full conversion to physical quantites but since F is not used it's not necessary
        real_t fx = Uout.at(iCell_Uout, VarIndex_Chem::Ifx_rad);
        real_t fy = Uout.at(iCell_Uout, VarIndex_Chem::Ify_rad);
        real_t fz = Uout.at(iCell_Uout, VarIndex_Chem::Ifz_rad);

        // Update fluxes
        real_t fact = 1.0 + sigma_n_c*nH*dt*(1-xnew); //fact is dimensionless
        fx = fx/fact;
        fy = fy/fact;
        fz = fz/fact;

        real_t F = sqrt(fx*fx + fy*fy + fz*fz);
        real_t Fred = F/(ctilde*N_new);

        if(Fred > 1.0){
          fx = fx/Fred;
          fy = fy/Fred;
          fz = fz/Fred;
        }

        Uout.at(iCell_Uout, VarIndex_Chem::Ifx_rad) = fx;
        Uout.at(iCell_Uout, VarIndex_Chem::Ify_rad) = fy;
        Uout.at(iCell_Uout, VarIndex_Chem::Ifz_rad) = fz;
      }     
    });

    timers.get("SourceUpdate_Ionization_Stromgren").stop();
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_Ionization_Stromgren, 
                  "SourceUpdate_Ionization_Stromgren" );
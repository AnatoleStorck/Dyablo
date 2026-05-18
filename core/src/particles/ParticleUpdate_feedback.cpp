#include "ParticleUpdate_base.h"
#include "utils/units/Units.h"
#include "ForeachParticle.h"
#include "states/State_hydro.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace timedelay_SNII {

  // Raiteri et al. (1996)
  inline real_t raiteri_ms_mass( real_t t, real_t Z ) {
    
    // clip metallicity
    real_t Z_eff;
    if (Z < 7e-5)       Z_eff = 7e-5;
    else if (Z > 3e-2)  Z_eff = 3e-2;
    else                Z_eff = Z;

    real_t a0 = 10.13 + 0.07547 * std::log10(Z_eff) - 0.008084 * std::pow(std::log10(Z_eff), 2);
    real_t a1 = 4.424 + 0.7939 * std::log10(Z_eff) + 0.1187 * std::pow(std::log10(Z_eff), 2);
    real_t a2 = 1.262 + 0.3385 * std::log10(Z_eff) + 0.05417 * std::pow(std::log10(Z_eff), 2);

    real_t c = a0 - std::log10(t);
    real_t b = a1;
    real_t a = a2;

    real_t discriminant = b*b - 4*a*c;

    real_t mass;
    if (discriminant > 0) {
      mass = (-b + std::sqrt(discriminant)) / (2*a);
      mass = std::pow(10, mass);
    }
    else {
      mass = 120.0;
    }
    return mass;
  }

  // Kroupa (2001), 0.1-100 Msun
  inline real_t kroupa_imf( real_t m1, real_t m2) {
    real_t A = 0.2244557;
    real_t ind = -2.3;
    return (-A/(ind + 1)) * ( std::pow(m2, ind+1) - std::pow(m1, ind+1) );
  }
}



namespace dyablo {

class ParticleUpdate_feedback : public ParticleUpdate {
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  ParticleUpdate_feedback(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers)
  : foreach_cell    ( foreach_cell ),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    timers          ( timers ),
    eta_SNII        ( configMap.getValue<real_t>("star_feedback", "eta_SNII", 0.1) ),
    yield_SNII      ( configMap.getValue<real_t>("star_feedback", "yield_SNII", 0.1) ),
    E_SNII_physical ( configMap.getValue_in_code_unit<Units::Energy>("star_feedback", "E_SNII", "1e51 erg") ),
    M_SNII_physical ( configMap.getValue_in_code_unit<Units::Mass>  ("star_feedback", "M_SNII", "10 solar_mass") ),
    t_SNII_physical ( configMap.getValue_in_code_unit<Units::Time>  ("star_feedback", "t_SNII", "10 Myr") ),
    cosmology       ( configMap.getValue<bool>("cosmology", "active", false) )
  {
  }

  ~ParticleUpdate_feedback() {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {

    const real_t t = cosmology ? scalar_data.get<real_t>("time_physical") : scalar_data.get<real_t>("time");
    const real_t dt = scalar_data.get<real_t>("dt");

    enum VarIndex {
      IRho, IE_tot, IRho_vx, IRho_vy, IRho_vz, IRho_Z,
    };
    enum VarIndex_particle {
      IMASS, IBIRTHMASS, IVX, IVY, IVZ, IBIRTH, IMETAL
    };

    timers.get("ParticleUpdate_feedback").start();

    std::vector<UserData::FieldAccessor_FieldInfo>
      Uin_infos = {{"rho", IRho},    {"e_tot", IE_tot},    {"rho_vx", IRho_vx},    {"rho_vy", IRho_vy},    {"rho_vz", IRho_vz}};
    std::vector<UserData::ParticleAccessor_AttributeInfo>
      pinfos = {{"mass", IMASS}, {"birth_mass", IBIRTHMASS}, {"vx", IVX}, {"vy", IVY}, {"vz", IVZ}, {"birth_time", IBIRTH}};

    bool has_metallicity = U.has_field("metallicity");
    if (has_metallicity) {
      Uin_infos.push_back( {"metallicity", IRho_Z} );
      pinfos.push_back( {"metallicity", IMETAL} );
    }

    // Get accessors
    auto Ppos = U.getParticleArray( "particles" );
    auto Pdata = U.getParticleAccessor( "particles", pinfos );
    auto Uin = U.getAccessor( Uin_infos );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t aexp = scalar_data.get<real_t>("aexp");

    // Gather SN feedback parameters
    const real_t eta_SNII = this->eta_SNII;
    const real_t yield_SNII = this->yield_SNII;
    const real_t E_SNII = Units::physical_to_supercomoving<Units::Energy>(E_SNII_physical, aexp);
    const real_t M_SNII = Units::physical_to_supercomoving<Units::Mass>(M_SNII_physical, aexp);
    const real_t E_per_M_SNII = E_SNII / M_SNII;
    const real_t t_SNII_physical = this->t_SNII_physical;
    const real_t dt_physical = Units::supercomoving_to_physical<Units::Time>(dt, aexp);

    Kokkos::View<int> N_supernovae_view("N_supernovae");
    Kokkos::deep_copy(N_supernovae_view, 0);

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

    foreach_particle.foreach_particle( "particles_update_feedback", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      // Age of the particle
      real_t age_physical = t - Pdata.at(iPart, IBIRTH);



      // get mass of stars that leave MS in current time step based on
      // Raiteri et al. 1996 and assuming a Salpeter IMF.
      // Check if M1-M2 mass range is within CCSN progenitor mass range (8-40 solar masses)
      real_t temp_metallicity = 0.02;

      real_t M1 = timedelay_SNII::raiteri_ms_mass(age_physical, temp_metallicity);
      real_t M2 = timedelay_SNII::raiteri_ms_mass(age_physical + dt_physical, temp_metallicity);

      // If the entire mass range is outside the SNII progenitor mass range, skip
      if (M2 < 8.0 || M1 > 40.0)
        return;
      M1 = std::max(M1, 8.0);
      M2 = std::min(M2, 40.0);

      real_t part_imass = Pdata.at(iPart, IBIRTHMASS);

      real_t num = part_imass * timedelay_SNII::kroupa_imf(M1, M2);
      real_t num_residual = num - int(num);
      num = int(num);

      // get random number between 0 and 1
      auto generator = random_pool.get_state();
      real_t rand_num = generator.frand();
      random_pool.free_state(generator);

      if (rand_num < num_residual)
        num = num + 1;

      // If we have a SNII explosion
      if ( num > 0 ) {

        real_t meanmass = 0.5 * (M1 + M2); // don't need to compute mean mass until here
        real_t Mloss = num * meanmass;

        pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
        pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};

        ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

        pos_t cell_size = cells.getCellSize( iCell );
        real_t cell_volume = cell_size[IX] * cell_size[IY] * cell_size[IZ];

        real_t rho_loss = Mloss / cell_volume;
        real_t ethermal = Mloss * E_per_M_SNII;
        real_t ekin = 0.5 * rho_loss * (
          SQR(part_vel[IX]) + SQR(part_vel[IY]) + SQR(part_vel[IZ])
        );

        // Atomic are mandatory since multiple particles can explode in the same cell
        Kokkos::atomic_add(&Uin.at(iCell, IRho), rho_loss);
        Kokkos::atomic_add(&Uin.at(iCell, IE_tot), ethermal + ekin);
        Kokkos::atomic_add(&Uin.at(iCell, IRho_vx), rho_loss * part_vel[IX]);
        Kokkos::atomic_add(&Uin.at(iCell, IRho_vy), rho_loss * part_vel[IY]);
        Kokkos::atomic_add(&Uin.at(iCell, IRho_vz), rho_loss * part_vel[IZ]);
        if (has_metallicity) {
          real_t Z_loss = yield_SNII + (1 - yield_SNII) * Pdata.at(iPart, IMETAL);
          Kokkos::atomic_add(&Uin.at(iCell, IRho_Z), rho_loss * Z_loss);
        }

        // Update particle properties
        Pdata.at(iPart, IMASS) -= Mloss;

        Kokkos::atomic_add(&N_supernovae_view(), num);

      }


      // If the SN will explode in this time step
      // if ((age_physical < t_SNII_physical) & ((age_physical + dt_physical) > t_SNII_physical)) {
      //   pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
      //   pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};

      //   ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

      //   pos_t cell_size = cells.getCellSize( iCell );
      //   real_t cell_volume = cell_size[IX] * cell_size[IY] * cell_size[IZ];

      //   // Compute ejecta mass, thermal energy + kinetic energy
      //   real_t Mstar = Pdata.at(iPart, IMASS);
      //   real_t Mloss = Mstar * eta_SNII;
      //   real_t rho_loss = Mloss / cell_volume;

      //   real_t ethermal = E_per_M_SNII * rho_loss;
      //   real_t ekin = 0.5 * rho_loss * (
      //     SQR(part_vel[IX]) + SQR(part_vel[IY]) + SQR(part_vel[IZ])
      //   );

      //   // Atomic are mandatory since multiple particles can explode in the same cell
      //   Kokkos::atomic_add(&Uin.at(iCell, IRho), rho_loss);
      //   Kokkos::atomic_add(&Uin.at(iCell, IE_tot), ethermal + ekin);
      //   Kokkos::atomic_add(&Uin.at(iCell, IRho_vx), rho_loss * part_vel[IX]);
      //   Kokkos::atomic_add(&Uin.at(iCell, IRho_vy), rho_loss * part_vel[IY]);
      //   Kokkos::atomic_add(&Uin.at(iCell, IRho_vz), rho_loss * part_vel[IZ]);
      //   if (has_metallicity) {
      //     real_t Z_loss = yield_SNII + (1 - yield_SNII) * Pdata.at(iPart, IMETAL);
      //     Kokkos::atomic_add(&Uin.at(iCell, IRho_Z), rho_loss * Z_loss);
      //   }

      //   // Update particle properties
      //   Pdata.at(iPart, IMASS) -= Mloss;

      //   Kokkos::atomic_add(&N_supernovae_view(), 1);

      // }
    });

    int N_supernovae = 0;
    Kokkos::deep_copy(N_supernovae, N_supernovae_view);

    if (N_supernovae > 0)
      std::cout << "Feedback: " << N_supernovae << " supernovae exploded" << std::endl;

    timers.get("ParticleUpdate_feedback").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;

  real_t eta_SNII;
  real_t yield_SNII;
  real_t E_SNII_physical;
  real_t M_SNII_physical;
  real_t t_SNII_physical;

  bool cosmology;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_feedback,
                  "ParticleUpdate_feedback")

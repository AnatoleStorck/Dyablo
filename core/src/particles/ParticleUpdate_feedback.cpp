#include "ParticleUpdate_base.h"
#include "utils/units/Units.h"
#include "ForeachParticle.h"
#include "states/State_hydro.h"

#include <Kokkos_Core.hpp>

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
    M_SNII_physical ( configMap.getValue_in_code_unit<Units::Mass>  ("star_feedback", "M_SNII", "10 Msun") ),
    t_SNII_physical ( configMap.getValue_in_code_unit<Units::Time>  ("star_feedback", "t_SNII", "10 Myr") ),
    cosmology       ( configMap.getValue<bool>("cosmology", "active") )
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
      IMASS, IVX, IVY, IVZ, IBIRTH, IMETAL
    };

    timers.get("ParticleUpdate_feedback").start();

    std::set<std::string> fields = {"rho_SN", "e_tot_SN", "rho_vx_SN", "rho_vy_SN", "rho_vz_SN"};
    std::set<std::string> pfields = {"rho", "e_tot", "rho_vx", "rho_vy", "rho_vz"};
    std::vector<UserData::FieldAccessor_FieldInfo>
      Uin_infos = {{"rho", IRho},    {"e_tot", IE_tot},    {"rho_vx", IRho_vx},    {"rho_vy", IRho_vy},    {"rho_vz", IRho_vz}},
      USN_infos = {{"rho_SN", IRho}, {"e_tot_SN", IE_tot}, {"rho_vx_SN", IRho_vx}, {"rho_vy_SN", IRho_vy}, {"rho_vz_SN", IRho_vz}};
    std::vector<UserData::ParticleAccessor_AttributeInfo>
      pinfos = {{"mass", IMASS}, {"vx", IVX}, {"vy", IVY}, {"vz", IVZ}, {"birth_time", IBIRTH}};

    bool has_metallicity = U.has_field("metallicity");
    if (has_metallicity) {
      fields.insert("metallicity_SN");
      pfields.insert("metallicity");

      Uin_infos.push_back( {"metallicity",    IRho_Z} );
      USN_infos.push_back( {"metallicity_SN", IRho_Z} );
      pinfos.push_back( {"metallicity", IMETAL} );
    }

    U.new_fields(fields);

    // Get accessors
    auto Ppos = U.getParticleArray( "particles" );
    auto Pdata = U.getParticleAccessor( "particles", pinfos );
    auto Uin = U.getAccessor( Uin_infos );
    auto USN = U.getAccessor( USN_infos );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t aexp = scalar_data.get<real_t>("aexp");

    // Gather SN feedback parameters
    const real_t eta_SNII = this->eta_SNII;
    const real_t yield_SNII = this->yield_SNII;
    const real_t E_SNII = Units::physical_to_supercomoving<Units::Energy>(E_SNII_physical, aexp);
    const real_t M_SNII = Units::physical_to_supercomoving<Units::Mass>(M_SNII_physical, aexp);
    const real_t E_per_M_SNII = E_SNII / M_SNII;
    const real_t t_SNII = Units::physical_to_supercomoving<Units::Time>(t_SNII_physical, aexp);

    foreach_particle.foreach_particle( "particles_update_feedback", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      // Age of the particle
      real_t age = t - Pdata.at(iPart, IBIRTH);

      // If the SN will explode in this time step
      if ((age < t_SNII) & ((age + dt) > t_SNII)) {
        pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
        pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};

        ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

        pos_t cell_size = cells.getCellSize( iCell );
        real_t cell_volume = cell_size[IX] * cell_size[IY] * cell_size[IZ];

        // Compute ejecta mass, thermal energy + kinetic energy
        real_t Mstar = Pdata.at(iPart, IMASS);
        real_t Mloss = Mstar * eta_SNII;
        real_t rho_loss = Mloss / cell_volume;

        real_t ethermal = E_per_M_SNII * rho_loss;
        real_t ekin = 0.5 * rho_loss * (
          SQR(part_vel[IX]) + SQR(part_vel[IY]) + SQR(part_vel[IZ])
        );

        // Atomic are mandatory since multiple particles can explode in the same cell
        Kokkos::atomic_add(&USN.at(iCell, IRho), rho_loss);
        Kokkos::atomic_add(&USN.at(iCell, IE_tot), ethermal + ekin);
        Kokkos::atomic_add(&USN.at(iCell, IRho_vx), rho_loss * part_vel[IX]);
        Kokkos::atomic_add(&USN.at(iCell, IRho_vy), rho_loss * part_vel[IY]);
        Kokkos::atomic_add(&USN.at(iCell, IRho_vz), rho_loss * part_vel[IZ]);
        if (has_metallicity) {
          real_t Z_loss = yield_SNII + (1 - yield_SNII) * Pdata.at(iPart, IMETAL);
          Kokkos::atomic_add(&USN.at(iCell, IRho_Z), rho_loss * Z_loss);
        }

        // Update particle properties
        Pdata.at(iPart, IMASS) -= Mloss;
      }
    });

    // Second pass, copy SN feedback yields to hydro fields
    foreach_cell.foreach_cell( "cells_update_feedback", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      // Note: we do not need atomic here since each cell is processed once
      Uin.at(iCell, IRho)    += USN.at(iCell, IRho);
      Uin.at(iCell, IE_tot)  += USN.at(iCell, IE_tot);
      Uin.at(iCell, IRho_vx) += USN.at(iCell, IRho_vx);
      Uin.at(iCell, IRho_vy) += USN.at(iCell, IRho_vy);
      Uin.at(iCell, IRho_vz) += USN.at(iCell, IRho_vz);
      if (has_metallicity)
        Uin.at(iCell, IRho_Z)  += USN.at(iCell, IRho_Z);
    });

    // Clean SN feedback yields
    for (const std::string& name : fields)
      U.delete_field(name);

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

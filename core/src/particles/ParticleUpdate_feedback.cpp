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
    gamma0          ( configMap.getValue<real_t>("hydro", "gamma0", 1.4) ),
    eta_SN          ( configMap.getValue<real_t>("star_feedback", "eta_SN", 0.1) ),
    E_SN_physical   ( configMap.getValue_in_code_unit<Units::Energy>("star_feedback", "energy_threshold", "1e51 erg") ),
    t_SN_physical   ( configMap.getValue_in_code_unit<Units::Time>("star_feedback", "time_delay", "10 Myr") )
  {
  }

  ~ParticleUpdate_feedback() {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {
    const real_t t = scalar_data.get<real_t>("time");
    const real_t dt = scalar_data.get<real_t>("dt");

    enum VarIndex {
      IRho, IE_tot, IRho_vx, IRho_vy, IRho_vz,
      // Updates from SN feedback
      IRho_SN, IE_tot_SN, IRho_vx_SN, IRho_vy_SN, IRho_vz_SN
    };
    enum VarIndex_particle {
      IMASS, IVX, IVY, IVZ, IBIRTH
    };

    timers.get("ParticleUpdate_feedback").start();

    U.new_fields({"rho_SN", "e_tot_SN", "rho_vx_SN", "rho_vy_SN", "rho_vz_SN"});

    auto Ppos = U.getParticleArray( "particles" );
    auto Pdata = U.getParticleAccessor( "particles", {{"mass", IMASS}, {"vx", IVX},{"vy", IVY},{"vz", IVZ}, {"birth_time", IBIRTH}} );

    // Accessor of hydro fields
    auto Uin = U.getAccessor({ {"rho", IRho}, {"e_tot", IE_tot}, {"rho_vx", IRho_vx}, {"rho_vy", IRho_vy}, {"rho_vz", IRho_vz} });
    // Accessor of SN feedback yields
    auto USN = U.getAccessor({ {"rho_SN", IRho_SN}, {"e_tot_SN", IE_tot_SN}, {"rho_vx_SN", IRho_vx_SN}, {"rho_vy_SN", IRho_vy_SN}, {"rho_vz_SN", IRho_vz_SN}});

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t aexp = scalar_data.get<real_t>("aexp");

    // Gather SN feedback parameters
    real_t eta_SN = this->eta_SN;
    real_t E_SN = Units::physical_to_supercomoving<Units::Energy>(E_SN_physical, aexp);
    real_t t_SN = Units::physical_to_supercomoving<Units::Time>(t_SN_physical, aexp);

    foreach_particle.foreach_particle( "particles_update_feedback", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      // Age of the particle
      // TODO: this is a difference of conformal times, should be converted to physical
      real_t age = t - Pdata.at(iPart, IBIRTH);

      // If the SN will explode in this time step
      if ((age < t_SN) & ((age + dt) > t_SN)) {
        pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
        pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};

        ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

        pos_t cell_size = cells.getCellSize( iCell );
        real_t cell_volume = cell_size[IX] * cell_size[IY] * cell_size[IZ];

        // Compute ejecta mass, thermal energy + kinetic energy
        real_t Mstar = Pdata.at(iPart, IMASS);
        real_t Mejecta = Mstar * eta_SN;
        real_t ethermal = E_SN / cell_volume;
        real_t rho_ejecta = Mejecta / cell_volume;
        real_t ekin = 0.5 * rho_ejecta * (
          SQR(part_vel[IX]) + SQR(part_vel[IY]) + SQR(part_vel[IZ])
        );

        // Atomic are mandatory since multiple particles can explode in the same cell
        Kokkos::atomic_add(&USN.at(iCell, IRho_SN), rho_ejecta);
        Kokkos::atomic_add(&USN.at(iCell, IE_tot_SN), ethermal + ekin);
        Kokkos::atomic_add(&USN.at(iCell, IRho_vx_SN), rho_ejecta * part_vel[IX]);
        Kokkos::atomic_add(&USN.at(iCell, IRho_vy_SN), rho_ejecta * part_vel[IY]);
        Kokkos::atomic_add(&USN.at(iCell, IRho_vz_SN), rho_ejecta * part_vel[IZ]);

        // Update particle properties
        Pdata.at(iPart, IMASS) -= Mejecta;
      }
    });

    // Second pass, copy SN feedback yields to hydro fields
    foreach_cell.foreach_cell( "cells_update_feedback", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      // Note: we do not need atomic here since each cell is processed once
      Uin.at(iCell, IRho)    += USN.at(iCell, IRho_SN);
      Uin.at(iCell, IE_tot)  += USN.at(iCell, IE_tot_SN);
      Uin.at(iCell, IRho_vx) += USN.at(iCell, IRho_vx_SN);
      Uin.at(iCell, IRho_vy) += USN.at(iCell, IRho_vy_SN);
      Uin.at(iCell, IRho_vz) += USN.at(iCell, IRho_vz_SN);
    });

    // Clean SN feedback yields
    U.delete_field("rho_SN");
    U.delete_field("e_tot_SN");
    U.delete_field("rho_vx_SN");
    U.delete_field("rho_vy_SN");
    U.delete_field("rho_vz_SN");

    timers.get("ParticleUpdate_feedback").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;
  real_t gamma0;

  real_t eta_SN;
  real_t E_SN_physical;
  real_t t_SN_physical;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_feedback,
                  "ParticleUpdate_feedback")

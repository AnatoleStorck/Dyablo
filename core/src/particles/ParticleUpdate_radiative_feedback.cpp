#include "ParticleUpdate_base.h"
#include "utils/units/Units.h"
#include "ForeachParticle.h"
#include "states/State_hydro.h"
#include "types.hpp"

#include <Kokkos_Core.hpp>

namespace dyablo {

class ParticleUpdate_radiative_feedback : public ParticleUpdate {
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  ParticleUpdate_radiative_feedback(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers)
  : foreach_cell    ( foreach_cell ),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    timers          ( timers ),
    photon_rate     ( configMap.getValue<real_t>("star_feedback", "photon_rate", 1e49) ),
    cosmology       ( configMap.getValue<bool>("cosmology", "active", false) )
  {
  }

  ~ParticleUpdate_radiative_feedback() {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {

    const real_t t = cosmology ? scalar_data.get<real_t>("time_physical") : scalar_data.get<real_t>("time");
    const real_t dt = scalar_data.get<real_t>("dt");

    enum VarIndex_rt {
      IE_rad
    };
    enum VarIndex_particle {
      IMASS, IBIRTH,
    };

    timers.get("ParticleUpdate_radiative_feedback").start();

    std::vector<UserData_fields::FieldAccessor_FieldInfo> Uout_infos;
    Uout_infos.reserve(N_GROUPS);
    for (int g = 0; g < N_GROUPS; ++g) {
      Uout_infos.push_back({"e_rad_" + std::to_string(g), g});
    }
    // std::vector<UserData_particles::ParticleAccessor_AttributeInfo>
    //   pinfos = {{"mass", IMASS}, {"birth_time", IBIRTH}};

    // Get accessors
    auto Ppos = U.getParticleArray( "particles" );
    // auto Pdata = U.getParticleAccessor( "particles", pinfos );
    auto Uout = U.getAccessor( Uout_infos );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t aexp = scalar_data.get<real_t>("aexp");

    // Gather SN feedback parameters
    const real_t photon_rate = this->photon_rate;
    const real_t photon_rate_group = photon_rate / N_GROUPS;
    const real_t dt_physical = Units::supercomoving_to_physical<Units::Time>(
      (dt * Units::code_units().getUnit<Units::Time>()).convert_to(Units::s()),
      aexp
    );
    const real_t code2cm3 = Units::supercomoving_to_physical<Units::Volume>(
      (1 * Units::code_units().getUnit<Units::Volume>()).convert_to(Units::cm3()),
      aexp
    );

    foreach_particle.foreach_particle( "particles_update_feedback", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      // Age of the particle
      // real_t age_physical = t - Pdata.at(iPart, IBIRTH);

      pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};

      ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

      pos_t cell_size = cells.getCellSize( iCell );
      real_t cell_volume = cell_size[IX] * cell_size[IY] * cell_size[IZ];

      const real_t cell_volume_physical = cell_volume * code2cm3;

      // Atomic are mandatory since multiple particles can explode in the same cell
      for (int g = 0; g < N_GROUPS; ++g) {
        Kokkos::atomic_add(&Uout.at(iCell, g), photon_rate_group * dt_physical / cell_volume_physical);
      }

    });

    timers.get("ParticleUpdate_radiative_feedback").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;

  real_t photon_rate;

  bool cosmology;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_radiative_feedback,
                  "ParticleUpdate_radiative_feedback")

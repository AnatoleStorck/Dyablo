#pragma once

#include "ParticleUpdate_base.h"

namespace dyablo {


class ParticleUpdate_tracers_move;
class ParticleUpdate_NGP_move;
class ParticleUpdate_NGP_density;
class ParticleUpdate_CIC_move;
class ParticleUpdate_CIC_density;
class ParticleUpdate_star_formation;

} //namespace dyablo 

#include "plugins_lib.h"

template<>
inline bool dyablo::ParticleUpdateFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  DECLARE_REGISTERED(dyablo::ParticleUpdate_tracers_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_NGP_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_NGP_density);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_CIC_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_CIC_density);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_star_formation);

  return true;
}

#pragma once

#include "ParticleUpdate_base.h"

namespace dyablo {


class ParticleUpdate_tracers_move;
class ParticleUpdate_NGP_move;
class ParticleUpdate_NGP_density;
class ParticleUpdate_CIC_move;
class ParticleUpdate_CIC_density;
class ParticleUpdate_TSC_move;
class ParticleUpdate_TSC_density;
class ParticleUpdate_star_formation;
class ParticleUpdate_star_formation_turbulent;

class ParticleUpdate_feedback;
class ParticleUpdate_radiative_feedback;

class ParticleUpdate_sink_formation;
class ParticleUpdate_sink_accretion;
class ParticleUpdate_sink_merging;

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
  DECLARE_REGISTERED(dyablo::ParticleUpdate_TSC_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_TSC_density);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_star_formation);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_star_formation_turbulent);

  DECLARE_REGISTERED(dyablo::ParticleUpdate_feedback);

  DECLARE_REGISTERED(dyablo::ParticleUpdate_radiative_feedback);

  DECLARE_REGISTERED(dyablo::ParticleUpdate_sink_formation);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_sink_accretion);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_sink_merging);

  return true;
}

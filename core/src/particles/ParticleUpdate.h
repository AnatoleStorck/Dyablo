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

class ParticleUpdate_feedback;

} //namespace dyablo 


template<>
inline bool dyablo::ParticleUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::ParticleUpdate_tracers_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_NGP_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_NGP_density);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_CIC_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_CIC_density);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_TSC_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_TSC_density);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_star_formation);

  DECLARE_REGISTERED(dyablo::ParticleUpdate_feedback);

  return true;
}

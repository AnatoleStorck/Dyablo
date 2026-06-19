#pragma once

#include <string>
#include <vector>

#include "utils/config/ConfigMap.h"

namespace dyablo {

/***
 * @brief Return the list of particle family names configured for this run.
 *
 * Particles are grouped into named families (e.g. "dark_matter", "star"), each
 * stored as its own ParticleArray in UserData. Routines that need to operate on
 * every particle (dt limiter, gravity density projection, position move) iterate
 * over this list; family-specific routines (e.g. stellar feedback) reference a
 * single family directly.
 *
 * Defaults to a single family named "particles".
 ***/
inline std::vector<std::string> getParticleFamilies( ConfigMap& configMap )
{
  return configMap.getValue<std::vector<std::string>>("particles", "families", {"particles"});
}

} // namespace dyablo

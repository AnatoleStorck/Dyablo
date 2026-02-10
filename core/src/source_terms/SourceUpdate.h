#pragma once

#include "source_terms/SourceUpdate_base.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

namespace dyablo {

class SourceUpdate_Cooling_FF;
class SourceUpdate_GLM;
class SourceUpdate_Photons_Beam;
class SourceUpdate_Photons_Stromgren;
class SourceUpdate_Ionization_Stromgren;

template< typename Policy >
class SourceUpdate_cooling_grackle_table;

} //namespace dyablo


template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::SourceUpdate_Cooling_FF);
  DECLARE_REGISTERED(dyablo::SourceUpdate_GLM);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Photons_Beam);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Photons_Stromgren);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Ionization_Stromgren);
  DECLARE_REGISTERED(dyablo::SourceUpdate_cooling_grackle_table<dyablo::HyperbolicPolicy_Hydro>);
  DECLARE_REGISTERED(dyablo::SourceUpdate_cooling_grackle_table<dyablo::HyperbolicPolicy_GLMMHD>);

  return true;
}

